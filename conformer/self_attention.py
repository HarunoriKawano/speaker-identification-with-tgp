from typing import Optional
import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from conformer.config import Config


class SelfAttentionModule(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.self_attn = MultiHeadSelfAttentionWithRelativePosition(config)
        self.dropout = nn.Dropout(p=config.dropout_probability)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`
            position_embeddings (torch.Tensor): with shape `(B, L, D)`
            attention_mask (torch.Tensor): with shape `(B, L)`

        Returns:
            torch.Tensor with shape`(B, L, D)`
        """
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings
        )
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class MultiHeadSelfAttentionWithRelativePosition(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.head_size = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(p=config.dropout_probability)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

        self.linear_pos = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

        xavier_uniform_(self.linear_q.weight)
        xavier_uniform_(self.linear_k.weight)
        xavier_uniform_(self.linear_v.weight)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`
            position_embeddings (torch.Tensor): with shape `(B, L, D)`
            attention_mask (torch.Tensor): with shape `(B, L)`

        Returns:
            torch.Tensor with shape`(B, L, D)`
        """
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # `(B, L, D)` -> `(B, L, H, D/H)`
        query = self.linear_q(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)

        # `(B, L, H, D/H)` -> `(B, L, H, D/H)`
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = self._apply_relative_embeddings(
            query=query, key=key, relative_position_embeddings=position_embeddings
        )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        hidden_states = torch.matmul(probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        out = self.linear_out(hidden_states)

        return out

    def _apply_relative_embeddings(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            relative_position_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate attention weight with relative position by Skew algorythm.

        Args:
            query (torch.Tensor): with shape `(B, H, L, D/H)`
            key: (torch.Tensor): with shape `(B, H, L, D/H)`
            relative_position_embeddings (torch.Tensor): with shape `(1, L, D)`

        Returns:
            torch.Tensor with shape `(B, H, L, L)`

        """

        # `(B, L, D)` -> `(B, H, D/H, L)`
        proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(
            relative_position_embeddings.size(0), -1, self.num_heads, self.head_size
        )
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3)

        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

        scores_bd = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)

        # Skew algorythm
        triangle_mask = torch.tril(torch.ones(scores_bd.shape, device=scores_bd.device)).flip(dims=[-1])
        scores_bd = scores_bd.masked_fill(triangle_mask == 0, value=0)
        padding = torch.zeros(scores_bd.size(0), scores_bd.size(1), scores_bd.size(2), 1, device=scores_bd.device)
        scores_bd = torch.cat([padding, scores_bd], dim=-1)
        scores_bd = scores_bd.view(scores_bd.size(0), scores_bd.size(1), scores_bd.size(3), scores_bd.size(2))
        scores_bd = scores_bd[:, :, 1:]

        scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)

        return scores


class PositionalEncoder(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.position_encoder = nn.Embedding(config.max_length, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`

        Returns:
            torch.Tensor: with shape `(1, L, D)`
        """
        max_length = hidden_states.size(1)
        position_ids = torch.arange(0, max_length, 1).to(hidden_states.device).flip(dims=[0])
        position_embeddings = self.position_encoder(position_ids).unsqueeze(0)

        return position_embeddings
