import numpy as np
import torch
from torch import nn

from pooling_layer import Config


class SelfAttentionPooling(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.weight_linear = nn.Linear(config.hidden_size // config.pooling_num_heads, 1)
        self.pooling_size = config.pooling_hidden_size
        self.num_head = config.pooling_num_heads

    def forward(self, hidden_states: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = hidden_states.size()

        hidden_states = self.linear(hidden_states).view(
            batch_size, -1, self.num_head, self.pooling_size // self.num_head
        ).transpose(-2, -3)

        attention_mask = torch.tensor(
            [range(length) for _ in range(batch_size)]
        ).to(hidden_states.device)
        attention_mask = torch.as_tensor(attention_mask < input_lengths.unsqueeze(1), device=hidden_states.device)
        attention_mask = attention_mask.unsqueeze(-2)

        # `(B, N, L)`
        weight = self.weight_linear(hidden_states).squeeze(-1)
        weight = weight.masked_fill(attention_mask == 0, torch.finfo(weight.dtype).min)
        weight = torch.softmax(weight, dim=-1)

        out = torch.sum(hidden_states * weight.unsqueeze(-1), dim=-2)
        out = out.view(batch_size, self.pooling_size)
        return out
