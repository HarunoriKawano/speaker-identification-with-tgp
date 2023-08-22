import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from pooling_layer import Config


class TemporalGatePooling(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_head = config.pooling_num_heads
        self.filter_size = config.filter_size
        self.pooling_size = config.pooling_hidden_size

        self.f_linear = nn.Linear(config.hidden_size, config.filter_size)
        self.v_linear = nn.Linear(config.hidden_size, config.pooling_hidden_size)

        self.timewise_linear = nn.Linear(config.max_length, config.max_length)
        self.layer_norm = nn.LayerNorm(config.filter_size // self.num_head)

        self.g_linear = nn.Linear(config.filter_size // self.num_head, 1)

        xavier_uniform_(self.f_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        nn.init.ones_(self.timewise_linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D1)`
        Returns:
            tuple(
                torch.Tensor with shape `(B, L/S, D2)`
            )
        """
        batch_size, length, hidden_size = hidden_states.size()

        # `(B, L/S, S, D1)` -> `(B, L/S, H, S, D2/H)`
        filters = self.f_linear(hidden_states).view(
            batch_size, -1, self.num_head, self.filter_size // self.num_head
        ).transpose(-2, -3)
        values = self.v_linear(hidden_states).view(
            batch_size, -1, self.num_head, self.pooling_size // self.num_head
        ).transpose(-2, -3)

        # Extract time unit features
        filters = self.timewise_linear(filters.transpose(-1, -2)).transpose(-1, -2)
        filters = self.layer_norm(filters)

        gates = self.g_linear(filters)
        gates = torch.sigmoid(gates)

        out = torch.sum(values * gates, dim=-2).view(batch_size, -1)

        return out
