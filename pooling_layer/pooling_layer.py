import torch
from torch import nn

from pooling_layer import Config
from pooling_layer.statistical_pooling import StatisticalPooling
from pooling_layer.self_attention_pooling import SelfAttentionPooling
from pooling_layer.temporal_gate_pooling import TemporalGatePooling


class PoolingLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.pooling_type = config.pooling_type
        if self.pooling_type == "tgp":
            self.pooling = TemporalGatePooling(config)
        elif self.pooling_type == "self-attention":
            self.pooling = SelfAttentionPooling(config)
        else:
            self.pooling = StatisticalPooling(config)

    def forward(self, hidden_states: torch.Tensor, input_lengths: torch.Tensor):
        if self.pooling_type == "tgp":
            out = self.pooling(hidden_states)
        else:
            out = self.pooling(hidden_states, input_lengths)

        return out
