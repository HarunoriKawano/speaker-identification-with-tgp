from typing import Literal, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    pooling_type: Literal["tgp", "max", "mean", "mean_std", "random", "self-attention"]
    pooling_hidden_size: int
    max_length: Optional[int]
    hidden_size: int
    pooling_num_heads: Optional[int]
    filter_size: Optional[int]

    def __post_init__(self):
        if self.pooling_type == "tgp":
            assert self.pooling_num_heads
            assert self.filter_size
            assert self.max_length
            assert self.pooling_hidden_size % self.pooling_num_heads == 0
            assert self.filter_size % self.pooling_num_heads == 0
        elif self.pooling_type == "self-attention":
            assert self.pooling_num_heads
            assert self.pooling_hidden_size % self.pooling_num_heads == 0
