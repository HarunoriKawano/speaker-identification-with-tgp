from dataclasses import dataclass

from conformer import Config as ConformerConfig
from pooling_layer import Config as PoolingConfig
from pre_processing import Config as PreProcessingConfig


@dataclass(frozen=True)
class Config(ConformerConfig, PoolingConfig, PreProcessingConfig):
    embedding_size: int
    factor_size: int
    num_speakers: int  # Number of speakers
    with_aam_softmax: bool

    def __post_init__(self):
        assert (self.fixed_time + (self.win_time - self.stride_time)) // (self.stride_time * 4) - 1 == self.max_length
