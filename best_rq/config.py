from dataclasses import dataclass

from conformer import Config as ConformerConfig
from pre_processing import Config as PreProcessingConfig


@dataclass(frozen=True)
class Config(ConformerConfig, PreProcessingConfig):
    mask_prob: float  # 0.0 - 1.0 (Default: 0.05)
    mask_time: float  # Mask time sec (Default: 0.2)
    mel_filter_size: int  # Dimension of input.
    stride_time: float  # stride_time sec.
    code_book_size: int  # Dimension of code book (Default: 16)
    num_code_books: int  # Number of code books (Default: 8192)
    hidden_size: int  # Number of encoder output dimensions
