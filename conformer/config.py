from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    hidden_size: int  # Dimension of encoder hidden states (Default: 512)
    intermediate_size: int  # Dimension of feed forward hidden states (Default: 2048)
    num_attention_heads: int  # Number of self attention heads (Default: 8)
    num_hidden_layers: int  # Number of Conformer blocks (Default 17)
    max_length: int  # Maximum input length of encoder (Default 374(15s))
    mel_filter_size: int  # Number of mel filter banks. (Default: 80)
    dropout_probability: float  # Probability of dropouts. (Default: 0.1)

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0
