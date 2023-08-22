import torch
from torch import nn

from conformer.config import Config
from conformer.conformer_encoder import ConformerEncoder
from conformer.conformer_subsampling import ConvSubsampling


class ConformerModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.subsampling_conv = ConvSubsampling(config)
        self.encoder = ConformerEncoder(config)

    def forward(self, input_values: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_values (torch.Tensor): with shape `(B, T, D1)`
            input_lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            torch.Tensor with shape `(B, L, D)`
            torch.Tensor with shape `(B)`
            )
        """
        hidden_states, input_lengths = self.subsampling_conv(input_values, input_lengths)

        batch_size, length, _ = hidden_states.size()
        range_tensor = \
            torch.cat([torch.arange(length).unsqueeze(0) for _ in range(batch_size)], dim=0).to(hidden_states.device)
        attention_mask = torch.as_tensor(range_tensor < input_lengths.unsqueeze(1), device=hidden_states.device)

        hidden_states = self.encoder(hidden_states, attention_mask)

        return hidden_states, input_lengths
