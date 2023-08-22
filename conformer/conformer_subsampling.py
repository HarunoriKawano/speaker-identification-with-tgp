import torch
from torch import nn

from conformer.config import Config


class ConvSubsampling(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.conv1 = nn.Conv2d(1, config.hidden_size // 4, kernel_size=3, stride=2)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(config.hidden_size // 4, config.hidden_size // 4, kernel_size=3, stride=2)
        self.activation2 = nn.ReLU()

    def forward(self, input_values: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_values (torch.Tensor): with shape `(B, T, D1)`
            input_lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            torch.Tensor with shape `(B, L, D2)`
            torch.Tensor with shape `(B)`
            )
        """

        hidden_states = self.conv1(input_values.unsqueeze(1))
        hidden_states = self.activation1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.activation2(hidden_states)

        batch_size, _, sub_sampled_lengths, _ = hidden_states.size()

        hidden_states = hidden_states.permute(0, 2, 1, 3)
        outputs = hidden_states.contiguous().view(batch_size, sub_sampled_lengths, -1)

        input_lengths //= 4
        input_lengths -= 1

        return outputs, input_lengths
