import random

import torch
from torch import nn

from best_rq.config import Config
from best_rq.random_projection_quantizer import RandomProjectionQuantizer


class BestRqFramework(nn.Module):
    def __init__(self, config: Config, encoder: nn.Module):
        super().__init__()
        self.K = 4
        self.layer_norm = nn.LayerNorm(config.mel_filter_size)
        self.random_projection_quantizer = RandomProjectionQuantizer(config, self.K)
        self.encoder = encoder
        self.config = config
        self.out_linear = nn.Linear(config.hidden_size, config.num_code_books)
        self.num_time_steps = int(config.mask_time // (config.stride_time * self.K))

    def forward(self, input_values: torch.Tensor, input_lengths: torch.Tensor):
        """
        Args:
            input_values (torch.Tensor): with shape `(B, T, D)`
            input_lengths (torch.Tensor): with shape `(B)`

        Returns:

        """
        input_values, input_lengths = self.input_reduction(input_values, input_lengths)
        batch_size, num_steps, hidden_size = input_values.size()

        input_values = self.layer_norm(input_values)

        # Reshape to number of encoder out steps
        # Depend on the subsampling layer in the encoder
        input_values = input_values.view(batch_size, -1, hidden_size * self.K)
        quantized_input_lengths = input_lengths // self.K - 1

        masked_input_values, time_mask_indices = self.masking(input_values.clone(), quantized_input_lengths)
        masked_input_values = masked_input_values.view(batch_size, num_steps, hidden_size)

        labels = self.random_projection_quantizer(input_values[:, :-1], time_mask_indices)

        encoder_out, _ = self.encoder(masked_input_values, input_lengths)

        targets = encoder_out[time_mask_indices]
        targets_out = self.out_linear(targets)

        return targets_out, labels

    def masking(self, input_values: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_values (torch.Tensor): with shape `(B, L, D)`
            input_lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            torch.Tensor with shape `(B, L, D)`
            torch.Tensor with shape `(B, L)`
            )
        """
        batch_size, num_steps, hidden_size = input_values.size()

        # non mask: 0, maks: 1
        time_mask_indices = torch.zeros(
            batch_size, num_steps + self.num_time_steps,
            device=input_values.device, dtype=torch.bool
        )

        for batch in range(batch_size):
            time_mask_idx_candidates = list(range(int(input_lengths[batch])))
            k = int(self.config.mask_prob * input_lengths[batch])
            start_time_mask_idx_array = torch.tensor(
                random.sample(time_mask_idx_candidates, k=k), device=input_values.device, dtype=torch.long
            )

            for i in range(self.num_time_steps):
                time_mask_indices[batch, start_time_mask_idx_array+i] = 1

        time_mask_indices = time_mask_indices[:, :-self.num_time_steps]
        num_masks = sum(time_mask_indices.flatten())

        # Replace to random value where mask
        random_values = torch.normal(mean=0, std=0.1, size=(num_masks, hidden_size), device=input_values.device)
        input_values[time_mask_indices == 1] = random_values

        return input_values, time_mask_indices[:, :-1]

    def input_reduction(self, inputs, input_lengths):
        if inputs.size(1) % self.K != 0:
            fixed_length = inputs.size(1) - inputs.size(1) % self.K
            inputs = inputs[:, :fixed_length]
            input_lengths[fixed_length < input_lengths] = fixed_length

        return inputs, input_lengths
