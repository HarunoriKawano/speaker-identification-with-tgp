import numpy as np
import torch
from torch import nn

from pooling_layer import Config


class StatisticalPooling(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.pooling_type = config.pooling_type
        if self.pooling_type == "mean_std":
            self.linear = nn.Linear(config.hidden_size, config.pooling_hidden_size // 2)
        else:
            self.linear = nn.Linear(config.hidden_size, config.pooling_hidden_size)

    def forward(self, hidden_states: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = hidden_states.size()

        hidden_states = self.linear(hidden_states)

        attention_mask = torch.tensor(
            [range(length) for _ in range(batch_size)]
        ).to(hidden_states.device)
        attention_mask = torch.as_tensor(attention_mask < input_lengths.unsqueeze(1), device=hidden_states.device)
        attention_mask = attention_mask.unsqueeze(-1)

        if self.pooling_type == "max":
            hidden_states = hidden_states.masked_fill(attention_mask == 0, value=torch.finfo(hidden_states.dtype).min)
            out = hidden_states.max(-2)[0]
            return out

        elif self.pooling_type == "mean":
            hidden_states = hidden_states.masked_fill(attention_mask == 0, value=0.0)
            out = hidden_states.sum(-2) / input_lengths.view(-1, 1)
            return out

        elif self.pooling_type == "mean_std":
            mean_stds = []
            for input_length, hidden_state in zip(input_lengths, hidden_states):
                mean_std = torch.cat(torch.std_mean(hidden_state[:int(input_length.item())], dim=0))
                mean_stds.append(mean_std)

            out = torch.stack(mean_stds, dim=0)
            return out

        elif self.pooling_type == "random":
            chosen_idxes = [np.random.choice(int(length), 1)[0] for length in input_lengths.cpu().tolist()]
            chosen_tensors_list = []
            for chosen_idx, hidden_state in zip(chosen_idxes, hidden_states):
                chosen_tensors_list.append(hidden_state[chosen_idx])
            out = torch.stack(chosen_tensors_list, dim=0)
            return out
