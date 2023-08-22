from typing import Optional

import torch
from torchaudio.functional import resample

from pre_processing import Config


class PreProcessing:
    def __init__(self, config: Config):
        self.max_sampling_num = config.fixed_time * config.sampling_rate
        self.sampling_rate = config.sampling_rate

    def __call__(self, wav: torch.Tensor, original_sampling_rate: Optional[int] = None) -> tuple[torch.Tensor, int]:
        if original_sampling_rate is not None and original_sampling_rate != self.sampling_rate:
            wav = resample(wav, original_sampling_rate, self.sampling_rate)

        wav = wav.squeeze()
        input_length = wav.size(0)
        if wav.size(0) < self.max_sampling_num:
            wav = torch.cat((wav, torch.zeros(self.max_sampling_num - wav.size(0), device=wav.device)), dim=0)
        elif self.max_sampling_num < wav.size(0):
            wav = wav[:self.max_sampling_num]
            input_length = self.max_sampling_num

        return wav, input_length
