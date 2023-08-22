import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram

from pre_processing import Config


class WavToMel(nn.Module):
    def __init__(
            self,
            config: Config,
            noise_scale: float = 1e-4
    ):
        super().__init__()
        self.noise_scale = noise_scale

        self.mel_sampler = MelSpectrogram(
            sample_rate=config.sampling_rate,
            win_length=int(config.sampling_rate * config.win_time),
            hop_length=int(config.sampling_rate * config.stride_time),
            n_fft=config.n_fft,
            n_mels=config.mel_filter_size
        )

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs (torch.Tensor): with shape `(B, T)`
            input_lengths (torch.Tensor): with shape `(B)`

        Returns:
            torch.Tensor with shape `(T, D)` or `(B, T, D)`

        """
        # Add noise for log scaling
        noise = torch.randn(inputs.size(), device=inputs.device) * self.noise_scale
        inputs += noise

        mel_feature = self.mel_sampler(inputs)
        log_mel_feature = mel_feature.log2()

        log_mel_feature = log_mel_feature.transpose(-1, -2)

        input_lengths = input_lengths // (inputs.size(-1) / log_mel_feature.size(-2))

        return log_mel_feature, input_lengths
