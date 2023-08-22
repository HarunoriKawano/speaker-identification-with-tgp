from conformer import ConformerModel
from best_rq import Config
from best_rq.best_rq_framework import BestRqFramework
from pre_processing import WavToMel

import torch
from torch import nn


class BestRqWithConformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.wav_to_mel = WavToMel(config)
        encoder = ConformerModel(config)
        self.best_rq = BestRqFramework(config, encoder)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        with torch.no_grad():
            mel_spectrogram, input_lengths = self.wav_to_mel(inputs, input_lengths)
        out, labels = self.best_rq(mel_spectrogram, input_lengths)

        return out, labels


