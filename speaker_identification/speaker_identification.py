from typing import Optional

import torch
from torch import nn

from pre_processing import WavToMel
from conformer import ConformerModel
from pooling_layer import PoolingLayer
from speaker_identification.fc_layer import FCLayer
from speaker_identification.classifier import Classifier
from speaker_identification import Config


class SpeakerIdentificationModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.wav_to_mel = WavToMel(config)
        self.encoder = ConformerModel(config)
        self.pooling_layer = PoolingLayer(config)
        self.fc_layer = FCLayer(config)
        self.classifier = Classifier(config)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor, labels: Optional[torch.Tensor] = None):
        with torch.no_grad():
            mel_spectrogram, input_lengths = self.wav_to_mel(inputs, input_lengths)

        hidden_states, input_lengths = self.encoder(mel_spectrogram, input_lengths)
        hidden_states = self.pooling_layer(hidden_states, input_lengths)

        embeddings = self.fc_layer(hidden_states)
        out = self.classifier(embeddings, labels)

        return out
