from typing import Optional

import torch
from torch import nn

from speaker_identification import Config
from speaker_identification.aamsoftmax import AAMSoftmax


class Classifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.linear = nn.Linear(config.embedding_size, config.factor_size)
        self.batch_norm = nn.BatchNorm1d(config.factor_size)
        self.with_aam_softmax = config.with_aam_softmax
        if config.with_aam_softmax:
            self.aam_softmax = AAMSoftmax(config)
        else:
            self.out_linear = nn.Linear(config.factor_size, config.num_speakers)

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: Optional[torch.Tensor] = None
    ):
        hidden_states = self.linear(embeddings)
        hidden_states = self.batch_norm(hidden_states)

        if self.with_aam_softmax:
            out = self.aam_softmax(hidden_states, labels)
        else:
            out = self.out_linear(hidden_states)

        return out
