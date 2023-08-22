import torch
from torch import nn
from torch.nn.init import kaiming_uniform_

from speaker_identification import Config


class FCLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.linear1 = nn.Linear(config.pooling_hidden_size, config.embedding_size)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout_probability)

        self.linear2 = nn.Linear(config.embedding_size, config.embedding_size)
        self.relu2 = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(config.embedding_size)

        kaiming_uniform_(self.linear1.weight)
        kaiming_uniform_(self.linear2.weight)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.relu1(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.linear2(hidden_states)
        hidden_states = self.relu2(hidden_states)
        embeddings = self.batch_norm(hidden_states)

        return embeddings
