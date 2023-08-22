import torch, math
import torch.nn as nn
import torch.nn.functional as F

from speaker_identification import Config


class AAMSoftmax(nn.Module):
    def __init__(self, config: Config, m=0.2, s=30.0):
        super().__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(config.num_speakers, config.factor_size), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        out = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        out = out * self.s

        return out
