import json

import torch
from torch.nn.functional import cross_entropy

from pre_processing import PreProcessing
from speaker_identification import Config, SpeakerIdentificationModel

if __name__ == '__main__':
    # Dummy data
    wavs = [torch.rand(250000), torch.rand(10000), torch.rand(320000)]
    labels = torch.tensor([128, 982, 54])

    with open("standard_config.json", "r", encoding="utf-8") as f:
        config = Config(**json.load(f))

    pre_processing = PreProcessing(config)
    model = SpeakerIdentificationModel(config)

    fixed_wavs = []
    input_lengths = []

    # to fixed
    for wav in wavs:
        fixed_wav, input_length = pre_processing(wav)
        fixed_wavs.append(fixed_wav)
        input_lengths.append(input_length)

    inputs = torch.stack(fixed_wavs, dim=0)
    input_lengths = torch.tensor(input_lengths)

    out = model(inputs, input_lengths, labels)

    loss = cross_entropy(out, labels)

    print(f"Loss: {loss}")
