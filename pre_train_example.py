import json

import torch
from torch.nn.functional import cross_entropy

from pre_processing import PreProcessing
from best_rq import Config as BestRqConfig, BestRqWithConformer

if __name__ == '__main__':
    # Dummy data
    wavs = [torch.rand(250000), torch.rand(10000), torch.rand(320000)]

    with open("pre_train_config.json", "r", encoding="utf-8") as f:
        config = BestRqConfig(**json.load(f))

    pre_processing = PreProcessing(config)
    model = BestRqWithConformer(config)

    fixed_wavs = []
    input_lengths = []

    # to fixed
    for wav in wavs:
        fixed_wav, input_length = pre_processing(wav)
        fixed_wavs.append(fixed_wav)
        input_lengths.append(input_length)

    inputs = torch.stack(fixed_wavs, dim=0)
    input_lengths = torch.tensor(input_lengths)

    probs, labels = model(inputs, input_lengths)

    loss = cross_entropy(probs, labels)

    print(f"Num labels: {len(labels)}")
    print(f"Loss: {loss}")
