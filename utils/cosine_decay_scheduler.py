import math


class CosineDecayScheduler:
    def __init__(self, warmup_steps: int, max_steps: int, max_warmup_steps: int = 10000):
        if max_warmup_steps < warmup_steps:
            self._warmup_steps = max_warmup_steps
        else:
            self._warmup_steps = warmup_steps
        self._max_steps = max_steps

    def __call__(self, epoch: int):
        epoch = max(epoch, 1)
        if epoch <= self._warmup_steps:
            return epoch / self._warmup_steps
        epoch -= 1
        rad = math.pi * epoch / self._max_steps
        weight = (math.cos(rad) + 1.) / 2
        return weight
