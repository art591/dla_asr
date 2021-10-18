import torchaudio
from torch import Tensor


from hw_asr.augmentations.base import AugmentationBase
import random
import numpy as np


class TimeStretch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        if random.random() < 0.5:
            rate = np.random.uniform(0.8, 1.25)
            x = data.unsqueeze(1)
            return self._aug(x, rate).squeeze(1)
        else:
            return data