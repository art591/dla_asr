import torchaudio
from torch import Tensor


from hw_asr.augmentations.base import AugmentationBase
import random


class TimeMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.TimeMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        if random.random() < 0.5:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        else:
            return data