import torch
from torch import Tensor
import numpy as np
import random

from hw_asr.augmentations.base import AugmentationBase


class Noise(AugmentationBase):
    def __init__(self, min_amplitude, max_amplitude, *args, **kwargs):
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        if random.random() < 0.5:
            amplitude = np.random.uniform(self.min_amplitude, self.max_amplitude)
            x = x + amplitude * torch.normal(mean=0, std=1, size=x.shape)
        return x.squeeze(1)
