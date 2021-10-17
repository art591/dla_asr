from torch.utils.data import Sampler
import numpy as np


class GroupLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, batches_per_group=20):
        super().__init__(data_source)
        print(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.batches_per_group = batches_per_group

    def __iter__(self):
        idxs = np.arange(len(self.data_source))
        idxs = np.array(sorted(idxs, key=lambda x : self.data_source[x]['spectrogram'].shape[2]))
        for i in range(idxs.shape[0] / self.batch_size + 1):
            yield idxs[i * self.batch_size:(i + 1) * self.batch_size]

    def __len__(self):
        return len(self.data_source)
