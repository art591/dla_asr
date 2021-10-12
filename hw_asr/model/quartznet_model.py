from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel



class TCSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, activation=nn.ReLU, separable=True):
        super().__init__()
        tcsconv = []
        if separable:
            tcsconv += [
                nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', groups=in_channels, dilation=dilation),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same', dilation=dilation)
            ]
        else:
            tcsconv += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', dilation=dilation)
            ]
        tcsconv.append(nn.BatchNorm1d(out_channels))
        if activation is not None:
            tcsconv.append(activation())
        self.tcsconv = nn.Sequential(*tcsconv)

    def forward(self, x):
        return self.tcsconv(x)

class ResidualTCSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_blocks, activation=nn.ReLU):
        super().__init__()
        self.num_blocks = num_blocks
        self.layers = []
        for i in range(num_blocks):
            if i + 1 == num_blocks:
                self.layers.append(TCSConv(in_channels, out_channels, kernel_size, activation=None))
                continue
            self.layers.append(TCSConv(in_channels, in_channels, kernel_size, activation=activation))
        self.layers = nn.Sequential(*self.layers)
        self.res_block = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
                                       nn.BatchNorm1d(out_channels))
        self.last_activation = activation()

    def forward(self, x):
        y = self.layers(x)
        return self.last_activation(self.res_block(x) + y)


class Quartznet(BaseModel):
    def __init__(self, n_feats, n_class, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        super().__init__()
        self.model = nn.Sequential(*
                        [
                            nn.Conv1d(n_feats, out_channels=256, kernel_size=33, stride=2, padding=16),
                            ResidualTCSConvBlock(256, 256, 33, 5),
                            ResidualTCSConvBlock(256, 256, 39, 5),
                            ResidualTCSConvBlock(256, 512, 51, 5),
                            ResidualTCSConvBlock(512, 512, 63, 5),
                            ResidualTCSConvBlock(512, 512, 75, 5),
                            TCSConv(512, 512, 87, dilation=2),
                            TCSConv(512, 1024, 1, separable=False),
                            nn.Conv1d(1024, n_class, kernel_size=1, padding='same')
                        ]
                    )

    def forward(self, spectrogram, *args, **kwargs):
        return self.model(spectrogram)

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
