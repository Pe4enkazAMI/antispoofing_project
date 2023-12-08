from .cnn_blocks.man import MovingAvgNorm
from .cnn_blocks.mfm import MFM
import torch.nn as nn
import torch

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class LightCNN(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self._filtersizes = kwargs.get('filtersizes', [3, 3, 3, 3, 3])
        self._filter = [1] + kwargs.get('filter', [16, 24, 32, 16, 16])
        self._pooling = kwargs.get('pooling', [2, 2, 2, 2, 2])
        self._linear_dim = kwargs.get('lineardim', 128)
        self._cmvn = kwargs.get(
            'cmvn', True)
        self.norm = MovingAvgNorm(80) if self._cmvn else nn.Sequential()
        net = nn.ModuleList()
        for nl, (h0, h1, filtersize, poolingsize) in enumerate(
                zip(self._filter, self._filter[1:], self._filtersizes,
                    self._pooling)):
            if nl == 0:
                net.append(
                    nn.Sequential(
                        nn.BatchNorm2d(h0),
                        nn.Conv2d(h0,
                                  h1 * 2,
                                  kernel_size=filtersize,
                                  padding=filtersize // 2,
                                  stride=1),
                        MFM(1),
                    ))

            else:
                net.append(
                    nn.Sequential(
                        nn.BatchNorm2d(h0),
                        nn.Conv2d(h0,
                                  h1 * 2,
                                  kernel_size=1,
                                  padding=0,
                                  stride=1),
                        MFM(1),
                        nn.BatchNorm2d(h1),
                        nn.Conv2d(h1,
                                  h1 * 2,
                                  kernel_size=filtersize,
                                  padding=filtersize // 2,
                                  stride=1),
                        MFM(1),
                    ))
            net.append(nn.MaxPool2d(kernel_size=poolingsize, ceil_mode=True))
        self.network = nn.Sequential(*net)
        with torch.no_grad():
            feature_output = self.network(torch.randn(1, 1, 300,
                                                      inputdim)).shape
            feature_output = feature_output[1] * feature_output[3]

        self.timepool = nn.AdaptiveAvgPool2d((1, None))
        self.outputlayer = nn.Sequential(
            nn.Conv1d(feature_output, self._linear_dim * 2, kernel_size=1),
            MFM(1), nn.Dropout(0.5),
            nn.Conv1d(self._linear_dim, outputdim, kernel_size=1, groups=1))

        self.network.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x):
        x = self.norm(x)
        x = x.unsqueeze(1)
        x = self.network(x)
        x = self.timepool(x).permute(0, 2, 1,3).contiguous().flatten(-2).permute(0, 2, 1).contiguous()
        return self.outputlayer(x).squeeze(-1)