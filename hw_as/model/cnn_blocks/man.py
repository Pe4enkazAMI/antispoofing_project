import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingAvgNorm(nn.Module):
    """
    Calculates multiple moving average estiamtes given a kernel_size
    Similar to kaldi's apply-cmvn-sliding 
    """
    def __init__(self, kernel_size=101, with_mean=True, with_std=True):
        super().__init__()
        self.register_buffer('kernel_size', torch.tensor(kernel_size))
        self.register_buffer('with_mean', torch.tensor(with_mean))
        self.register_buffer('with_std', torch.tensor(with_std))
        self.register_buffer('eps', torch.tensor(1e-12))

    def forward(self, x):
        assert x.ndim == 3, "Input needs to be tensor of shape B x T x D"
        n_batch, timedim, featdim = x.shape
        with torch.no_grad():
            # Too small utterance, just normalize per time-step
            if timedim < self.kernel_size:
                v, m = torch.std_mean(x, dim=1, keepdims=True)
                return (x - m) / (v + self.eps)
            else:
                sliding_window = F.pad(
                    x.transpose(1, 2),
                    (self.kernel_size // 2, self.kernel_size // 2 - 1),
                    mode='reflect').unfold(-1, self.kernel_size,
                                           1).transpose(1, 2)
            if self.with_mean and self.with_std:
                v, m = torch.std_mean(sliding_window, dim=-1)
                return (x - m) / (v + self.eps)
            elif self.with_mean:
                m = sliding_window.mean(-1)
                return (x - m)
            elif self.with_std:
                v = sliding_window.std(-1)
                return x / (v + self.eps)
            else:
                return x  # no normalization done