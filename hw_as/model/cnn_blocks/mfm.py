import torch
import torch.nn as nn

class MFM(nn.Module):
    """
    Max Feature Map (MFM) Activation:

    Takes in a tensor of size 
    :math:`(N, C_{\text{in}}, H, W)` 
    and output 
    :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    by calculating:
    :math:`max((N, C_{\text{out}}1//2, H_{\text{out}}, W_{\text{out}}), 
    :math:`(N, C_{\text{out}}2//2, H_{\text{out}}, W_{\text{out}})`)`
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        low, up = torch.chunk(x, 2, self.dim)
        return torch.max(low, up)