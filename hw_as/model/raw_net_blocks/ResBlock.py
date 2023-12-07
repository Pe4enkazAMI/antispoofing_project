import torch
import torch.nn as nn
import torch.nn.functional as F


class FRM(nn.Module):
    def __init__(self, frm_dim, add=True, mult=True) -> None:
        super().__init__()
        self.linear = nn.Linear(frm_dim, frm_dim)
        self.sigmoid = nn.Sigmoid()
        self.add = add
        self.mult = mult
    def forward(self, x):
        out = F.adaptive_avg_pool1d(x, 1).reshape(x.shape[0], -1)
        out = self.sigmoid(self.linear(out)).reshape(x.shape[0], x.shape[1], -1)

        if self.add:
            x = x + out
        if self.mult:
            x = x * out
        return x

class ResBlock(nn.Module):
    def __init__(self, emb_dims, first=False) -> None:
        super().__init__()
        self.first = first
        blocks = []
        if not self.first:
            blocks.append(nn.BatchNorm1d(num_features=emb_dims[0]))
            blocks.append(nn.LeakyReLU(0.3))

        
        blocks.append(nn.Conv1d(
            in_channels=emb_dims[0],
            out_channels=emb_dims[1],
            kernel_size=3,
            padding=1,
            stride=1
        ))
        blocks.append(nn.BatchNorm1d(num_features=emb_dims[1]))
        blocks.append(nn.LeakyReLU(0.3))
        blocks.append(nn.Conv1d(
            in_channels = emb_dims[1],
			out_channels = emb_dims[1],
			padding = 1,
			kernel_size = 3,
			stride = 1    
        ))
        if emb_dims[0] != emb_dims[1]:
            self.flag = True
            self.downsample = nn.Conv1d(in_channels=emb_dims[0],
                                         out_channels=emb_dims[1], 
                                         padding=0, 
                                         kernel_size=1, 
                                         stride=1)
        else:
            self.flag = False

        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.MaxPool1d(3)
        self.frm = FRM(frm_dim=emb_dims[1], add=True, mult=True)

    def forward(self, x):
        out = self.blocks(x)

        if self.flag:
            x = self.downsample(x)

        
        out = out + x
        
        
        out = self.out_proj(out)
        
        out = self.frm(out)

        return out

        


