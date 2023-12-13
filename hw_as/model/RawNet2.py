import torch.nn as nn
from .raw_net_blocks.ResBlock import ResBlock
from .raw_net_blocks.SincConv import SincConv_fast
import torch.nn.functional as F
import torch
import numpy as np

class NormLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x_norm = x.norm(p=2,dim=1, keepdim=True) / 10.
        x = torch.div(x, x_norm)
        return x


class RawNet2(nn.Module):
    def __init__(self, 
                 in_channels_sinc=1, 
                 filts = [20, [20, 20], [20, 128], [128, 128]], 
                 ks_conv_sinc=1024, 
                 gru_node=1024,
                 num_gru_layers=3,
                 num_fc_feats=1024,
                 num_classes=2,
                 *args, **kwargs):
        super().__init__()
        self.first_conv = SincConv_fast(in_channels = in_channels_sinc,
            out_channels = filts[0],
            kernel_size = ks_conv_sinc
            )

        self.prejump = nn.Sequential(
            nn.BatchNorm1d(num_features = filts[0]),
            nn.LeakyReLU(negative_slope = 0.3),
            ResBlock(emb_dims = filts[1], first = True),
            ResBlock(emb_dims = filts[1]),
            ResBlock(emb_dims = filts[2])
        )

        filts[2][0] = filts[2][1]
        self.afjump = nn.Sequential(
            ResBlock(emb_dims = filts[2]),
            ResBlock(emb_dims = filts[2]),
            ResBlock(emb_dims = filts[2]),
            nn.BatchNorm1d(num_features = filts[2][-1]),
            nn.LeakyReLU(negative_slope = 0.3)
        )

        self.gru = nn.GRU(input_size = filts[2][-1],
            hidden_size = gru_node,
            num_layers = num_gru_layers,
            batch_first = True)
        
        self.fc1_gru = nn.Linear(in_features = gru_node,
            out_features = num_fc_feats)
        
        self.norm_layer = NormLayer()
        self.fc2_gru = nn.Linear(in_features = num_fc_feats,
            out_features = num_classes,
            bias = True)
        
    def forward(self, audio, *args, **kwargs):
        x = audio
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1,len_seq)
        x = F.max_pool1d(self.first_conv(x), 3)

        x = self.prejump(x)

        x = self.afjump(x)
        x = x.permute(0, 2, 1)  #(batch, filt, time) >> (batch, time, filt)
        x, _ = self.gru(x)
        x = x[:,-1,:]
        code = self.fc1_gru(x)
        code = self.norm_layer(code)
        out = self.fc2_gru(code)
        return {"logits": out}
    

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)