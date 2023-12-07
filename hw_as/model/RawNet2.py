import torch.nn as nn
from .raw_net_blocks.ResBlock import ResBlock
from .raw_net_blocks.SincConv import SincConv_fast
import torch.nn.functional as F
import torch


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

        self.first_bn = nn.BatchNorm1d(num_features = filts[0])
        self.lrelu_keras = nn.LeakyReLU(negative_slope = 0.3)
        
        self.block0 = nn.Sequential(ResBlock(emb_dims = filts[1], first = True))
        self.block1 = nn.Sequential(ResBlock(emb_dims = filts[1]))
 
        self.block2 = nn.Sequential(ResBlock(emb_dims = filts[2]))
        filts[2][0] = filts[2][1]
        self.block3 = nn.Sequential(ResBlock(emb_dims = filts[2]))
        self.block4 = nn.Sequential(ResBlock(emb_dims = filts[2]))
        self.block5 = nn.Sequential(ResBlock(emb_dims = filts[2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.bn_before_gru = nn.BatchNorm1d(num_features = filts[2][-1])
        self.gru = nn.GRU(input_size = filts[2][-1],
            hidden_size = gru_node,
            num_layers = num_gru_layers,
            batch_first = True)

        
        self.fc1_gru = nn.Linear(in_features = gru_node,
            out_features = num_fc_feats)
        self.fc2_gru = nn.Linear(in_features = num_fc_feats,
            out_features = num_classes,
            bias = True)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, audio, is_test=False, *args, **kwargs):
        x = audio
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = self.ln(x)
        x = x.view(nb_samp,1,len_seq)
        x = F.max_pool1d(torch.abs(self.first_conv(x)), 3)
        x = self.first_bn(x)
        x = self.lrelu_keras(x)
        
        x = self.block0(x)
        x = self.block1(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.bn_before_gru(x)
        x = self.lrelu_keras(x)
        x = x.permute(0, 2, 1)  #(batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:,-1,:]
        code = self.fc1_gru(x)
        if is_test: 
            return code
        code_norm = code.norm(p=2,dim=1, keepdim=True) / 10.
        code = torch.div(code, code_norm)
        out = self.fc2_gru(code)
        return {"logits": out}