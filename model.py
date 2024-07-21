import torch
import torch.nn as nn

class cnn_block(nn.module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(cnn_block,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)


        
