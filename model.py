import torch
import torch.nn as nn

class cnn_block(nn.module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(cnn_block,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.conv(x)))
    

class yolo(nn.module):
    def __init__(self, in_channels=3, **kwargs):
        super(yolo,self).__init__()
        self.in_channels = in_channels
        self.darknet = self.create_layers()






        
