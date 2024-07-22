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
    def __init__(self, in_channels=3, architecture = None, **kwargs):
        super(yolo,self).__init__()
        self.architecture = architecture
        self.in_channels = in_channels
        self.darknet = self._create_layers(self.architecture)
        self.fcs = self._create_fully_connected(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_layers(self, arch):
        layers = []
        in_channels = self.in_channels
        for layer in self.architecture:
            if isinstance(layer[0],int):
                layers += cnn_block(in_channels,layer[1],layer[2])











        
