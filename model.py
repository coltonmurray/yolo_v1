import torch
import torch.nn as nn
import yaml
import cv2

class cnn_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(cnn_block,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.conv(x)))
    

class yolo(nn.Module):
    def __init__(self, in_channels=3, architecture = None, split_size = 7, num_boxes = 2, num_classes = 20):
        super(yolo,self).__init__()
        self.architecture = architecture
        self.in_channels = in_channels
        self.darknet = self._create_layers(self.architecture)
        self.fcs = self._create_fully_connected(split_size,num_boxes,num_classes)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_layers(self, arch):
        layers = []
        in_channels = self.in_channels
        for layer in arch["backbone"]:
            
            if isinstance(layer[0],int):
                
                layers += [cnn_block(
                    in_channels=in_channels, 
                    out_channels=layer[1], 
                    kernel_size = layer[0], 
                    stride = layer[2], 
                    padding = layer[3]
                    )]
                in_channels = layer[1]
            elif isinstance(layer[0],str):
                
                layers += [nn.MaxPool2d(kernel_size=2, stride = 2)]

            elif isinstance(layer[0],list):
                
                conv1 = layer[0]
                conv2 = layer[1]
                num_repeats = layer[2]

                for _ in range(num_repeats):
                    
                    layers += [cnn_block(
                    in_channels=in_channels, 
                    out_channels=conv1[1], 
                    kernel_size = conv1[0], 
                    stride = conv1[2], 
                    padding = conv1[3]
                    )]
                    in_channels = conv1[1]
                    layers += [cnn_block(
                    in_channels=in_channels, 
                    out_channels=conv2[1], 
                    kernel_size = conv2[0], 
                    stride = conv2[2], 
                    padding = conv2[3]
                    )]
                    in_channels = conv2[1]
        return nn.Sequential(*layers)
    
    def _create_fully_connected(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * ((B*5)+C))
        )

    
def test(arch_path = 'yolo.yaml',split_size=7, num_boxes=2, num_classes=20):
    with open(arch_path, 'r') as a:
        arch = yaml.safe_load(a)
   

    model = yolo(split_size = split_size, num_boxes = num_boxes, num_classes = num_classes, architecture=arch)
    x = torch.randn((2,3,448,448))
 
    print(model(x).shape)


if __name__ == "__main__":

    test()















        
