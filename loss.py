import torch
import torch.nn as nn
from utils import iou


class YoloLoss(nn.Module):
    def __init__(self,S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = .5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.B*5 + self.C)
        
