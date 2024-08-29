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

        iou_b1 = calc_iou(predictions[...,21:25], target[...,21:25])
        iou_b2 = calc_iou(predictions[...,26:30], target[...,21:25])
        ious = torch.cat([iou_b1.unsqueeze(0)],[iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious,dim=0)
        exists_box = target[...,20].unsqueeze(3) # does object exist

        #box coord
        box_preds = exists_box * (
            bestbox * predictions[...,26:30] + (1-bestbox) * predictions[..., 21:25] #best box is 1 if the second box has the best iou otherwise 0, TODO this is unreadable 
        )
        box_targets = exists_box * target[..., 21:25]

        box_preds[...,2:4] = torch.sign(box_preds[...,2:4]) * torch.sqrt(torch.abs(box_preds[...,2:4] + 1e-6))

        box_targets = torch.sqrt(box_preds[...,2:4])

        box_loss = self.mse(
            torch.flatten(box_preds, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        #object loss
        pred_box = (
            bestbox * predictions[...,25:26] + (1-bestbox) * predictions[..., 20:21]
        )

        obj_loss = self.mse(
            torch.flatten(exists_box * pred_box),

        )

        #no object loss

        #class loss




        
