import torch
from torchvision.ops import box_iou
def calc_iou(box1_xywh, box2_xywh):
    def xywh_to_xyxy(boxes):
        x, y, w, h = boxes.unbind(1)
        return torch.stack([x, y, x + w, y + h], dim=1)

    # Convert to (x1, y1, x2, y2) format
    boxes1_xyxy = xywh_to_xyxy(box1_xywh)
    boxes2_xyxy = xywh_to_xyxy(box2_xywh)

    # Compute IoU
    iou = box_iou(boxes1_xyxy, boxes2_xyxy)

    print("IoU:\n", iou)