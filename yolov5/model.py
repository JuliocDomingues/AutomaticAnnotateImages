import torch
from yolov5.models.common import DetectMultiBackend


def load_model(weights='yolov5n.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False):

    return DetectMultiBackend(weights=weights, device=device, dnn=dnn, data=data, fp16=fp16)
