import torch
from yolov5.models.common import DetectMultiBackend


def load_model(weights='yolov5n.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False):

    if data is None:
        from pathlib import Path
        import sys
        import os

        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0]  # YOLOv5 root directory
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # add ROOT to PATH
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

        return DetectMultiBackend(weights=weights, device=device, dnn=dnn, data=ROOT / 'data/coco128.yaml', fp16=fp16)
    return None
