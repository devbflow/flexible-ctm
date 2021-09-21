import torchvision.models as models
import torch

import os
from pathlib import Path

PATH = Path("../models")

resnets = {'resnet18': models.resnet18, 'resnet34': models.resnet34, 'resnet50': models.resnet50, 'resnet101': models.resnet101, 'resnet152': models.resnet152}


if __name__ == "__main__":
    name = 'resnet18'
    pretrained = True
    backbone = resnets[name](pretrained)
    model_name = "backbone_"+name+".pth"
    if not PATH.exists():
        PATH.mkdir()
    torch.save(backbone, PATH / model_name)

