import torch
import torch.nn as nn


def preprocess_backbone(backbone, description='resnet18', dims=14):
    """Eliminates unused layers from backbone model and removes its parameters
    from the autograd backpropagation graph.
    """
    model = backbone
    if description == 'resnet18':
        model.avgpool = nn.Identity()
        model.fc = nn.Identity()

        if dims == 14:
            # set layer4 to Identity for (14, 14, 256) instead of (7, 7, 512)
            model.layer4 = nn.Identity()

    for param in model.parameters():
        param.requires_grad = False

    return model
