import torch
from ..ctm.components import Identity


def preprocess_backbone(backbone, description='resnet18', dims=7):
    """Eliminates unused layers from backbone model and removes its parameters
    from the autograd backpropagation graph.
    """
    model = backbone
    if description == 'resnet18':
        model.avgpool = Identity()
        model.fc = Identity()

        if dims == 14:
            # set layer4 to Identity for (14, 14, 256) instead of (7, 7, 512)
            model.layer4 = Identity()

    for param in model.parameters():
        param.requires_grad = False

    return model
