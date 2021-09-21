import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """BasicBlock module based on torchvision/models/resnet.py implementation.
    Implements a residual basic block (3x3 Conv -> 3x3 Conv).
    """
    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = F.relu(inplace=True)

    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        y += identity
        y = self.relu(y)

        return y

class Bottleneck(nn.Module):
    """Bottleneck module based on torchvision/models/resnet.py implementaton.
    Implements a residual bottleneck block (1x1 Conv -> 3x3 Conv -> 1x1 Conv).
    """

    expansion = 4

    def __init__(self,inplanes, planes, stride=1, downsample=None, groups=1):
        super().__init__()
        # placeholder in case of different plane choice, see ref'd implementation
        width = planes

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = F.relu(inplace=True)

    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)

        y += identity
        y = self.relu(y)

        return y
