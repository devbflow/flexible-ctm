import torch
from torch import nn as nn

class MetricModule(nn.Module):
    """Wrapper module class for a metric learner.
    Can be a simple metric or a neural network, as long as it works like
    a function.

    If your metric is more complicated, adapt the code accordingly.
    """

    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    def forward(self, x):
        return self.metric(x)


class CosineModule(MetricModule):
    """CosineSimilarity metric."""
    def __init__(self, dim):
        super().__init__(nn.CosineSimilarity(dim=dim))

