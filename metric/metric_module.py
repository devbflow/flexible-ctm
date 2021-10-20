import torch
from torch import nn as nn

class MetricModule(nn.Module):
    """Wrapper module class for a metric learner.
    Can be a simple metric or a neural network, as long as it inherits from nn.Module.

    If your metric is more complicated, adapt the code accordingly.
    """

    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    def forward(self, x):
        return self.metric(x)


class CosineSimModule(MetricModule):
    """CosineSimilarity metric."""
    def __init__(self, dim=1):
        super().__init__(nn.CosineSimilarity(dim=dim))

class PairwiseDistModule(MetricModule):
    """Pairwise Distance metric."""
    def __init__(self, p=2.0):
        super().__init__(nn.PairwiseDistance(p))

# contains all implemented 
METRICS = {'cosine': CosineSimModule,
           'pairwise': PairwiseDistModule}
