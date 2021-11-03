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

    def forward(self, *args, **kwargs):
        return self.metric(*args, **kwargs)


class CosineSimModule(MetricModule):
    """CosineSimilarity metric."""
    def __init__(self, dim=1):
        super().__init__(nn.CosineSimilarity(dim=dim))

class PairwiseDistModule(MetricModule):
    """Pairwise Distance metric."""
    def __init__(self, p=2.0):
        super().__init__(nn.PairwiseDistance(p))

    def forward(self, support, query, n_way, k_shot):
        support = support.view(support.shape[0], -1)
        query = query.view(query.shape[0], -1)
        score = self.metric(support, query)
        return score.view(query.shape[0], n_way, k_shot)

# contains all implemented 
METRICS = {'cosine': CosineSimModule,
           'pairwise': PairwiseDistModule}


if __name__ == "__main__":
    print("Test PairwiseDistModule...")
    n = 2
    k = 5
    support = torch.rand((n*k, 2, 3, 4))
    query = torch.rand((n*k, 2, 3, 4))
    pdist = PairwiseDistModule()
    score = pdist(support, query, n, k)
