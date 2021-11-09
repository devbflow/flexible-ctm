import torch
from torch import nn as nn

class MetricModule(nn.Module):
    """Wrapper module class for a metric learner.
    Can be a simple metric or a neural network, as long as it inherits from nn.Module.

    If your metric is more complicated, adapt the code accordingly.
    """

    def __init__(self, metric, trainable):
        super().__init__()
        self.metric = metric

    def forward(self, *args, **kwargs):
        return self.metric(*args, **kwargs)


class CosineSimModule(MetricModule):
    """CosineSimilarity metric."""
    def __init__(self, dim=1, trainable=False):
        super().__init__(nn.CosineSimilarity(dim=dim), trainable=False)

class PairwiseDistModule(MetricModule):
    """Pairwise Distance metric."""
    def __init__(self, p=2.0, trainable=False):
        super().__init__(nn.PairwiseDistance(p), trainable=False)

    def forward(self, support, query, n_way, k_shot):
        # conflate channels and dims for support and query set
        support = support.view(support.shape[0], -1)
        cdd = support.shape[-1]
        qsize = query.shape[0] # query batch size
        query = query.view(qsize, -1)
        # expand support/query to get pairwise distance between all query and support samples
        exp_supp = support.unsqueeze(0).expand(qsize, -1, -1).contiguous().view(-1, cdd)
        exp_query = query.unsqueeze(1).expand(-1, n_way*k_shot, -1).contiguous().view(-1, cdd)
        # get metric score and reshape to (query size, n, k) to sum over k samples
        # take negative of PairwiseDistance so we maximize
        score = -self.metric(exp_supp, exp_query).view(qsize, n_way, k_shot).sum(dim=2) # (query size, n)
        return score

# contains all implemented 
METRICS = {'cosine': CosineSimModule,
           'pairwise': PairwiseDistModule}


if __name__ == "__main__":
    print("Test PairwiseDistModule...")
    n = 2
    k = 5
    support = torch.rand((n*k, 3, 4, 4))
    query = torch.rand((3*n*k, 3, 4, 4))
    pdist = PairwiseDistModule()
    score = pdist(support, query, n, k)
    print(score)
