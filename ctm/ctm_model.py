import torch
from torch import nn
from torch.nn import functional as F
from .components import BasicBlock, Bottleneck
from .utils import conv_out_dims


# (module)dictionaries containing module names mapped to their implementation
# PyTorch standard module implementations are imported from torch.nn,
# custom implementations from components
BLOCK_TYPES = nn.ModuleDict({'conv2d': nn.Conv2d,
               'basicblock': BasicBlock,
               'bottleneck': Bottleneck,
               'identity': nn.Identity})

# Module form of activations is used
ACTIVATIONS = nn.ModuleDict({'relu': nn.ReLU(),
                             'leakyrelu': nn.LeakyReLU()})

class CTM(nn.Module):
    """ Category Traversal Module.
    The goal is to develop an as abstract/modular as possible solution.
    That way, it will be possible to plug-and-play it into different metric learners as an optional, performance improving module.
    """

    def __init__(self, config, n_way: int, k_shot: int, support_set: torch.Tensor, query: torch.Tensor):
        super().__init__()
        # init structure
        self.n = n_way
        self.k = k_shot
        parts = config['parts']

        if parts['concentrator']:
            self.concentrator = Concentrator(config['concentrator'], self.n, self.k, support_set)
        else:
            self.concentrator = nn.Identity()
        if parts['projector']:
            self.projector = Projector(config['projector'], self.n)
        else:
            self.projector = nn.Identity()
        if parts['reshaper']:
            self.reshaper = Reshaper(config['reshaper'], self.n, self.k)
        else:
            self.reshaper = nn.Identity()


    def forward(self, support, query):
        """Forward pass."""
        p = self.concentrator(support)
        # if projector not meaningful, mask p equals multiplication with 1
        if type(self.projector) != nn.Identity:
            p = self.projector(p)
        else:
            p = 1.0

        # reshape the support and query sets
        s_resh = self.reshaper(support)
        q_resh = self.reshaper(query)

        # apply projector output p to reshaped embeddings to get improved features
        ifeat_support = s_resh * p
        ifeat_query = q_resh * p

        return ifeat_support, ifeat_query


class Concentrator(nn.Module):
    """ Concentrator module.
    Finds intra-class commonalities - features shared by each member of a class.
    """

    def __init__(self, config, n_way: int, k_shot: int):
        super().__init__()
        # init structure
        self.n = n_way
        self.k = k_shot
        self.modules = make_modulelist(config['structure'])

    def forward(self, x):
        """Forward pass."""
        # reshape support set features from (N*K, m1,d1, d1) to (N, m2, d2, d2)

        c = x.shape[1]
        d = x.shape[2]

        # reshape input batch
        out = torch.reshape(x, (self.n, self.k * c, d, d))



        return out

class Projector(nn.Module):
    """ Projector module.
    Finds inter-class uniquenesses - most discriminative features for the given few-shot task.
    """

    def __init__(self, config, n_way):
        super().__init__()
        # init structure
        self.n = n_way
        self.softmax = F.softmax(dim=1)
        self.modules = make_modulelist(config['structure'])

    def forward(self, x):
        """Forward pass, returns mask p."""
        # reshape concentrator output from (1, N*m2, d2, d2) to (1, m3, d3, d3)

        c = x.shape[1]
        d = x.shape[2]
        out = torch.reshape(x, (1, self.n_way * c, d, d))

        # forward pass through CNN
        # softmax over channel dimension m3
        # TODO

        return p

class Reshaper(nn.Module):
    """Reshapes input to desired output dimensions, if possible."""

    def __init__(self, config):
        super().__init__()
        # init structure
        self.modules = make_modulelist(config['structure'])

    def forward(self, x, output_channels, output_dims):
        # reshape input x to output dims (NK, m3, d3, d3)
        # TODO
        return x


def make_modulelist(cfg):
    """Parses structure of a CTM block from given CTM config and returns a PyTorch ModuleList."""
    btype = cfg['btype']
    bnum = cfg['bnum']
    activation = cfg['activation']
    bn = cfg['batchnorm']

    layers = []
    # check for block type and build list accordingly
    if btype[:6] == 'conv2d':
        kernel_size = int(btype[-1])

        for i in range(1, bnum+1):
            #TODO make sequential layers
    elif btype == 'basicblock':
        raise NotImplementedError("BasicBlock not implemented.")
    elif btype == 'bottleneck':
        raise NotImplementedError("Bottleneck not implemented.")
    elif btype == 'identity':
        #TODO
    else:
        return None

    return nn.ModuleList(layers)

