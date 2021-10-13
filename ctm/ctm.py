import torch
from torch import nn


def make_layers(structure, ret_type='seq'):
    """
    Receives a structural configuration and builds layers with respect to the
    configuration given.
    Can return a ModuleList or Sequential.

    Parameters
    ----------
    structure : dict
        structural configuration as parsed from config.yml
    ret_type : str, optional
        return type, can be 'list' or 'seq' for ModuleList/Sequential respectively

    Returns
    -------
    layers : ModuleList or Sequential
        created layer structure

    Raises
    ------
    ValueError
        if ret_type is an illegal value
    """
    #TODO: create proper version
    if ret_type == 'list':
        layers = nn.ModuleList()
        #TODO: add ModuleList capacities
    elif ret_type == 'seq':
        layers = nn.Sequential(nn.Conv2d(3, 64, 3))
        #TODO: make a proper version
    else:
        raise ValueError("make_layers received improper ret_type argument!")
    return layers

def get_conv_params(in_dim, out_dim,
                    stride=[1,2],
                    kernel_size=[3,5,7],
                    padding=[0],
                    dilation=[1]):
    """
    Function to search for fitting parameters to automatically set reshaper conv parameters.
    Returns a dictionary with fitting key and value pairs.
    It returns the first solution found, prioritizing as follows.
        kernel_size > stride > padding > dilation

    Parameters
    ----------
    in_dim : int or tuple
        input dimensions
    out_dim : int or tuple
        output dimensions
    stride : int or list
        stride values to be checked, contains int values if list
    kernel_size : int or list
        kernel sizes to be checked, assumes symmetrical kernel (nxn) and contains int values if list
    padding : int or list
        padding amount to be checked, contains int values if list
    dilation : int or list
        dilation values to be checked, contains int values if list

    Returns
    -------
    dictionary with found parameter combinations with 'stride', 'kernel_size', 'padding' and 'dilation' as keys
    """

    if type(stride) == int:
        stride = [stride]
    if type(kernel_size) == int:
        kernel_size = [kernel_size]
    if type(padding) == int:
        padding = [padding]
    if type(dilation) == int:
        dilation = [dilation]

    for d in dilation:
        for p in padding:
            for s in stride:
                for k in kernel_size:
                    res = int((in_dim + 2*p - d*(k-1) - 1)/s + 1)
                    if res == out_dim:
                        return {'kernel_size': k, 'stride': s, 'padding': p, 'dilation': d}


class CTM(nn.Module):

    def __init__(self, config, dataset_config):
        """

        """
        super().__init__()
        self.cfg = config
        self._init_dataset(dataset_config)

        self.concentrator = Concentrator(self.cfg['concentrator'], self.n, self.k)
        self.projector = Projector(self.cfg['projector'], self.n)

        self._init_reshaper()

    def _init_reshaper(self):
        """
        Set Reshaper by passing a zero tensor through Concentrator and Projector.
        Also saves output channels as class variable.
        """

        zeros = torch.zeros(1, self.input_channels, *self.input_dim)
        zero_output = self.projector(self.concentrator(zeros))
        self.output_channels = zero_output.shape[1]
        self.reshaper = Reshaper(in_channels=self.input_channels,
                                 out_channels=self.output_channels,
                                 in_dims=self.input_dim,
                                 out_dims=zero_output.shape[2:],
                                 auto_params=True,
                                 params=None)

    def _init_dataset(self, dataset_config):
        """Setup dataset-related class variables."""

        self.dataset = dataset_config['name']
        self.input_channels = dataset_config['channels']
        shape = dataset_config['shape']
        if type(shape) == int and dataset_config['datatype'] == 'img':
            self.input_dim = (shape, shape)
        elif type(shape) == list:
            self.input_dim = tuple(shape)
        else:
            raise TypeError("Dataset shape type mismatch in config file. (list or int required)")
        self.n = dataset_config['n_way']
        self.k = dataset_config['k_shot']



class Concentrator(nn.Module):
    """Simple Concentrator"""

    def __init__(self, config, n_way: int, k_shot: int):
        super().__init__()
        self.n = n_way
        self.k = k_shot
        self.cfg = config
        self.layers = make_layers(self.cfg['structure']) # is ModuleList or Sequential

    def forward(self, X):
        # reshape to (N, K*c, d, d)
        X = X.view(self.n, self.k*X.shape[1], X.shape[2], X.shape[3])

        if type(self.layers) == nn.ModuleList:
            Y = X
            for l in self.layers:
                Y = l(Y)
            return Y
        elif type(self.layers) == nn.Sequential:
            return self.layers(X)
        else:
            raise TypeError("Concentrator layers are neither ModuleList nor Sequential!")


class Projector(nn.Module):
    """Simple Projector"""

    def __init__(self, config, n_way: int):
        super().__init__()
        self.n = n_way
        self.cfg = config

        self.layers = make_layers(self.cfg['structure'])

    def forward(self, X):
        # reshape to (1, N*c, d, d)
        X = X.view(1, self.n*X.shape[1], X.shape[2], X.shape[3])

        if type(self.layers) == nn.ModuleList:
            Y = X
            for l in self.layers:
                Y = l(Y)
            return Y
        elif type(self.layers) == nn.Sequential:
            return self.layers(X)
        else:
            raise TypeError("Concentrator layers are neither ModuleList nor Sequential!")

class Reshaper(nn.Module):
    """Reshaper module.

    Transforms given data into the same shape as the output of the Projector.
    Thus, the mask returned by the Projector can be applied to the transformed data.

    Attributes
    ----------
    layers : nn.Module
        The convolutional layer of the Reshaper. Name was kept in plural for consistency.
    """

    def __init__(self, in_channels, out_channels, in_dims, out_dims, auto_params=True, params=None):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels of data to reshape
        out_channels : int
            Number of resulting output channels
        in_dims : int, tuple or list
            Input dimensions of a data point
        out_dims : int, tuple or list
            Output dimensions of a data point
        auto_params : bool
            If True, uses get_conv_params to find fitting parameters. Default is True
        params : dict
            If auto_params is False, contains the parameters necessary for initializing the Conv layer. Default is None.
        """
        super().__init__()
        # the reshaper needs not to be configured and can stay a simple Conv layer
        # to keep consistency, 'layers' stays as var name

        # if auto_params is True, search for parameters in param space, else assume params to be given
        if auto_params:
            params = get_conv_params(in_dims, out_dims)
            if params is None:
                raise TypeError("get_conv_params returned None!")
        if params is None and not auto_params:
            raise ValueError("Did not receive parameter dict while auto_params is False.")
        self.layers = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=params['kernel_size'],
                                stride=params['stride'],
                                padding=params['padding'],
                                dilation=params['dilation'])

    def forward(self, X):
        return self.layers(X)
