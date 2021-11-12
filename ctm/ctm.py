import torch
from torch import nn
from torch.nn import functional as F

ACTIVATIONS = {'relu': nn.ReLU,
               'leaky': nn.LeakyReLU}

def make_layers(structure, input_channels, ret_type='seq'):
    """
    Receives a structural configuration and builds layers with respect to the
    configuration given.
    Can return a ModuleList or Sequential.

    Parameters
    ----------
    structure : dict
        structural configuration as parsed from config.yml
    input_channels : int
        number of input channels for the first layer
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

    lact = ACTIVATIONS[structure['activation']](inplace=True)
    ltype = structure['type'] # layer type
    out_channels = 64 # base number, is being doubled every layer

    if ret_type == 'list':
        layers = nn.ModuleList()
    elif ret_type == 'seq':
        layers = []
    else:
        raise ValueError("make_layers received improper ret_type argument ({})".format(ret_type))

    for i in range(structure['num']):
        # in this loop, the layers are instantiated, followed by (if desired) batchnorm and activation
        if ltype.startswith('conv2d'):
            layers.append(nn.Conv2d(in_channels=input_channels, out_channels=out_channels*(2**i), kernel_size=int(ltype[-1])))
        if structure['batchnorm']:
            layers.append(nn.BatchNorm2d(out_channels*(2**i)))
        layers.append(lact)
        input_channels = out_channels*(2**i)

    if ret_type == 'seq':
        layers = nn.Sequential(*layers)
    return layers

def get_conv_params(in_dim, out_dim,
                    stride=[1,2,3,4,5],
                    kernel_size=[3,5,7,9],
                    padding=[0,1,2],
                    dilation=[1,2]):
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
    # make single int inputs to lists to prevent errors
    if type(stride) == int:
        stride = [stride]
    if type(kernel_size) == int:
        kernel_size = [kernel_size]
    if type(padding) == int:
        padding = [padding]
    if type(dilation) == int:
        dilation = [dilation]
    # make sure input/output dims are symmetrical (square images)
    if type(in_dim) == tuple:
        if in_dim[0] != in_dim[1]:
            raise ValueError("get_conv_params: in_dim H and W are not equal ({},{})".format(*in_dim))
        in_dim = in_dim[0]
    if type(out_dim) == tuple:
        if out_dim[0] != out_dim[1]:
            raise ValueError("get_conv_params: out_dim H and W are not equal ({},{})".format(*out_dim))
        out_dim = out_dim[0]

    for d in dilation:
        for p in padding:
            for s in stride:
                for k in kernel_size:
                    #print(d, p, s, k)
                    #print(in_dim)
                    res = int((in_dim + 2*p - d*(k-1) - 1)/s + 1)
                    if res == out_dim:
                        return {'kernel_size': k, 'stride': s, 'padding': p, 'dilation': d}


class CTM(nn.Module):
    """
    CTM module.
    This module receives the data in its embedded space, found by means of a backbone like ResNet-18,
    and finds a mask p that improves the embedded features in (further) reduced dimensions.

    The Concentrator finds commonalities (or average features) in data points from one class, while the Projector finds
    distinguishing features among different classes.
    From this results the mask p, that emphasizes relevant features.
    The Reshaper makes sure, we can apply p to the feature embeddings to create 'improved feature embeddings'.

    These 'improved features' then are supplied to a metric or metric learner.

    Attributes
    ----------
    cfg : dict
        CTM configuration dict containing information about the structure of all parts
    concentrator : Concentrator
        Concentrator module
    projector : Projector
        Projector module
    reshaper : Reshaper
        Reshaper module
    dataset : str
        Name of the dataset for which the CTM instance is used
    input_channels : int
        Number of input channels of the data
    output_channels : int
        Number of output channels of CTM output
    input_dim : int or tuple
        Input dimensions of the data
    output_dim : tuple
        Output dimensions of the data
    n : int
        N in the N-way K-shot learning scenario
    k : int
        K in the N-way K-shot learning scenario

    Methods
    -------
    _init_reshaper()
        Initializes the Reshaper and certain attributes
    _init_dataset()
        Initializes the dataset configuration necessary for CTM

    """

    def __init__(self, config, dataset_config, backbone_outchannels=0, backbone_outdim=0):
        """
        Parameters
        ----------
        config : dict
            CTM configuration dict as parsed from config file
        dataset_config : dict
            Dataset configuration dict as parsed from config file
        """
        super().__init__()
        self.cfg = config
        if backbone_outchannels:
            self.input_channels = backbone_outchannels
        else:
            self.input_channels = dataset_config['channels']
        if backbone_outdim:
            self.input_dim = backbone_outdim
        else:
            self.input_dim = dataset_config['shape']

        self.dataset = dataset_config['name']
        self.n = dataset_config['n_way']
        self.k = dataset_config['k_shot']

        self._init_modules(self.cfg['concentrator'], self.cfg['projector'])

    def _init_modules(self, concentrator_cfg, projector_cfg):
        """
        Set Concentrator/Projector/Reshaper (C/P/R) by passing a zero tensor to get the correct shapes.
        Also saves output channels adn dimensions as attribute.
        """
        with torch.no_grad():
            # pass through C and P to get dimensions for R
            if isinstance(self.input_dim, int):
                zeros = torch.zeros(self.n*self.k, self.input_channels, self.input_dim, self.input_dim)
            else:
                zeros = torch.zeros(self.n*self.k, self.input_channels, *self.input_dim)

            self.concentrator = Concentrator(concentrator_cfg, self.input_channels, self.n, self.k)
            z = self.concentrator(zeros)
            self.projector = Projector(projector_cfg, z.shape[1], self.n)
            zero_output = self.projector(z)

        self.output_channels = zero_output.shape[1]
        self.output_dim = tuple(zero_output.shape[2:])
        self.reshaper = Reshaper(in_channels=self.input_channels,
                                 out_channels=self.output_channels,
                                 in_dims=self.input_dim,
                                 out_dims=self.output_dim,
                                 auto_params=True,
                                 params=None)

    def forward(self, support_set, query_set):
        # reshape support and query sets into further embedded form
        supp_r = self.reshaper(support_set)
        query_r = self.reshaper(query_set)

        o = self.concentrator(support_set) # notation as in paper
        p = self.projector(o)

        improved_supp = supp_r * p
        improved_query = query_r * p

        return improved_supp, improved_query


class Concentrator(nn.Module):
    """Simple Concentrator

    Attributes
    ----------
    cfg : dict
        Concentrator configuration dict containing information about the structure
    n : int
        N in the N-way K-shot learning scenario
    k : int
        K in the N-way K-shot learning scenario
    layers : nn.ModuleList or nn.Sequential
        convolutional layers in the Concentrator

    """

    def __init__(self, config, input_channels, n_way: int, k_shot: int):
        super().__init__()
        self.n = n_way
        self.k = k_shot
        self.cfg = config
        self.layers = make_layers(self.cfg['structure'], input_channels) # is ModuleList or Sequential

    def forward(self, X):
        # pass through layers first to reduce dimensions
        if isinstance(self.layers, nn.ModuleList):
            Y = X
            for l in self.layers:
                Y = l(Y)
        elif isinstance(self.layers, nn.Sequential):
            Y = self.layers(X)
        else:
            raise TypeError("Concentrator layers are neither ModuleList nor Sequential!")
        # average over samples in each class (reshape to (N, K*c, d, d))
        Y = Y.view(self.n, self.k, Y.shape[1], Y.shape[2], Y.shape[3])
        Y = torch.mean(Y, dim=1)
        # reshape to (1, N*c, d, d) when returning for the projector
        return Y.view(1, self.n*Y.shape[1], Y.shape[2], Y.shape[3])

class Projector(nn.Module):
    """Simple Projector"""

    def __init__(self, config, input_channels, n_way: int):
        super().__init__()
        self.n = n_way
        self.cfg = config

        self.layers = make_layers(self.cfg['structure'], input_channels)

    def forward(self, X):
        # reshape of input happens in the concentrator above
        if isinstance(self.layers, nn.ModuleList):
            Y = X
            for l in self.layers:
                Y = l(Y)
        elif isinstance(self.layers, nn.Sequential):
            Y = self.layers(X)
        else:
            raise TypeError("Concentrator layers are neither ModuleList nor Sequential!")
        return F.softmax(Y, dim=1)

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
        # the reshaper needs not to be configured and can stay a simple Conv layer (see paper)
        # to keep consistency, 'layers' stays as var name

        # if auto_params is True, search for parameters in param space, else assume params to be given
        if auto_params:
            params = get_conv_params(in_dims, out_dims)
            if not params:
                raise TypeError("get_conv_params returned None!")
        if not params and not auto_params:
            raise ValueError("Did not receive parameter dict while auto_params is False.")
        self.layers = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=params['kernel_size'],
                                stride=params['stride'],
                                padding=params['padding'],
                                dilation=params['dilation'])

    def forward(self, X):
        return self.layers(X)

