import unittest

from . import ctm


def test_get_conv_params():
    in_dims = [14]
    out_dims = [12]
    correct_p = [{'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1}]

    for i, o, cp in zip(in_dims, out_dims, correct_p):
        p = ctm.get_conv_params(i, o)
        assert cp == p
    print("No assertionError found, test_get_conv_params OK!")


