"""
A collection of utility functions/tools, that can be helpful.
"""



def conv_out_dims(input_width, kernel_size, padding, stride):
    """
    Returns the output dimensions after convolution.
    Input width can be integer or tuple of integers, output size will return
    in the same style.
    """
    if type(input_width) == tuple:
        b = True
        for i in range(len(input_width)-1):
            if input_width[i] != input_width[i+1]:
                b = False
        if b == True:
            width = input_width[0]
            out_size = (width - kernel_size + 2 * padding) / stride + 1
        else:
            o = []
            for i in range(len(input_width)):
                o.append(input_width[i] - kernel_size + 2 * padding) / stride + 1)
            out_size = tuple(o)
    elif type(input_width) == int:
        out_size = (input_width - kernel_size + 2 * padding) / stride + 1
    else:
        raise ValueError("input_width must be int or tuple!")
    return out_size
