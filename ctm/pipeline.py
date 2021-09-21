from torch import save, load


def pass_and_save_to_file(model, in_batch, out_fname):
    """Passes an input (batch) through a model (e.g. backbone) and
    saves the output to disk.
    This can be considered a utility function to facilitate training
    sequentially.
    Use this, if you have limited time on hand and don't need
    a full pass through multiple modules at once.
    """
    out = model(in_batch)
    save(out, out_fname+'.pt')


def load_tensor_from_file(fname):
    """Loads a tensor from a given filename."""
    return load(fname+'.pt')
