import numpy as np

def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0, 0), (0, patch_size - x.shape[1] % patch_size)), "wrap")


def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std * 1.0)