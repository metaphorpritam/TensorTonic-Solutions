import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x = np.asarray(x, dtype = float)
    mask = x < 0
    l_relu = np.where(mask, alpha*x, x)
    return l_relu