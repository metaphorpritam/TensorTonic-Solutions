import numpy as np

def norm_diff(a, b, lo, hi):
    """Returns: np.ndarray of absolute differences after clipping and rescaling to [0, 1]"""
    a = np.array(a, dtype = np.float64)
    b = np.array(b, dtype = np.float64)

    a_clipped = np.clip(a, lo, hi)
    b_clipped = np.clip(b, lo, hi)

    a_hat = (a_clipped - lo)/(hi -lo)
    b_hat = (b_clipped - lo)/(hi -lo)

    return np.abs(a_hat - b_hat)
    