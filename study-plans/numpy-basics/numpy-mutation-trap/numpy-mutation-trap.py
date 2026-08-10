import numpy as np

def original_and_clipped(data, row_idx, lo, hi):
    """
    Returns: 2D ndarray of float64 with shape (2, ncols)
    """
    arr = np.array(data, dtype=np.float64)
    row = arr[row_idx].copy()
    clipped = np.clip(row, lo, hi)

    return np.stack([row, clipped], axis = 0)