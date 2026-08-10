import numpy as np

def filter_and_extract(data, row_start, row_stop, threshold):
    """
    Returns: 1D ndarray of float64
    """
    arr = np.array(data, dtype=np.float64)
    arr_new = arr[row_start:row_stop, :]
    mask = arr_new > threshold
    return np.extract(mask, arr_new)