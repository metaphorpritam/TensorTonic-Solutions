import numpy as np

def select_by_index(arr, indices, axis):
    """
    Returns: 2D ndarray of float64
    """
    np_array = np.array(arr, dtype = np.float64)

    return  np.take(np_array, indices=indices, axis = axis)