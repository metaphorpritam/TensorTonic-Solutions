import numpy as np

def normalize(data):
    """Returns: np.ndarray of shape (m, n), z-score normalized per column"""
    data = np.array(data, dtype= np.float64)
    col_mean = data.mean(axis = 0)
    col_std = data.std(axis = 0)
    return (data - col_mean)/col_std