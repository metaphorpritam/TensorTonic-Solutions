import numpy as np

def row_summary(data, threshold):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""
    np_arr = np.array(data, dtype = np.float64)
    elem_mask = (np_arr > threshold).astype(np.float64)

    row_any = np.any( np_arr > threshold, axis = 1)
    row_all = np.all( np_arr > threshold, axis = 1)

    any_filtered = np.where(row_any[:, None], np_arr, 0.0)
    all_filtered = np.where(row_all[:, None], np_arr, 0.0)

    return np.stack([elem_mask, any_filtered, all_filtered], axis = 0)
    
    