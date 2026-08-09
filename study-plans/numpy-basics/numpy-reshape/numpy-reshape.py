import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    data_arr = np.array(data, dtype=np.float64)

    match operation:
        case "flatten":
            return data_arr.flatten()
        case "transpose":
            return data_arr.T
        case "add_batch":
            return np.expand_dims(data_arr, axis = 0)
        case _:
            print("{operation} not supported")
            return None
