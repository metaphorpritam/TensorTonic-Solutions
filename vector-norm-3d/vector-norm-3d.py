import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    # Your code here
    v = np.asarray(v, dtype=float)
    is_single_vec = (v.ndim == 1)

    if is_single_vec:
        norm = np.sqrt(np.sum(v**2))
    else:
        norm = np.sqrt(np.sum(v**2, axis = 1))
        

    return norm
