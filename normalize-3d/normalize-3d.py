import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.asarray(v, dtype = float)
    is_single = (v.ndim == 1)

    if v.ndim == 1:
        v = v.reshape(1, 3)

    norm = np.linalg.norm(v, axis = 1, keepdims=True)

    mask = norm > 10e-10

    out = np.zeros_like(v)
    out[mask[:, 0]] = v[mask[:,0]] / norm[mask[:,0]]

    # Alternative
    # out = v /np.where(mask, norms, 1.0)
    # out[~mask[:,0]] = 0.0

    if is_single:
        out = out.reshape(3,)

    return out








    return out