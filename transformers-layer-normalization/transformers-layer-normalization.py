import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.

    Args:
        x: Input array of shape (..., d_model)
        gamma: Scale parameter of shape (d_model,)
        beta: Shift parameter of shape (d_model,)
        eps: Small constant for numerical stability

    Returns:
        Normalized array of same shape as x
    """
    # Your code here
    mean_x = np.mean(x, axis = -1, keepdims=True)
    var_x = np.var(x, axis = -1, keepdims=True)
    z = (x - mean_x)/np.sqrt(var_x + eps)
    x_out = gamma * z + beta
    x_out = x_out.reshape(*x.shape)
    return x_out