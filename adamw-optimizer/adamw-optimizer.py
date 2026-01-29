import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.

    Params:
    w: np.ndarray - Current parameters (any shape)
    m: np.ndarray - First moment estimates (same shape as w)
    v: np.ndarray - Second moment estimates (same shape as w)
    grad: np.ndarray - Current gradients (same shape as w)
    lr: float = 0.001 - Learning rate
    beta1: float = 0.9 - First moment decay rate
    beta2: float = 0.999 - Second moment decay rate
    weight_decay: float = 0.01 - Weight decay coefficient
    eps: float = 1e-8 - Small constant for numerical stability

    """
    # Write code here
    w    = np.asarray(w, dtype = float)
    m    = np.asarray(m, dtype = float)
    v    = np.asarray(v, dtype = float)
    grad = np.asarray(grad, dtype=float)

    m = beta1 * m + (1 - beta1)* grad
    v = beta2 * v + (1 - beta2)* (grad**2)
    w = w - lr *weight_decay * w - lr * m / (np.sqrt(v) + eps)

    return w, m, v