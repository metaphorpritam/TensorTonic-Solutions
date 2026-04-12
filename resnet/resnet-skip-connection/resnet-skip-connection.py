import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    # YOUR CODE HERE
    grad = x.copy()

    for g in reversed(gradients_F):
        I = np.eye(g.shape[0])
        grad = grad @ (I + g)

    return grad

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    # YOUR CODE HERE
    grad = x.copy()

    for g in reversed(gradients_F):
        grad = grad @ g

    return grad
