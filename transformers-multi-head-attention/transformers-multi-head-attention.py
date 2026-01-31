import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def split_heads(X: np.ndarray, num_heads: int) -> np.ndarray:
    B, N, d_model = X.shape
    assert d_model % num_heads == 0, "d_model should be divisible by num_heads"
    d_k = d_model // num_heads
    # Reshape
    X = X.reshape(B, N, num_heads, d_k)
    # Transpose
    X = X.transpose(0, 2, 1, 3)
    return X

def concat_heads(X: np.ndarray) -> np.ndarray:
    B, h, N, d_k = X.shape
    d_model = h * d_k
    # Transpose
    X = X.transpose(0, 2, 1, 3)
    # Reshape
    X = X.reshape(B, N, d_model)
    return X

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    Q_heads = split_heads(Q_proj, num_heads)
    K_heads = split_heads(K_proj, num_heads)
    V_heads = split_heads(V_proj, num_heads)
    d_k = Q_heads.shape[-1]

    scores = (Q_heads @ K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores, axis = -1)
    Ah = weights @ V_heads
    A = concat_heads(Ah)

    output = A @ W_o

    return output



