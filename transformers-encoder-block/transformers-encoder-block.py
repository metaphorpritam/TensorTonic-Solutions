import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    B, N, d_model = x.shape
    assert d_model % num_heads == 0, "d_model should be divisible by num_heads"
    d_k = d_model // num_heads
    x = x.reshape(B, N, num_heads, d_k)
    x = x.transpose(0, 2, 1, 3)
    return x

def concat_heads(x: np.ndarray) -> np.ndarray:
    B, num_heads, N, d_k = x.shape
    d_model = num_heads * d_k
    x = x.transpose(0, 2, 1, 3)
    x = x.reshape(B, N, d_model)
    return x


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mean_x = np.mean(x, axis = -1, keepdims = True)
    var_x = np.var(x, axis = -1, keepdims = True)
    x_out = gamma*(x - mean_x)/np.sqrt(var_x + eps) + beta
    return x_out

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v
    Q_heads = split_heads(Q_proj, num_heads)
    K_heads = split_heads(K_proj, num_heads)
    V_heads = split_heads(V_proj, num_heads)
    d_k = Q_proj.shape[-1]
    prod = Q_heads @ K_heads.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    scores = softmax(prod, axis = -1)
    A_h = scores @ V_heads
    A = concat_heads(A_h)
    out = A @ W_o
    return A

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    hidden = x @ W1 + b1
    hidden = np.maximum(0, hidden)
    out = hidden @ W2 + b2
    return out

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    x_mha = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + x_mha, gamma1, beta1)
    x_ff = feed_forward(x, W1, b1, W2, b2)
    x = layer_norm(x + x_ff, gamma2, beta2)
    return x
