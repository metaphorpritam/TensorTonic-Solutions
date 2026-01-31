import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    Q = Q.float()
    K = K.float()
    V = V.float()
    d_k = Q.shape[-1]
    K_T = K.transpose(-2, -1)
    scores = Q @ K_T
    scores = scores / math.sqrt(d_k)
    weights = F.softmax(scores, dim = -1)
    output = weights @ V
    return output
   