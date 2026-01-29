import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pos = np.arange(seq_length)
    idx = np.arange(d_model)    
    even_idx = np.arange(0, d_model, 2)
    div_term = np.exp(-np.log(10_000)* even_idx / d_model )
    PE = np.zeros((seq_length, d_model))
    PE[:, 0::2] = np.sin(pos[:, None]*div_term)
    PE[:, 1::2] = np.cos(pos[:, None]*div_term)

    return PE
