import numpy as np

def relu(x):
    return np.maximum(0, x)

def linear_channels(x, W):
    """
    Applies channel-wise linear projection.

    If x is 4D: (B, C_in, H, W) -> (B, C_out, H, W)
    If x is 2D: (B, C_in)      -> (B, C_out)
    """
    if x.ndim == 4:
        B, C, H, W_ = x.shape
        x_flat = x.transpose(0, 2, 3, 1)   # (B, H, W, C_in)
        y = x_flat @ W                     # (B, H, W, C_out)
        y = y.transpose(0, 3, 1, 2)        # (B, C_out, H, W)
        return y

    elif x.ndim == 2:
        # Dense-style input
        return x @ W

    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")



class ConvBlock:
    """
    Convolutional Block with projection shortcut.
    Used when input/output dimensions differ.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main path weights
        self.W1 = np.random.randn(in_channels, out_channels) * 0.01
        self.W2 = np.random.randn(out_channels, out_channels) * 0.01
        
        # Shortcut projection (1x1 conv equivalent)
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with projection shortcut.
        """
        # YOUR CODE HERE
        # Main path
        h = relu(linear_channels(x, self.W1))
        z = linear_channels(h, self.W2)

        # Shortcut path (projection)
        s = linear_channels(x, self.Ws)

        # Combine
        return relu(z + s)
        
