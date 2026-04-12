import numpy as np

class BatchNorm:
    """Batch Normalization layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply batch normalization.
        """
        # YOUR CODE HERE
        if training:
            # 1. Compute batch mean and variance along the batch dimension
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # 2. Update running statistics using an exponential moving average
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Use batch statistics for normalization during training
            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics for normalization during inference
            mean = self.running_mean
            var = self.running_var
            
        # 3. Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # 4. Scale and shift (using learnable parameters gamma and beta)
        out = self.gamma * x_norm + self.beta

        return out

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

def post_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Post-activation ResNet block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Uses x @ W for "convolution" (simplified as linear transform).
    """
    # YOUR CODE HERE
    # Main path F(x)
    out = x @ W1
    out = bn1.forward(out)
    out = relu(out)

    out = out @ W2
    out = bn2.forward(out)
    out = relu(out)

    out = out + x
    out = relu(out)

    return out

    

def pre_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Pre-activation ResNet block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    This ordering often works better for very deep networks.
    """
    # YOUR CODE HERE
    # Main path F(x)
    out = bn1.forward(x)
    out = relu(out)
    out = out @ W1
    
    out = bn2.forward(out)
    out = relu(out)
    out = out @ W2
    
    # Skip connection (Addition happens at the very end)
    out = out + x
    
    return out
