import numpy as np

def relu(x):
    return np.maximum(0, x)

class BasicBlock:
    """Basic residual block (2 conv layers with skip connection)."""
    
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        self.downsample = downsample
        self.W1 = np.random.randn(in_ch, out_ch) * 0.01
        self.W2 = np.random.randn(out_ch, out_ch) * 0.01
        # Projection shortcut if dimensions change
        self.W_proj = np.random.randn(in_ch, out_ch) * 0.01 if in_ch != out_ch or downsample else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Conv -> ReLU -> Conv -> Add Skip -> ReLU
        """
        # YOUR CODE HERE
        # Main path F(x)
        out = x @ self.W1
        out = relu(out)
        out = out @ self.W2
        
        # Skip connection (Shortcut)
        if self.W_proj is not None:
            # If dimensions changed, project x to match the output
            shortcut = x @ self.W_proj
        else:
            # If dimensions match, just use x directly
            shortcut = x
            
        # Add skip connection BEFORE the final ReLU
        out = out + shortcut
        out = relu(out)
        
        return out

class ResNet18:
    """
    Simplified ResNet-18 architecture.
    
    Structure:
    - conv1: 3 -> 64 channels
    - layer1: 2 BasicBlocks, 64 channels
    - layer2: 2 BasicBlocks, 128 channels (first block downsamples)
    - layer3: 2 BasicBlocks, 256 channels (first block downsamples)
    - layer4: 2 BasicBlocks, 512 channels (first block downsamples)
    - fc: 512 -> num_classes
    """
    
    def __init__(self, num_classes: int = 10):
        # Initial Convolution
        self.conv1 = np.random.randn(3, 64) * 0.01
        
        # Build layers - FIX: Each layer needs 2 blocks as per the docstring!
        # Layer 1: 64 -> 64 (No downsampling needed anywhere here)
        self.layer1 = [
            BasicBlock(64, 64, downsample=False),
            BasicBlock(64, 64, downsample=False)
        ]
        
        # Layer 2: 64 -> 128 (First block downsamples, second block keeps 128)
        self.layer2 = [
            BasicBlock(64, 128, downsample=True),
            BasicBlock(128, 128, downsample=False)
        ]
        
        # Layer 3: 128 -> 256 (First block downsamples, second block keeps 256)
        self.layer3 = [
            BasicBlock(128, 256, downsample=True),
            BasicBlock(256, 256, downsample=False)
        ]
        
        # Layer 4: 256 -> 512 (First block downsamples, second block keeps 512)
        self.layer4 = [
            BasicBlock(256, 512, downsample=True),
            BasicBlock(512, 512, downsample=False)
        ]
        
        # Final Fully Connected layer
        self.fc = np.random.randn(512, num_classes) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through ResNet-18.
        """
        # YOUR CODE HERE
        out = x @ self.conv1
        out = relu(out)
        
        # 2. Pass through all 4 layers (and their respective blocks)
        # We can combine them into a single list to iterate cleanly
        all_blocks = self.layer1 + self.layer2 + self.layer3 + self.layer4
        
        for block in all_blocks:
            out = block.forward(out)
            
        # 3. Final Fully Connected Layer (Simplifying pooling for this setup)
        out = out @ self.fc
        
        return out
