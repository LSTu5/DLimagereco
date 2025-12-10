"""Spatial branch CNN model for AI-generated image detection.

This module defines a ResNet-inspired convolutional neural network that operates
on spatial RGB images to detect AI-generated content. The architecture uses:
- Residual connections to enable deeper networks and better gradient flow
- Progressive downsampling to build hierarchical feature representations
- Batch normalization for training stability
- Dropout for regularization
"""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Residual block with two convolutional layers.
    
    This block implements the core building unit of ResNet-style architectures:
    - Two 3x3 convolutions with batch norm
    - Skip connection from input to output
    - Optional downsampling via stride and 1x1 projection
    
    The residual connection helps with:
    - Training deeper networks (addresses vanishing gradient)
    - Learning identity mappings when beneficial
    - Faster convergence
    
    Args:
        in_channels: Number of input feature channels
        out_channels: Number of output feature channels
        stride: Stride for first conv (default: 1). Use 2 for downsampling.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # First conv: potentially downsamples via stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second conv: maintains spatial dimensions
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut for dimension matching when:
        # 1. Spatial dimensions change (stride != 1)
        # 2. Channel count changes (in_channels != out_channels)
        self.down = None
        if stride != 1 or in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch, out_channels, height//stride, width//stride)
        """
        identity = x
        # Main path: conv -> bn -> relu -> conv -> bn
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Adjust identity dimensions if needed
        if self.down is not None:
            identity = self.down(x)
        # Add residual and apply final activation
        return F.relu(out + identity)


class SpatialBranch(nn.Module):
    """Lightweight CNN for classifying real vs AI-generated images from RGB data.
    
    Architecture overview:
    - Input: (batch, 3, 224, 224) RGB images
    - Stem: Initial downsampling to 112x112 with 32 channels
    - Layer 1-4: Progressive feature extraction with residual blocks
      * Each layer doubles channels and halves spatial dimensions
      * Final feature maps: 512 channels at 7x7 resolution
    - Global average pooling: Aggregates spatial information
    - Dropout + Linear: Classification head with regularization
    - Output: (batch, 2) logits for real/fake classes
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
    """
    def __init__(self, num_classes=2):
        super().__init__()

        # Stem: Initial feature extraction with aggressive downsampling
        # 224x224 -> 112x112
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Progressive downsampling and feature extraction
        # Each layer: 2x channels, 0.5x spatial dimensions
        self.layer1 = BasicBlock(32, 64, stride=2)    # 112x112 -> 56x56
        self.layer2 = BasicBlock(64, 128, stride=2)   # 56x56 -> 28x28
        self.layer3 = BasicBlock(128, 256, stride=2)  # 28x28 -> 14x14
        self.layer4 = BasicBlock(256, 512, stride=2)  # 14x14 -> 7x7

        # Global pooling: Aggregate spatial features to (batch, 512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Regularization to prevent overfitting
        self.dropout = nn.Dropout(p=0.5)
        
        # Classification head
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights for better training dynamics
        self._init_weights()

    def forward(self, x):
        """Forward pass through the spatial branch.
        
        Args:
            x: Input tensor of shape (batch, 3, 224, 224)
            
        Returns:
            logits: Output tensor of shape (batch, num_classes)
        """
        # Feature extraction pathway
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Aggregate spatial information and classify
        x = self.pool(x).flatten(1)  # (batch, 512, 1, 1) -> (batch, 512)
        x = self.dropout(x)
        return self.fc(x)

    def _init_weights(self):
        """Initialize network weights using best practices.
        
        - Conv layers: Kaiming initialization (accounts for ReLU nonlinearity)
        - BatchNorm: Ones for weight, zeros for bias
        - Linear: Small random weights to break symmetry
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming init helps maintain activation variance through ReLU layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)