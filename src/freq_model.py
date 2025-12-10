"""Frequency branch CNN model for AI-generated image detection.

This module defines a convolutional neural network that operates on frequency-domain
images (FFT magnitude spectra) to detect AI-generated content. The frequency domain
can reveal artifacts and patterns that are invisible in the spatial domain, such as:
- Periodic patterns from GAN upsampling layers
- Compression artifacts
- Unnatural frequency distributions

The architecture mirrors the spatial branch but processes single-channel grayscale
FFT images instead of RGB.
"""

import torch.nn as nn
import torch.nn.functional as F

class FreqBlock(nn.Module):
    """Residual block for frequency-domain feature extraction.
    
    Identical structure to BasicBlock in spatial_model.py, but named separately
    to distinguish frequency-specific processing. Uses the same residual design:
    - Two 3x3 convolutions with batch norm
    - Skip connection for gradient flow
    - Optional projection for dimension matching
    
    Args:
        in_channels: Number of input feature channels
        out_channels: Number of output feature channels
        stride: Stride for first conv (default: 1). Use 2 for downsampling.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Main pathway: two conv layers with batch norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection for dimension matching when needed
        self.down = None
        if stride != 1 or in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """Forward pass with residual connection."""
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(x)
        return F.relu(out + identity)

class FrequencyBranch(nn.Module):
    """CNN for classifying real vs AI-generated images from frequency data.
    
    This network processes FFT magnitude spectra to detect AI generation artifacts.
    Many generative models leave characteristic signatures in the frequency domain:
    - GANs often produce checkerboard patterns from transposed convolutions
    - Diffusion models may have specific frequency biases
    - Compressed/processed AI images show distinct spectral patterns
    
    Architecture:
    - Input: (batch, 1, 320, 320) grayscale FFT magnitude images
    - Progressive downsampling through 4 residual layers
    - Global average pooling + linear classifier
    - Output: (batch, 2) logits for real/fake classes
    
    Args:
        num_classes: Number of output classes (default: 2)
        stem_channels: Number of channels after initial conv (default: 32)
    """
    def __init__(self, num_classes=2, stem_channels=32):
        super().__init__()
        # Stem: Initial processing of single-channel FFT input
        # 320x320 -> 160x160 (using stride=2 for downsampling)
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        
        # Progressive feature extraction with downsampling
        self.layer1 = FreqBlock(stem_channels, 64, stride=2)   # 160x160 -> 80x80
        self.layer2 = FreqBlock(64, 128, stride=2)             # 80x80 -> 40x40
        self.layer3 = FreqBlock(128, 256, stride=2)            # 40x40 -> 20x20
        self.layer4 = FreqBlock(256, 512, stride=2)            # 20x20 -> 10x10

        # Global pooling and classification
        self.pool = nn.AdaptiveAvgPool2d(1)  # Any size -> 1x1
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """Forward pass through the frequency branch.
        
        Args:
            x: Input tensor of shape (batch, 1, 320, 320) - FFT magnitude
            
        Returns:
            logits: Output tensor of shape (batch, num_classes)
        """
        # Feature extraction pathway
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Aggregate and classify
        feats = self.pool(x).flatten(1)  # (batch, 512, 1, 1) -> (batch, 512)
        return self.fc(feats)