import torch
import torch.nn as nn

from spatial_model import SpatialBranch
from freq_model import FrequencyBranch


class FusionModel(nn.Module):
    """
    Dual-stream model that pools features from the spatial and frequency branches
    and fuses them with a small MLP head.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4, freeze_backbones: bool = False):
        super().__init__()
        self.spatial = SpatialBranch(num_classes=num_classes)
        self.freq = FrequencyBranch(num_classes=num_classes)

        if freeze_backbones:
            for p in self.spatial.parameters():
                p.requires_grad = False
            for p in self.freq.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fuse = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, rgb: torch.Tensor, fft_mag: torch.Tensor):
        s_feats, _ = self.spatial(rgb, return_features=True)   # (B, 512, Hs, Ws)
        f_feats, _ = self.freq(fft_mag, return_features=True)  # (B, 512, Hf, Wf)

        s_vec = self.pool(s_feats).flatten(1)
        f_vec = self.pool(f_feats).flatten(1)
        fused = torch.cat([s_vec, f_vec], dim=1)
        return self.fuse(fused)
