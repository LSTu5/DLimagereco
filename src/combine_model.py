import torch
import torch.nn as nn

from spatial_model import SpatialBranch
from freq_model import FrequencyBranch


class DualBranchFusion(nn.Module):
    """
    Fusion model that runs the spatial and frequency branches in parallel and
    combines them either by averaging logits (simple ensemble) or by
    concatenating pooled features and learning a small MLP classifier.
    """

    def __init__(
        self,
        num_classes: int = 2,
        fusion: str = "concat",
        hidden_dim: int = 512,
        freeze_backbones: bool = True,
    ) -> None:
        super().__init__()
        fusion = fusion.lower()
        if fusion not in {"concat", "avg"}:
            raise ValueError("fusion must be 'concat' or 'avg'")
        self.fusion = fusion

        self.spatial = SpatialBranch(num_classes=num_classes)
        self.frequency = FrequencyBranch(num_classes=num_classes)

        if freeze_backbones:
            for param in list(self.spatial.parameters()) + list(self.frequency.parameters()):
                param.requires_grad = False

        if self.fusion == "concat":
            # Spatial and frequency branches each output 512-d pooled features.
            fused_dim = 512 * 2
            self.head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = None

    def forward(
        self,
        spatial_x: torch.Tensor,
        freq_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if freq_x is None:
            freq_x = spatial_x

        if self.fusion == "avg":
            spatial_logits = self.spatial(spatial_x)
            freq_logits = self.frequency(freq_x)
            return (spatial_logits + freq_logits) * 0.5

        # concat fusion: grab pooled features from both branches
        spatial_feats, _ = self.spatial(spatial_x, return_features=True)
        spatial_repr = self.spatial.pool(spatial_feats).flatten(1)

        freq_feats, _ = self.frequency(freq_x, return_features=True)
        freq_repr = self.frequency.pool(freq_feats).flatten(1)

        fused = torch.cat([spatial_repr, freq_repr], dim=1)
        return self.head(fused)

    def load_branch_weights(self, spatial_ckpt: str, freq_ckpt: str, map_location="cpu") -> None:
        """Convenience helper to load pretrained weights for both branches."""
        spatial_state = torch.load(spatial_ckpt, map_location=map_location, weights_only=True)
        freq_state = torch.load(freq_ckpt, map_location=map_location, weights_only=True)
        self.spatial.load_state_dict(spatial_state)
        self.frequency.load_state_dict(freq_state)

