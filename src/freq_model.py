import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.down = None
        if stride != 1 or in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(x)
        return F.relu(out + identity)


class FrequencyBranch(nn.Module):
    """
    Frequency branch using fixed high-pass filters (SRM-style) followed by a shallow CNN.
    More robust to resize/compression than full DFT magnitude.
    """

    def __init__(self, num_classes=2, stem_channels=32):
        super().__init__()
        # Fixed high-pass filter bank (5 filters)
        kernels = torch.tensor([
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]],  # Laplacian
            [[-1, 2, -1],
             [-2, 4, -2],
             [-1, 2, -1]],  # Vertical-ish
            [[-1, -2, -1],
             [2, 4, 2],
             [-1, -2, -1]],  # Horizontal-ish
            [[2, -1, -1],
             [-1, 2, -1],
             [-1, -1, 2]],  # Diagonal
            [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]],   # Sharpen/high-pass
        ], dtype=torch.float32).unsqueeze(1)  # (5,1,3,3)

        self.hpf = nn.Conv2d(1, kernels.shape[0], kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.hpf.weight.copy_(kernels)
        for p in self.hpf.parameters():
            p.requires_grad = False

        in_ch = kernels.shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )
        self.layer1 = FreqBlock(stem_channels, 64, stride=2)
        self.layer2 = FreqBlock(64, 128, stride=2)
        self.layer3 = FreqBlock(128, 256, stride=2)
        self.layer4 = FreqBlock(256, 512, stride=2)

        self.proj = nn.Sequential(
            nn.Conv2d(512, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def forward(self, x, return_features=False):
        # x: (B, C, H, W); if RGB, convert to grayscale
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)

        # Fixed HPF bank
        x = self.hpf(x)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feats = self.proj(x)

        if return_features:
            pooled = self.pool(feats).flatten(1)
            pooled = self.dropout(pooled)
            logits = self.fc(pooled)
            return feats, logits

        pooled = self.pool(feats).flatten(1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)