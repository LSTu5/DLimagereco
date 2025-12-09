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
    def __init__(self, num_classes=2, stem_channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channels, 3, stride=2, padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = FreqBlock(stem_channels, 64, stride=2)   # 56x56
        self.layer2 = FreqBlock(64, 128, stride=2)             # 28x28
        self.layer3 = FreqBlock(128, 256, stride=2)            # 14x14
        self.layer4 = FreqBlock(256, 512, stride=2)            # 7x7

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feats = self.pool(x).flatten(1)
        return self.fc(feats)