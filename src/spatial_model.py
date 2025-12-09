import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
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
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down is not None:
            identity = self.down(x)

        out += identity
        return F.relu(out)


class SpatialBranch(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # NOTE: Keeping layer1 stride=2 as per your original code
        # (It was set to 1 in the final pasted code, but 2 is more common ResNet style)
        self.layer1 = BasicBlock(32, 64, stride=2)  # 56x56
        self.layer2 = BasicBlock(64, 128, stride=2)  # 28x28
        self.layer3 = BasicBlock(128, 256, stride=2)  # 14x14
        self.layer4 = BasicBlock(256, 512, stride=2)  # 7x7

        self.pool = nn.AdaptiveAvgPool2d(1)

        # ADDED DROPOUT for regularization
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x).flatten(1)

        # Apply Dropout before the classifier
        x = self.dropout(x)

        return self.fc(x)

    # ADDED Kaiming He Initialization for training stability
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)