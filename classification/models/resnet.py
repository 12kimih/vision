from typing import Sequence
from dataclasses import dataclass
import math
import torch
import torch.nn as nn

@dataclass
class Config:
    block: str
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    expand_ratio: int
    num_blocks: int

class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int
    ) -> None:
        super().__init__()
        mid_channels = out_channels // expand_ratio

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if stride > 1 or in_channels != out_channels else None
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut

class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int
    ) -> None:
        super().__init__()
        mid_channels = out_channels // expand_ratio

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if stride > 1 or in_channels != out_channels else None
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False)

        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        return out + shortcut

class ResNet(nn.Module):
    def __init__(
        self,
        configs: Sequence[Config],
        in_channels: int,
        num_classes: int,
        dropout: float
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, configs[0].in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        layers = []
        for config in configs:
            layers.append(self._make_layer(config))
        self.layers = nn.Sequential(*layers)
        self.bn1 = nn.BatchNorm2d(configs[-1].out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(configs[-1].out_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _make_layer(self, config):
        config.block = config.block.lower()
        if config.block not in ['basicblock', 'bottleneck']:
            raise ValueError('block only supports {BasicBlock|Bottleneck}')
        block = BasicBlock if config.block == 'basicblock' else Bottleneck

        layers = []
        layers.append(block(config.in_channels, config.out_channels, config.kernel_size, config.stride, config.expand_ratio))
        for _ in range(1, config.num_blocks):
            layers.append(block(config.out_channels, config.out_channels, config.kernel_size, 1, config.expand_ratio))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layers(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.dropout(out)
        out = self.fc(out)

        return out

def resnet18(
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout: float = 0.0
) -> ResNet:
    return ResNet(
        configs = [
            Config('basicblock',  64,  64, 3, 1, 1, 2),
            Config('basicblock',  64, 128, 3, 2, 1, 2),
            Config('basicblock', 128, 256, 3, 2, 1, 2),
            Config('basicblock', 256, 512, 3, 2, 1, 2)
        ],
        in_channels = in_channels,
        num_classes = num_classes,
        dropout = dropout
    )

def resnet34(
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout: float = 0.0
) -> ResNet:
    return ResNet(
        configs = [
            Config('basicblock',  64,  64, 3, 1, 1, 3),
            Config('basicblock',  64, 128, 3, 2, 1, 4),
            Config('basicblock', 128, 256, 3, 2, 1, 6),
            Config('basicblock', 256, 512, 3, 2, 1, 3)
        ],
        in_channels = in_channels,
        num_classes = num_classes,
        dropout = dropout
    )

def resnet50(
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout: float = 0.0
) -> ResNet:
    return ResNet(
        configs = [
            Config('bottleneck',   64,  256, 3, 1, 4, 3),
            Config('bottleneck',  256,  512, 3, 2, 4, 4),
            Config('bottleneck',  512, 1024, 3, 2, 4, 6),
            Config('bottleneck', 1024, 2048, 3, 2, 4, 3)
        ],
        in_channels = in_channels,
        num_classes = num_classes,
        dropout = dropout
    )

def resnet101(
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout: float = 0.0
) -> ResNet:
    return ResNet(
        configs = [
            Config('bottleneck',   64,  256, 3, 1, 4,  3),
            Config('bottleneck',  256,  512, 3, 2, 4,  4),
            Config('bottleneck',  512, 1024, 3, 2, 4, 23),
            Config('bottleneck', 1024, 2048, 3, 2, 4,  3)
        ],
        in_channels = in_channels,
        num_classes = num_classes,
        dropout = dropout
    )

def resnet152(
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout: float = 0.0
) -> ResNet:
    return ResNet(
        configs = [
            Config('bottleneck',   64,  256, 3, 1, 4,  3),
            Config('bottleneck',  256,  512, 3, 2, 4,  8),
            Config('bottleneck',  512, 1024, 3, 2, 4, 36),
            Config('bottleneck', 1024, 2048, 3, 2, 4,  3)
        ],
        in_channels = in_channels,
        num_classes = num_classes,
        dropout = dropout
    )
