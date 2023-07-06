from typing import Sequence
from dataclasses import dataclass
import math
import torch
import torch.nn as nn

@dataclass
class EfficientNetConfig:
    block: str
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    expand_ratio: int
    squeeze_ratio: int
    num_blocks: int

class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        channels: int,
        squeeze_ratio: int,
    ) -> None:
        super().__init__()

        mid_channels = channels // squeeze_ratio

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Conv2d(channels, mid_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(mid_channels, channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)
        self.scale_activation = nn.Sigmoid()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)

        return scale * x

class MBConv(nn.Module):
    def __init__(
        self,
        size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        squeeze_ratio: int
    ) -> None:
        super().__init__()

        mid_channels = in_channels * expand_ratio

        if stride == 1:
            if kernel_size % 2 == 0:
                kernel_size -= 1
        elif stride == 2:
            if kernel_size % 2 == 0 and size % 2 == 1:
                kernel_size -= 1
            elif kernel_size % 2 == 1 and size % 2 == 0:
                kernel_size += 1
        else:
            raise ValueError('stride must be 1 or 2')

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.silu = nn.SiLU(inplace=True)
        self.se = SqueezeExcitation(in_channels, squeeze_ratio)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=mid_channels,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

        if stride == 2:
            if size % 2 == 0:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False)
        elif in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.bn1(x)
        out = self.silu(out)
        shortcut = x if self.shortcut is None else self.shortcut(out)
        out = self.se(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.silu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.silu(out)
        out = self.conv3(out)

        return out + shortcut

class FusedMBConv(nn.Module):
    def __init__(
        self,
        size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        squeeze_ratio: int
    ) -> None:
        super().__init__()

        mid_channels = in_channels * expand_ratio

        if stride == 1:
            if kernel_size % 2 == 0:
                kernel_size -= 1
        elif stride == 2:
            if kernel_size % 2 == 0 and size % 2 == 1:
                kernel_size -= 1
            elif kernel_size % 2 == 1 and size % 2 == 0:
                kernel_size += 1
        else:
            raise ValueError('stride must be 1 or 2')

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.silu = nn.SiLU(inplace=True)
        self.se = SqueezeExcitation(in_channels, squeeze_ratio)
        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

        if stride == 2:
            if size % 2 == 0:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False)
        elif in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.bn1(x)
        out = self.silu(out)
        shortcut = x if self.shortcut is None else self.shortcut(out)
        out = self.se(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.silu(out)
        out = self.conv2(out)

        return out + shortcut

class EfficientNet(nn.Module):
    def __init__(
        self,
        configs: Sequence[EfficientNetConfig],
        size: int = 224,
        in_channels: int = 3,
        out_channels: int = 1280,
        num_classes: int = 1000,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.size = size

        self.conv1 = nn.Conv2d(in_channels, configs[0].in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(configs[-1].out_channels)
        self.silu = nn.SiLU(inplace=True)

        layers = []
        for config in configs:
            layers.append(self._make_layer(config))
        self.layers = nn.Sequential(*layers)

        self.conv2 = nn.Conv2d(configs[-1].out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(out_channels, num_classes)

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
        if config.block not in ['mbconv', 'fusedmbconv']:
            raise ValueError('block only supports {MBConv|FusedMBConv}')
        block = MBConv if config.block == 'mbconv' else FusedMBConv

        layers = []
        layers.append(block(self.size, config.in_channels, config.out_channels, config.kernel_size, config.stride, config.expand_ratio, config.squeeze_ratio))

        if config.stride == 2:
            self.size = self.size // 2 if self.size % 2 == 0 else self.size // 2 + 1

        for _ in range(1, config.num_blocks):
            layers.append(block(self.size, config.out_channels, config.out_channels, config.kernel_size, 1, config.expand_ratio, config.squeeze_ratio))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layers(out)
        out = self.bn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.silu(out)

        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.dropout(out)
        out = self.fc(out)

        return out

def efficientnet_b0(
        size: int = 224,
        in_channels: int = 3,
        out_channels: int = 1280,
        num_classes: int = 1000,
        dropout: float = 0.0
) -> EfficientNet:
    return EfficientNet(
        configs = [
            EfficientNetConfig('mbconv',  32,  16, 3, 1, 1, 4, 1),
            EfficientNetConfig('mbconv',  16,  24, 3, 2, 6, 4, 2),
            EfficientNetConfig('mbconv',  24,  40, 3, 1, 6, 4, 2),
            EfficientNetConfig('mbconv',  40,  80, 3, 2, 6, 4, 3),
            EfficientNetConfig('mbconv',  80, 112, 3, 1, 6, 4, 3),
            EfficientNetConfig('mbconv', 112, 192, 3, 2, 6, 4, 4),
            EfficientNetConfig('mbconv', 192, 320, 3, 1, 6, 4, 1),
        ],
        size = size,
        in_channels = in_channels,
        out_channels = out_channels,
        num_classes = num_classes,
        dropout = dropout
    )

def efficientnetv2_s(
        size: int = 224,
        in_channels: int = 3,
        out_channels: int = 1280,
        num_classes: int = 1000,
        dropout: float = 0.0
) -> EfficientNet:
    return EfficientNet(
        configs = [
            EfficientNetConfig('fusedmbconv',  24,  24, 3, 1, 1, 4,  2),
            EfficientNetConfig('fusedmbconv',  24,  48, 3, 2, 4, 4,  4),
            EfficientNetConfig('fusedmbconv',  48,  64, 3, 1, 4, 4,  4),
            EfficientNetConfig(     'mbconv',  64, 128, 3, 2, 4, 4,  6),
            EfficientNetConfig(     'mbconv', 128, 160, 3, 1, 6, 4,  9),
            EfficientNetConfig(     'mbconv', 160, 256, 3, 2, 6, 4, 15),
        ],
        size = size,
        in_channels = in_channels,
        out_channels = out_channels,
        num_classes = num_classes,
        dropout = dropout
    )
