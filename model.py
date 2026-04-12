"""Model definitions for CIFAR-10 energy-based classifier."""

from typing import Callable

import torch
import torch.nn as nn
import torchvision.models as models


class BasicBlock(nn.Module):
    """Basic residual block used in WideResNet."""

    def __init__(self, in_planes: int, out_planes: int, stride: int, dropout_rate: float = 0.0):
        super().__init__()
        self.equal_in_out = in_planes == out_planes

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.shortcut = None

        if not self.equal_in_out:
            self.shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        residual = x if self.equal_in_out else self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + residual


class NetworkBlock(nn.Module):
    """Group of residual blocks for WideResNet."""

    def __init__(
        self,
        num_layers: int,
        in_planes: int,
        out_planes: int,
        block: Callable[..., nn.Module],
        stride: int,
        dropout_rate: float,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                block(
                    in_planes=in_planes if i == 0 else out_planes,
                    out_planes=out_planes,
                    stride=stride if i == 0 else 1,
                    dropout_rate=dropout_rate,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class WideResNet(nn.Module):
    """WideResNet implementation (e.g., WRN-28-2 for CIFAR-10)."""

    def __init__(self, depth: int = 28, widen_factor: int = 2, dropout_rate: float = 0.0, num_classes: int = 10):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth should satisfy (depth - 4) % 6 == 0")

        n = (depth - 4) // 6
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], BasicBlock, stride=1, dropout_rate=dropout_rate)
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], BasicBlock, stride=2, dropout_rate=dropout_rate)
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], BasicBlock, stride=2, dropout_rate=dropout_rate)
        self.bn = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def build_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """Build a ResNet-18 adjusted for CIFAR-10 (3x3 stem, no maxpool)."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(model_name: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Factory function returning a model configured for CIFAR-10."""
    model_name = model_name.lower()
    if model_name == "resnet18":
        return build_resnet18_cifar10(num_classes=num_classes)
    if model_name in {"wideresnet", "wideresnet28-2", "wrn28-2"}:
        return WideResNet(depth=28, widen_factor=2, dropout_rate=0.0, num_classes=num_classes)
    raise ValueError(f"Unsupported model_name: {model_name}")
