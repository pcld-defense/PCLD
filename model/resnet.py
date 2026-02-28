## Original GitHub: https://github.com/kuangliu/pytorch-cifar

"""ResNet in PyTorch (CIFAR-10 variant).

Reference:
    Kaiming He et al., "Deep Residual Learning for Image Recognition",
    arXiv:1512.03385.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two-conv residual block for ResNet-18 and ResNet-34 (CIFAR-10 variant).

    Each block applies two 3×3 convolutions with batch normalisation and a
    skip connection. A projection shortcut is added when the stride or channel
    count changes.
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """Builds the two-conv residual block.

        Args:
            in_planes: Number of input channels.
            planes: Number of output channels for both convolutions.
            stride: Stride for the first convolution and the shortcut projection.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies two convolutions with a residual skip connection.

        Args:
            x: Input feature map of shape (B, C_in, H, W).

        Returns:
            Output feature map of shape (B, C_out, H', W').
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Three-conv bottleneck block for ResNet-50 and deeper (CIFAR-10 variant).

    Applies a 1×1 reduction, a 3×3 spatial convolution, and a 1×1 expansion
    with a factor-4 channel expansion defined by `expansion`.
    """

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """Builds the bottleneck block.

        Args:
            in_planes: Number of input channels.
            planes: Number of internal (bottleneck) channels; output channels
                are planes * expansion.
            stride: Stride for the 3×3 conv and the shortcut projection.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the bottleneck block with a residual skip connection.

        Args:
            x: Input feature map of shape (B, C_in, H, W).

        Returns:
            Output feature map of shape (B, C_out, H', W').
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """Generic ResNet backbone for CIFAR-10 (32×32 input).

    Starts with a single 3×3 conv (no maxpool) and ends with global average
    pooling and a fully-connected classification head.
    """

    def __init__(self, block: type, num_blocks: list[int],
                 num_classes: int = 10) -> None:
        """Builds the ResNet backbone.

        Args:
            block: Residual block class (BasicBlock or Bottleneck).
            num_blocks: List of four ints specifying the number of blocks per
                layer group.
            num_classes: Number of output classes for the linear head.
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: type, planes: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        """Stacks residual blocks into a sequential layer group.

        Args:
            block: Block class to instantiate.
            planes: Number of output channels for this group.
            num_blocks: Number of blocks to stack.
            stride: Stride applied to the first block only.

        Returns:
            Sequential container of residual blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the full ResNet forward pass.

        Args:
            x: Input image batch of shape (B, 3, 32, 32).

        Returns:
            Class logits of shape (B, num_classes).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18() -> ResNet:
    """Constructs a ResNet-18 for CIFAR-10."""
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34() -> ResNet:
    """Constructs a ResNet-34 for CIFAR-10."""
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50() -> ResNet:
    """Constructs a ResNet-50 for CIFAR-10."""
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101() -> ResNet:
    """Constructs a ResNet-101 for CIFAR-10."""
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152() -> ResNet:
    """Constructs a ResNet-152 for CIFAR-10."""
    return ResNet(Bottleneck, [3, 8, 36, 3])
