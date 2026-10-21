"""Models from the papers being reproduced.

* ``CNNMnist``: McMahan et al. (2017), Sec. 3: two 5x5 conv layers (32, 64
  channels), each followed by 2x2 max pooling, a 512-unit ReLU FC layer and a
  softmax output; 1,663,370 parameters (checked in tests).
* ``CNNCifar``: the TensorFlow CIFAR-10 tutorial model that McMahan et al. use
  for CIFAR-10 (conv5x5-64, pool3x3/2, LRN, conv5x5-64, LRN, pool3x3/2, FC 384,
  FC 192, linear) on 24x24 crops; about 10^6 parameters.
* ``CNNFemnist``: LEAF's FEMNIST CNN (models/femnist/cnn.py): conv5x5-32,
  pool2x2, conv5x5-64, pool2x2, FC 2048, 62 outputs.

Layers are written from those descriptions; no code is copied.
"""

from __future__ import annotations

import torch
from torch import nn


class _TwoConv(nn.Module):
    def __init__(self, in_ch: int, hidden: int, num_classes: int, side: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * (side // 4) ** 2, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


class CNNMnist(_TwoConv):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(1, 512, num_classes, 28)


class CNNFemnist(_TwoConv):
    def __init__(self, num_classes: int = 62) -> None:
        super().__init__(1, 2048, num_classes, 28)


class CNNCifar(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # TF LRN(depth_radius=4, alpha=0.001/9, beta=0.75, bias=1) sums over
        # 2*4+1 = 9 channels and scales by alpha; torch divides alpha by size,
        # so the equivalent torch alpha is 0.001.
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.lrn = nn.LocalResponseNorm(9, alpha=0.001, beta=0.75, k=1.0)
        self.fc1 = nn.Linear(6 * 6 * 64, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lrn(self.pool(torch.relu(self.conv1(x))))
        x = self.pool(self.lrn(torch.relu(self.conv2(x))))
        x = torch.relu(self.fc1(x.flatten(1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


MODELS = {"mnist": CNNMnist, "cifar10": CNNCifar, "femnist": CNNFemnist}


def build_model(dataset: str) -> nn.Module:
    return MODELS[dataset]()


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
