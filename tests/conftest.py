from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from fedlab.data import FederatedData

CPU = torch.device("cpu")


def synthetic_data(
    sizes: list[int], side: int = 2, num_classes: int = 2, seed: int = 0, n_test: int = 20
) -> FederatedData:
    """Random 1-channel ``side`` x ``side`` images split into clients of ``sizes``."""
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    x = torch.from_numpy(rng.integers(0, 256, size=(n, 1, side, side), dtype=np.uint8))
    y = torch.from_numpy(rng.integers(0, num_classes, size=n).astype(np.int64))
    xt = torch.from_numpy(rng.integers(0, 256, size=(n_test, 1, side, side), dtype=np.uint8))
    yt = torch.from_numpy(rng.integers(0, num_classes, size=n_test).astype(np.int64))
    offsets = np.cumsum([0, *sizes])
    return FederatedData(
        name="synthetic",
        x_train=x,
        y_train=y,
        x_test=xt,
        y_test=yt,
        client_indices=[np.arange(offsets[i], offsets[i + 1]) for i in range(len(sizes))],
        mean=torch.full((1, 1, 1, 1), 0.5),
        std=torch.full((1, 1, 1, 1), 0.25),
        crop=None,
        num_classes=num_classes,
    )


class Tiny(nn.Module):
    """Linear softmax classifier on flattened inputs."""

    def __init__(self, d: int = 4, k: int = 2, zero: bool = False) -> None:
        super().__init__()
        self.fc = nn.Linear(d, k)
        if zero:
            nn.init.zeros_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


@pytest.fixture
def tiny_cfg() -> dict:
    return {
        "dataset": "synthetic",
        "fraction": 1.0,
        "rounds": 2,
        "local_epochs": 1,
        "batch_size": 8,
        "lr": 0.1,
        "eval_every": 1,
        "device": "cpu",
        "seed": 0,
    }
