"""Same seed -> identical results on a real (small) MNIST federation."""

from __future__ import annotations

import numpy as np

from fedlab.runner import run_experiment

CFG = {
    "dataset": "mnist",
    "partition": "dirichlet",
    "alpha": 0.5,
    "num_clients": 4,
    "fraction": 0.5,
    "rounds": 2,
    "local_epochs": 1,
    "batch_size": 32,
    "lr": 0.05,
    "eval_every": 1,
    "device": "cpu",
    "max_train": 1200,
}


def _key(res: dict) -> list:
    return [(h["round"], h["acc"], h["loss"], h["bytes_up"]) for h in res["history"]]


def test_same_seed_same_result() -> None:
    a = run_experiment({**CFG, "seed": 7})
    b = run_experiment({**CFG, "seed": 7})
    assert _key(a) == _key(b)
    c = run_experiment({**CFG, "seed": 8})
    assert _key(a) != _key(c)
    assert np.isfinite(a["history"][-1]["loss"])
