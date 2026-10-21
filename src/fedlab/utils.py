"""Seeding, device selection and small I/O helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Must be set before the first cuBLAS call for deterministic GEMMs on CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch global RNGs.

    Flower's ``sample_nodes`` draws from Python's global ``random`` module and
    its DP wrappers draw noise from NumPy's global RNG, so both must be seeded.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def derive_seed(*parts: int | str) -> int:
    """Derive a 31-bit seed from a tuple, stable across processes and platforms."""
    digest = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "cuda":
        raise RuntimeError("CUDA requested but not available")
    return torch.device("cpu")


def data_root() -> Path:
    """Directory holding the cached numpy copies of the datasets."""
    return Path(os.environ.get("FEDLAB_DATA", Path(__file__).resolve().parents[2] / "data"))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")
