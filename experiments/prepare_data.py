"""Download MNIST, CIFAR-10 and FEMNIST with flwr-datasets and cache them."""

from __future__ import annotations

import argparse
import time

from fedlab.data import HF_IDS, load_arrays, prepare


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(HF_IDS))
    args = ap.parse_args()
    for name in args.datasets:
        t = time.time()
        path = prepare(name)
        arr = load_arrays(name)
        shapes = {k: v.shape for k, v in arr.items()}
        print(f"{name}: {path} {shapes} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
