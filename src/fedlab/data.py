"""Datasets, partitioners and on-device client batching.

Raw data is downloaded once with flwr-datasets (``FederatedDataset``) and cached
as uint8 numpy arrays. Partitions are computed with flwr-datasets partitioners
run on a label-only ``datasets.Dataset`` (images are not needed to partition),
then kept as index arrays so that every client batch is a slice of a tensor that
already sits on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fedlab.utils import data_root

# Hugging Face IDs that flwr-datasets 0.6.1 lists as tested
# (flwr_datasets/utils.py), plus flwrlabs/femnist (BSD-2-Clause, dataset card).
HF_IDS = {
    "mnist": "ylecun/mnist",
    "cifar10": "uoft-cs/cifar10",
    "femnist": "flwrlabs/femnist",
}
IMAGE_COLUMN = {"mnist": "image", "cifar10": "img", "femnist": "image"}
LABEL_COLUMN = {"mnist": "label", "cifar10": "label", "femnist": "character"}
NUM_CLASSES = {"mnist": 10, "cifar10": 10, "femnist": 62}


def cache_path(name: str) -> Path:
    return data_root() / f"{name}.npz"


def _raw_parquet(name: str) -> dict[str, str] | None:
    """Local parquet copies (same files as on the Hub), if they were fetched."""
    d = data_root() / "raw" / name
    files = {split: d / f"{split}-00000-of-00001.parquet" for split in ("train", "test")}
    found = {k: str(v) for k, v in files.items() if v.exists()}
    return found or None


def prepare(name: str) -> Path:
    """Load ``name`` with flwr-datasets and cache it as uint8 numpy arrays.

    The data comes from the Hugging Face Hub (``HF_IDS``), or from local copies
    of the Hub's parquet files under ``<data_root>/raw/<name>/`` when present
    (used here because the Hub download was slow; the files are checked against
    the Hub's SHA-256 by ``experiments/fetch_data.sh``).
    """
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import IidPartitioner

    out = cache_path(name)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    local = _raw_parquet(name)
    source = {"dataset": "parquet", "data_files": local} if local else {"dataset": HF_IDS[name]}
    fds = FederatedDataset(
        partitioners={"train": IidPartitioner(1)},
        shuffle=False,
        **source,  # keep canonical order
    )
    arrays: dict[str, np.ndarray] = {}
    splits = ["train"] if name == "femnist" else ["train", "test"]
    for split in splits:
        ds = fds.load_split(split).with_format("numpy")
        arrays[f"x_{split}"] = np.ascontiguousarray(ds[IMAGE_COLUMN[name]]).astype(np.uint8)
        arrays[f"y_{split}"] = np.asarray(ds[LABEL_COLUMN[name]]).astype(np.int64)
        if name == "femnist":
            arrays["writer_train"] = np.asarray(ds["writer_id"]).astype(str)
    np.savez(out, **arrays)
    return out


def load_arrays(name: str) -> dict[str, np.ndarray]:
    path = cache_path(name)
    if not path.exists():
        prepare(name)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------
# Partitioners (thin wrappers around flwr-datasets)
# --------------------------------------------------------------------------


def _label_dataset(labels: np.ndarray, column: str = "label"):
    import datasets

    return datasets.Dataset.from_dict(
        {column: labels.tolist(), "idx": np.arange(len(labels)).tolist()}
    )


def _indices(partitioner, num_partitions: int) -> list[np.ndarray]:
    return [
        np.asarray(partitioner.load_partition(i)["idx"], dtype=np.int64)
        for i in range(num_partitions)
    ]


def partition_dirichlet(
    labels: np.ndarray, num_clients: int, alpha: float, seed: int, min_partition_size: int = 2
) -> list[np.ndarray]:
    """Label skew of Hsu, Qi and Brown (2019): one Dirichlet(alpha) draw over
    clients per class (``DirichletPartitioner`` with ``self_balancing=False``)."""
    from flwr_datasets.partitioner import DirichletPartitioner

    p = DirichletPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        alpha=float(alpha),
        min_partition_size=min_partition_size,
        self_balancing=False,
        shuffle=True,
        seed=seed,
    )
    p.dataset = _label_dataset(labels)
    return _indices(p, num_clients)


def partition_shards(
    labels: np.ndarray, num_clients: int, seed: int, shards_per_client: int = 2
) -> list[np.ndarray]:
    """Pathological split of McMahan et al. (2017): sort by label, cut into
    ``num_clients * shards_per_client`` equal shards, give each client 2."""
    from flwr_datasets.partitioner import ShardPartitioner

    p = ShardPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        num_shards_per_partition=shards_per_client,
        shard_size=len(labels) // (num_clients * shards_per_client),
        shuffle=True,
        seed=seed,
    )
    p.dataset = _label_dataset(labels)
    return _indices(p, num_clients)


def partition_iid(labels: np.ndarray, num_clients: int, seed: int) -> list[np.ndarray]:
    from flwr_datasets.partitioner import IidPartitioner

    p = IidPartitioner(num_partitions=num_clients)
    perm = np.random.default_rng(seed).permutation(len(labels))
    p.dataset = _label_dataset(labels[perm])
    return [perm[ix] for ix in _indices(p, num_clients)]


def partition_writers(
    writers: np.ndarray,
    num_writers: int,
    subset_seed: int,
    test_fraction: float = 0.1,
    min_samples: int = 20,
) -> tuple[list[np.ndarray], np.ndarray]:
    """FEMNIST natural split by ``writer_id`` (``NaturalIdPartitioner``).

    A fixed random subset of ``num_writers`` writers (chosen with ``subset_seed``)
    becomes the client population. Each writer keeps ``1 - test_fraction`` of its
    samples for training; the held-out parts form the central test set, which
    mirrors LEAF's per-writer train/test split.
    """
    from flwr_datasets.partitioner import NaturalIdPartitioner

    p = NaturalIdPartitioner(partition_by="writer_id")
    import datasets

    p.dataset = datasets.Dataset.from_dict(
        {"writer_id": writers.tolist(), "idx": np.arange(len(writers)).tolist()}
    )
    all_parts = _indices(p, p.num_partitions)
    eligible = [i for i, ix in enumerate(all_parts) if len(ix) >= min_samples]
    rng = np.random.default_rng(subset_seed)
    chosen = sorted(rng.choice(eligible, size=num_writers, replace=False).tolist())
    train_parts, test_parts = [], []
    for i in chosen:
        ix = rng.permutation(all_parts[i])
        n_test = max(1, int(round(test_fraction * len(ix))))
        test_parts.append(ix[:n_test])
        train_parts.append(np.sort(ix[n_test:]))
    return train_parts, np.sort(np.concatenate(test_parts))


def label_histogram(labels: np.ndarray, parts: list[np.ndarray], num_classes: int) -> np.ndarray:
    """Clients x classes count matrix."""
    return np.stack([np.bincount(labels[ix], minlength=num_classes) for ix in parts])


# --------------------------------------------------------------------------
# On-device federated dataset
# --------------------------------------------------------------------------


@dataclass
class FederatedData:
    """Training/test tensors on ``device`` plus per-client index arrays."""

    name: str
    x_train: torch.Tensor  # uint8, NCHW
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    client_indices: list[np.ndarray]
    mean: torch.Tensor
    std: torch.Tensor
    crop: int | None  # random-crop size for training (CIFAR-10), None = no augmentation
    num_classes: int

    @property
    def num_clients(self) -> int:
        return len(self.client_indices)

    def client_size(self, cid: int) -> int:
        return len(self.client_indices[cid])

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() / 255.0 - self.mean) / self.std

    def train_batches(self, cid: int, batch_size: int, epochs: int, gen: torch.Generator):
        """Yield shuffled, augmented minibatches of client ``cid``.

        Each epoch is split into ``ceil(n / batch_size)`` batches of nearly equal
        size (sizes differ by at most one) instead of full batches plus a short
        remainder: a remainder of one or two samples, stepped with the same
        learning rate, can dominate a client's update (docs/decisions.md D21).

        All randomness comes from ``gen`` (a CPU generator), so the batch order
        and augmentation are identical on CPU and GPU for the same seed.
        """
        ix = torch.as_tensor(self.client_indices[cid])
        n = len(ix)
        num_batches = -(-n // batch_size)
        for _ in range(epochs):
            perm = ix[torch.randperm(n, generator=gen)].to(self.x_train.device)
            for b in torch.tensor_split(perm, num_batches):
                yield self.augment(self.x_train[b], gen), self.y_train[b]

    def augment(self, xb: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        x = self._normalise(xb)
        if self.crop is None:
            return x
        return random_crop_flip(x, self.crop, gen)

    def client_tensors(self, cid: int) -> tuple[torch.Tensor, torch.Tensor]:
        b = torch.as_tensor(self.client_indices[cid]).to(self.x_train.device)
        return self.x_train[b], self.y_train[b]

    def test_batches(self, batch_size: int = 1000):
        for s in range(0, len(self.y_test), batch_size):
            x = self._normalise(self.x_test[s : s + batch_size])
            if self.crop is not None:
                x = center_crop(x, self.crop)
            yield x, self.y_test[s : s + batch_size]


def random_crop_flip(x: torch.Tensor, size: int, gen: torch.Generator) -> torch.Tensor:
    """Per-sample random ``size`` x ``size`` crop and horizontal flip (NCHW)."""
    b, c, h, w = x.shape
    dev = x.device
    i = torch.randint(0, h - size + 1, (b,), generator=gen).to(dev)
    j = torch.randint(0, w - size + 1, (b,), generator=gen).to(dev)
    flip = (torch.rand(b, generator=gen) < 0.5).to(dev)
    ar = torch.arange(size, device=dev)
    rows = (i[:, None] + ar)[:, None, :, None]
    cols = (j[:, None] + ar)[:, None, None, :]
    out = x[
        torch.arange(b, device=dev)[:, None, None, None],
        torch.arange(c, device=dev)[None, :, None, None],
        rows,
        cols,
    ]
    return torch.where(flip[:, None, None, None], out.flip(3), out)


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top : top + size, left : left + size]


def _to_nchw(x: np.ndarray) -> np.ndarray:
    return x[:, None] if x.ndim == 3 else x.transpose(0, 3, 1, 2)


def build_federated_data(
    name: str,
    partition: str,
    num_clients: int,
    seed: int,
    device: torch.device,
    alpha: float | None = None,
    femnist_subset_seed: int = 0,
    max_train: int | None = None,
    max_test: int | None = None,
) -> FederatedData:
    """Load ``name`` and split it across ``num_clients``.

    ``partition`` is one of ``dirichlet``, ``shards``, ``iid`` or ``natural``
    (FEMNIST writers). ``max_train`` keeps only the first ``max_train`` training
    samples and ``max_test`` the first ``max_test`` test samples (used by
    ``--quick`` runs).
    """
    arr = load_arrays(name)
    if name == "femnist":
        if partition != "natural":
            raise ValueError("FEMNIST is only used with its natural writer split")
        parts, test_ix = partition_writers(arr["writer_train"], num_clients, femnist_subset_seed)
        x_all, y_all = arr["x_train"], arr["y_train"]
        x_test, y_test = x_all[test_ix], y_all[test_ix]
        # Keep only the chosen writers' samples and re-index the partitions.
        used = np.concatenate(parts)
        x_train, y_train = x_all[used], y_all[used]
        offsets = np.cumsum([0] + [len(p) for p in parts])
        parts = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(parts))]
    else:
        x_train, y_train = arr["x_train"], arr["y_train"]
        x_test, y_test = arr["x_test"], arr["y_test"]
        if max_train is not None:
            x_train, y_train = x_train[:max_train], y_train[:max_train]
        if partition == "dirichlet":
            assert alpha is not None
            parts = partition_dirichlet(y_train, num_clients, alpha, seed)
        elif partition == "shards":
            parts = partition_shards(y_train, num_clients, seed)
        elif partition == "iid":
            parts = partition_iid(y_train, num_clients, seed)
        else:
            raise ValueError(f"unknown partition {partition!r}")

    if max_test is not None:
        x_test, y_test = x_test[:max_test], y_test[:max_test]
    x_train_t = torch.from_numpy(_to_nchw(x_train)).contiguous()
    # Normalisation statistics from the training images of the client population.
    used = np.concatenate(parts)
    sub = x_train_t[torch.as_tensor(used[:: max(1, len(used) // 20000)])].float() / 255.0
    mean = sub.mean(dim=(0, 2, 3)).view(1, -1, 1, 1)
    std = sub.std(dim=(0, 2, 3)).view(1, -1, 1, 1)
    return FederatedData(
        name=name,
        x_train=x_train_t.to(device),
        y_train=torch.from_numpy(y_train).to(device),
        x_test=torch.from_numpy(_to_nchw(x_test)).contiguous().to(device),
        y_test=torch.from_numpy(y_test).to(device),
        client_indices=parts,
        mean=mean.to(device),
        std=std.to(device),
        crop=24 if name == "cifar10" else None,
        num_classes=NUM_CLASSES[name],
    )
