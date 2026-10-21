"""Figures for the README, drawn only from files under results/.

Run ``experiments/aggregate.py`` first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fedlab.data import (
    label_histogram,
    load_arrays,
    partition_dirichlet,
    partition_shards,
    partition_writers,
)  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# First three slots of the validated categorical palette (all-pairs safe).
COLORS = {"fedavg": "#2a78d6", "fedprox": "#eb6834", "scaffold": "#1baf7a"}
NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD"}
DP_COLORS = {"central": "#2a78d6", "dpsgd": "#eb6834"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "font.size": 10,
        "legend.frameon": False,
    }
)


def _save(fig, name: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (out / name).relative_to(ROOT).as_posix())


def partitions(out: Path) -> None:
    """Class proportions per client for the partitions used (seed 0)."""
    cifar = load_arrays("cifar10")["y_train"]
    mnist = load_arrays("mnist")["y_train"]
    panels = [
        (f"CIFAR-10, Dirichlet alpha={a:g}", cifar, partition_dirichlet(cifar, 100, a, seed=0), 10)
        for a in (0.1, 0.5, 1.0, 100.0)
    ]
    panels.append(("MNIST, 2 shards per client", mnist, partition_shards(mnist, 100, seed=0), 10))
    fem = load_arrays("femnist")
    parts, _ = partition_writers(fem["writer_train"], 100, subset_seed=0)
    panels.append(("FEMNIST, 100 writers (62 classes)", fem["y_train"], parts, 62))
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
    for ax, (title, y, p, k) in zip(axes.flat, panels, strict=True):
        h = label_histogram(y, p, k).astype(float)
        sizes = h.sum(1)
        prop = h / np.maximum(sizes[:, None], 1)
        order = np.argsort(-sizes)[:50]
        im = ax.imshow(
            prop[order].T, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("client (50 largest, by size)")
        ax.set_ylabel("class")
        ax.grid(False)
        ax.text(
            0.99,
            0.02,
            f"samples/client: min {int(sizes.min())}, max {int(sizes.max())}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=INK2,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )
    fig.colorbar(im, ax=axes, shrink=0.8, label="share of the client's samples")
    _save(fig, "partitions.png", out)


def _curves(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hs = [json.loads(f.read_text())["history"] for f in files]
    rounds = np.array([x["round"] for x in hs[0]])
    acc = np.array([[x["acc"] for x in h] for h in hs])
    return rounds, acc.mean(0), acc.std(0, ddof=1) if len(hs) > 1 else np.zeros(len(rounds))


def e1_curves(res: Path, out: Path) -> None:
    sets = [
        ("cifar10__alpha0p1", "CIFAR-10, alpha=0.1"),
        ("cifar10__alpha0p5", "CIFAR-10, alpha=0.5"),
        ("femnist", "FEMNIST, 100 writers"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, (prefix, title) in zip(axes, sets, strict=True):
        for algo in COLORS:
            pat = (
                f"{prefix}_algorithm{algo}__s*.json"
                if "alpha" in prefix
                else f"{prefix}__algorithm{algo}__s*.json"
            )
            files = sorted((res / "e1").glob(pat))
            if not files:
                continue
            r, m, s = _curves(files)
            ax.plot(r, m * 100, color=COLORS[algo], label=f"{NAMES[algo]} (n={len(files)})")
            ax.fill_between(
                r, (m - s) * 100, (m + s) * 100, color=COLORS[algo], alpha=0.15, linewidth=0
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("round")
        ax.set_ylabel("test accuracy (%)")
        ax.legend(loc="lower right")
    _save(fig, "e1_curves.png", out)


def e2_alpha(res: Path, out: Path) -> None:
    s = pd.concat(
        [
            pd.read_csv(res / f"summary_{e}.csv")
            for e in ("e1", "e2")
            if (res / f"summary_{e}.csv").exists()
        ]
    )
    s = s[(s.dataset != "femnist") & s.algorithm.isin(["fedavg", "scaffold"])]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for algo in ("fedavg", "scaffold"):
        part = s[s.algorithm == algo].sort_values("alpha")
        ax.errorbar(
            part.alpha,
            part.acc_last3 * 100,
            yerr=part.acc_last3_std.fillna(0) * 100,
            color=COLORS[algo],
            marker="o",
            markersize=7,
            capsize=3,
            label=f"{NAMES[algo]} (n={int(part.n_seeds.min())} seeds)",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet alpha (smaller = more label skew)")
    ax.set_ylabel("final test accuracy (%)")
    ax.legend(loc="lower right")
    _save(fig, "e2_alpha.png", out)


def _pareto(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of points not dominated (smaller x, larger y), sorted by x."""
    idx = np.argsort(x)
    front, best = [], -np.inf
    for i in idx:
        if y[i] > best:
            front.append(i)
            best = y[i]
    return np.array(front, dtype=int)


def e3(res: Path, out: Path) -> None:
    s = pd.read_csv(res / "summary_e3.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    nd = s[s.group == "nodp"]
    markers = {1: "o", 5: "s"}
    for algo in COLORS:
        for e, mk in markers.items():
            p = nd[(nd.algorithm == algo) & (nd.local_epochs == e)]
            ax.scatter(
                p.bytes_total / 1e9,
                p.acc_last3 * 100,
                color=COLORS[algo],
                marker=mk,
                s=60,
                edgecolors=SURFACE,
                linewidths=1.5,
                zorder=3,
            )
    f = _pareto(nd.bytes_total.values, nd.acc_last3.values)
    ax.plot(
        nd.bytes_total.values[f] / 1e9,
        nd.acc_last3.values[f] * 100,
        color=INK2,
        linewidth=1,
        linestyle="--",
        zorder=2,
        label="Pareto front",
    )
    for algo in COLORS:
        ax.scatter([], [], color=COLORS[algo], marker="o", s=60, label=NAMES[algo])
    for e, mk in markers.items():
        ax.scatter([], [], color=INK2, marker=mk, s=60, label=f"E={e}")
    ax.set_xscale("log")
    ax.set_xlabel("total bytes exchanged, up + down (GB, log)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"No DP: accuracy vs communication ({int(nd.rounds.max())} rounds)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    ax = axes[1]
    nodp_ref = nd[(nd.algorithm == "fedavg") & (nd.local_epochs == 1)]
    for frac, mk in zip((0.05, 0.1, 0.2), ("v", "o", "^"), strict=True):
        c = s[(s.group == "central") & (s.fraction == frac) & (s.central_dp_noise > 0)]
        ax.scatter(
            c.eps_central_poisson,
            c.acc_last3 * 100,
            color=DP_COLORS["central"],
            marker=mk,
            s=60,
            edgecolors=SURFACE,
            linewidths=1.5,
            zorder=3,
        )
        d = s[(s.group == "dpsgd") & (s.fraction == frac)]
        ax.scatter(
            d.eps_sample_max,
            d.acc_last3 * 100,
            color=DP_COLORS["dpsgd"],
            marker=mk,
            s=60,
            edgecolors=SURFACE,
            linewidths=1.5,
            zorder=3,
        )
        ax.scatter(
            [], [], color=INK2, marker=mk, s=60, label=f"{int(round(frac * 100))} clients/round"
        )
    ax.scatter(
        [],
        [],
        color=DP_COLORS["central"],
        s=60,
        label="user-level central DP (eps, Poisson-assumed)",
    )
    ax.scatter([], [], color=DP_COLORS["dpsgd"], s=60, label="sample-level DP-SGD (max client eps)")
    if not nodp_ref.empty:
        ax.axhline(
            nodp_ref.acc_last3.max() * 100,
            color=INK2,
            linewidth=1,
            linestyle=":",
            label="best no-DP FedAvg, E=1",
        )
    ax.axhline(10, color=GRID, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("epsilon at delta=1e-5 (log)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"DP: accuracy vs privacy budget ({int(s.rounds.max())} rounds)", fontsize=10)
    ax.legend(loc="center right", fontsize=8)
    _save(fig, "e3_tradeoff.png", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    res = ROOT / "results" / ("quick" if args.quick else "")
    out = ROOT / ("figures/quick" if args.quick else "figures")
    jobs = {
        "partitions": lambda: partitions(out),
        "e1": lambda: e1_curves(res, out),
        "e2": lambda: e2_alpha(res, out),
        "e3": lambda: e3(res, out),
    }
    for name, job in jobs.items():
        if args.only and name != args.only:
            continue
        if args.quick and name == "partitions":
            continue
        job()


if __name__ == "__main__":
    main()
