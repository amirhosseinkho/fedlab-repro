"""Pilot and diagnostic runs behind the choices in docs/decisions.md.

None of these are headline results. Pilots use seed 100, which no reported
experiment uses. Output: results/phase_checks/<name>.json

    phase1   MNIST quick config reaches >= 95% (acceptance of phase 1)
    lr       learning-rate pilot for CIFAR-10 (D19)
    femnist  learning-rate pilot for FEMNIST (D19)
    scaffold SCAFFOLD weighting diagnostic (D12)
    scaffold_lr0p1  the same partial-participation runs at lr 0.1 (D19)
    stability  SCAFFOLD full-participation stability at lower lr (D19)
    norms    client update norms, for the central-DP clip (D20)
    dp       central-DP / DP-SGD pilots (D20)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from fedlab.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase_checks"

CIFAR = {
    "dataset": "cifar10",
    "partition": "dirichlet",
    "num_clients": 100,
    "fraction": 0.1,
    "batch_size": 50,
    "lr": 0.03,  # D19: largest lr at which SCAFFOLD stayed stable
    "eval_every": 5,
    "device": "cuda",
}


def checks() -> dict[str, dict[str, dict]]:
    e1 = yaml.safe_load((ROOT / "configs" / "e1.yaml").read_text(encoding="utf-8"))
    phase1 = {
        **e1["base"],
        "alpha": 0.5,
        "algorithm": "fedavg",
        **e1["quick"]["set"],
        "max_test": None,
        "seed": 0,
    }
    lr = {
        f"lr_pilot_lr{lr}_s100": {
            **CIFAR,
            "alpha": 0.5,
            "local_epochs": 2,
            "rounds": 30,
            "eval_every": 10,
            "lr": lr,
            "seed": 100,
        }
        for lr in (0.02, 0.05, 0.1)
    }
    part = {**CIFAR, "alpha": 0.1, "local_epochs": 2, "rounds": 60, "lr": 0.05, "seed": 0}
    full = {
        **CIFAR,
        "alpha": 0.1,
        "num_clients": 10,
        "fraction": 1.0,
        "local_epochs": 1,
        "rounds": 30,
        "lr": 0.05,
        "seed": 0,
    }
    variants = {
        "fedavg": {"algorithm": "fedavg"},
        "fedavg-uniform": {"algorithm": "fedavg", "uniform_weights": True},
        "fedprox": {"algorithm": "fedprox", "mu": 0.01},
        "scaffold-uniform": {"algorithm": "scaffold", "scaffold_weighted": False},
        "scaffold-weighted": {"algorithm": "scaffold", "scaffold_weighted": True},
    }
    scaffold = {f"phase2_partial_{k}_alpha0p1_s0": {**part, **v} for k, v in variants.items()}
    scaffold_lr0p1 = {
        f"phase2_partial_{k}_alpha0p1_lr0p1_s0": {**part, **v, "lr": 0.1}
        for k, v in variants.items()
    }
    scaffold |= {
        f"phase2_fullpart_{k}_alpha0p1_s0": {**full, **v}
        for k, v in variants.items()
        if k != "fedprox"
    }
    # SCAFFOLD diverged in the full-participation run at lr 0.05; find the
    # largest lr at which it stays stable over twice as many rounds.
    stability = {
        f"phase2_fullpart_{k}_alpha0p1_lr{str(lr).replace('.', 'p')}_r60_s0": {
            **full,
            **variants[k],
            "lr": lr,
            "rounds": 60,
        }
        for lr in (0.03, 0.02)
        for k in ("scaffold-weighted", "fedavg")
    }
    # The E3 point with the most local steps, and SCAFFOLD on FEMNIST.
    stability["scaffold_e3_E5_frac0p2_lr0p03_s100"] = {
        **CIFAR,
        "algorithm": "scaffold",
        "alpha": 0.5,
        "local_epochs": 5,
        "fraction": 0.2,
        "rounds": 40,
        "eval_every": 10,
        "seed": 100,
    }
    stability["femnist_scaffold_lr0p05_s100"] = {
        **CIFAR,
        "algorithm": "scaffold",
        "dataset": "femnist",
        "partition": "natural",
        "alpha": None,
        "batch_size": 20,
        "local_epochs": 2,
        "rounds": 60,
        "eval_every": 10,
        "lr": 0.05,
        "seed": 100,
    }
    dp_base = {
        **CIFAR,
        "alpha": 0.5,
        "local_epochs": 1,
        "rounds": 40,
        "eval_every": 10,
        "seed": 100,
    }
    norms = {
        f"phase3_update_norms_E{e}_s100": {**dp_base, "local_epochs": e, "rounds": 20}
        for e in (1, 5)
    }
    dp = {"phase3_nodp_s100": dp_base}
    dp |= {
        f"phase3_central_z{z}_s100": {**dp_base, "central_dp_noise": z, "central_dp_clip": 0.3}
        for z in (0.0, 0.01, 0.05, 0.1, 0.5)
    }
    dp |= {
        f"phase3_dpsgd_s{s}_s100": {**dp_base, "dpsgd_noise": s, "dpsgd_clip": 1.0}
        for s in (0.5, 1.0, 1.5)
    }
    # Clipping each per-sample gradient to norm 1 makes the step much smaller
    # than in non-private SGD, so DP-SGD gets its own lr pilot (D20).
    dp |= {
        f"phase3_dpsgd_s1.0_lr{str(lr).replace('.', 'p')}_s100": {
            **dp_base,
            "dpsgd_noise": 1.0,
            "dpsgd_clip": 1.0,
            "lr": lr,
        }
        for lr in (0.1, 0.3, 1.0)
    }
    femnist = {
        f"femnist_lr_pilot_lr{lr}_s100": {
            **CIFAR,
            "dataset": "femnist",
            "partition": "natural",
            "alpha": None,
            "batch_size": 20,
            "local_epochs": 2,
            "rounds": 30,
            "eval_every": 10,
            "lr": lr,
            "seed": 100,
        }
        for lr in (0.05, 0.1, 0.2)
    }
    return {
        "phase1": {"phase1_mnist_quick_fedavg_s0": phase1},
        "lr": lr,
        "femnist": femnist,
        "scaffold": scaffold,
        "scaffold_lr0p1": scaffold_lr0p1,
        "stability": stability,
        "norms": norms,
        "dp": dp,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "which",
        nargs="*",
        default=[
            "phase1",
            "lr",
            "femnist",
            "scaffold_lr0p1",
            "scaffold",
            "stability",
            "norms",
            "dp",
        ],
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for group in args.which:
        for name, cfg in checks()[group].items():
            out = OUT / f"{name}.json"
            if out.exists():
                continue
            r = run_experiment(cfg, out)
            accs = [round(h["acc"], 3) for h in r["history"]]
            eps = {k: round(v, 2) for k, v in r["history"][-1].items() if k.startswith("eps")}
            print(
                f"{name}: {accs} last3={r['final_acc_last3']:.4f} {eps} {r['wall_seconds']:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
