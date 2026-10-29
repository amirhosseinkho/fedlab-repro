"""Collect run logs into per-run and per-configuration tables.

Outputs (under ``results/`` or ``results/quick/``):

* ``runs.csv``           one row per run (every seed)
* ``summary_<exp>.csv``  mean, std and n over seeds for every configuration
* ``tables.md``          the Markdown tables quoted in the README
* ``compute.csv``        wall-clock seconds per experiment
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (0.4, 0.5, 0.6)
KEYS = [
    "exp",
    "group",
    "dataset",
    "alpha",
    "algorithm",
    "local_epochs",
    "fraction",
    "central_dp_noise",
    "dpsgd_noise",
    "rounds",
]
ALGO = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD"}


def load_runs(results: Path) -> pd.DataFrame:
    rows = []
    for exp in ("e1", "e2", "e3"):
        for f in sorted((results / exp).glob("*.json")):
            if f.name == "timing.json":
                continue
            r = json.loads(f.read_text(encoding="utf-8"))
            c, h = r["config"], r["history"]
            last = h[-1]
            row = {
                "exp": exp,
                "file": f"{exp}/{f.name}",
                "group": f.name.split("__")[0],
                **{k: c.get(k) for k in KEYS[2:]},
                "seed": c["seed"],
                "acc_last3": r["final_acc_last3"],
                "acc_final": r["final_acc"],
                "acc_best": max(x["acc"] for x in h),
                # the training loss became NaN; the model then predicts one class
                "diverged": any(math.isnan(x["loss"]) for x in h),
                "bytes_total": r["total_bytes_raw"],
                "bytes_up": last["bytes_up"],
                "bytes_down": last["bytes_down"],
                "num_params": r["num_params"],
                "num_sampled": r["num_sampled"],
                "wall_seconds": r["wall_seconds"],
                "eps_central_noamp": last.get("eps_central_no-amplification"),
                "eps_central_poisson": last.get("eps_central_poisson"),
                "eps_sample_max": last.get("eps_sample_max"),
                "eps_sample_mean": last.get("eps_sample_mean_participants"),
            }
            for t in TARGETS:
                hit = next((x for x in h if x["acc"] >= t), None)
                row[f"bytes_to_{int(t * 100)}"] = (
                    None if hit is None else hit["bytes_up"] + hit["bytes_down"]
                )
                row[f"rounds_to_{int(t * 100)}"] = None if hit is None else hit["round"]
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        for k in ("central_dp_noise", "dpsgd_noise", "alpha"):
            df[k] = df[k].astype("float64")
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in KEYS if k in df]
    metrics = [c for c in df.columns if c not in keys + ["file", "seed"]]
    g = df.groupby(keys, dropna=False)
    out = g[metrics].mean(numeric_only=True)
    std = g[["acc_last3", "acc_final"]].std(ddof=1).add_suffix("_std")
    n = g.size().rename("n_seeds")
    div = g["diverged"].sum().rename("n_diverged")
    return pd.concat([out.drop(columns="diverged"), std, n, div], axis=1).reset_index()


def pm(mean: float, std: float, n: int, scale: float = 100.0) -> str:
    if n < 2 or np.isnan(std):
        return f"{mean * scale:.1f}"
    return f"{mean * scale:.1f} ± {std * scale:.1f}"


def cell(r) -> str:
    """Accuracy cell; notes how many seeds diverged (NaN loss)."""
    text = pm(r.acc_last3, r.acc_last3_std, r.n_seeds)
    if r.n_diverged:
        text += f" ({int(r.n_diverged)} of {int(r.n_seeds)} diverged)"
    return text


def mb(x: float | None) -> str:
    return (
        "not reached" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x / 1e6:,.0f}"
    )


def eps(x: float | None) -> str:
    if x is None or np.isnan(x):
        return "-"
    if np.isinf(x):
        return "inf"
    return f"{x:,.1f}" if x < 1000 else f"{x:,.0f}"


def tables(s: pd.DataFrame) -> str:
    out = []
    e1 = s[s.exp == "e1"]
    if not e1.empty:
        out.append("### E1: final test accuracy (%), mean ± std over seeds\n")
        out.append("| dataset | partition | FedAvg | FedProx (mu=0.01) | SCAFFOLD | seeds |")
        out.append("|---|---|---|---|---|---|")
        for (ds, a), part in e1.groupby(["dataset", "alpha"], dropna=False):
            cells = {r.algorithm: cell(r) for r in part.itertuples()}
            label = "writers (natural)" if ds == "femnist" else f"Dirichlet alpha={a:g}"
            out.append(
                f"| {ds} | {label} | {cells.get('fedavg', 'not run')} | {cells.get('fedprox', 'not run')} "
                f"| {cells.get('scaffold', 'not run')} | {int(part.n_seeds.min())} |"
            )
        out.append("")
    e12 = s[
        (s.exp.isin(["e1", "e2"]))
        & (s.dataset == "cifar10")
        & (s.algorithm.isin(["fedavg", "scaffold"]))
    ]
    if not e12.empty:
        out.append("### E2: CIFAR-10 accuracy (%) vs Dirichlet alpha\n")
        out.append("| alpha | FedAvg | SCAFFOLD | SCAFFOLD - FedAvg | seeds |")
        out.append("|---|---|---|---|---|")
        for a, part in e12.groupby("alpha"):
            cells = {r.algorithm: r for r in part.itertuples()}
            fa, sc = cells.get("fedavg"), cells.get("scaffold")
            diff = (
                f"{(sc.acc_last3 - fa.acc_last3) * 100:+.1f}"
                if fa is not None and sc is not None
                else "-"
            )
            out.append(
                f"| {a:g} | {cell(fa) if fa is not None else 'not run'} "
                f"| {cell(sc) if sc is not None else 'not run'} | {diff} "
                f"| {int(part.n_seeds.min())} |"
            )
        out.append("")
    e3 = s[s.exp == "e3"]
    if not e3.empty:
        nd = e3[e3.group == "nodp"].sort_values(["algorithm", "local_epochs", "fraction"])
        out.append(
            f"### E3a: communication (no DP), CIFAR-10 alpha=0.5, {int(nd.rounds.max())} rounds\n"
        )
        out.append(
            "| algorithm | E | clients/round | accuracy (%) | total MB | MB to 40% | MB to 50% | seeds |"
        )
        out.append("|---|---|---|---|---|---|---|---|")
        for r in nd.itertuples():
            out.append(
                f"| {ALGO[r.algorithm]} | {r.local_epochs} | {int(r.num_sampled)} | {cell(r)} "
                f"| {mb(r.bytes_total)} | {mb(r.bytes_to_40)} | {mb(r.bytes_to_50)} | {int(r.n_seeds)} |"
            )
        out.append("")
        cd = e3[e3.group == "central"].sort_values(["fraction", "central_dp_noise"])
        if not cd.empty:
            out.append(
                f"### E3b: user-level central DP (DP-FedAvg), delta=1e-5, {int(cd.rounds.max())} rounds\n"
            )
            out.append(
                "| clients/round | noise z | accuracy (%) | epsilon, no amplification | epsilon, Poisson-assumed | seeds |"
            )
            out.append("|---|---|---|---|---|---|")
            for r in cd.itertuples():
                out.append(
                    f"| {int(r.num_sampled)} | {r.central_dp_noise:g} | {cell(r)} "
                    f"| {eps(r.eps_central_noamp)} | {eps(r.eps_central_poisson)} | {int(r.n_seeds)} |"
                )
            out.append("")
        dp = e3[e3.group == "dpsgd"].sort_values(["fraction", "dpsgd_noise"])
        if not dp.empty:
            out.append(
                f"### E3c: sample-level DP (DP-SGD in each client), delta=1e-5, {int(dp.rounds.max())} rounds\n"
            )
            out.append(
                "| clients/round | noise sigma | accuracy (%) | max client epsilon | mean client epsilon | seeds |"
            )
            out.append("|---|---|---|---|---|---|")
            for r in dp.itertuples():
                out.append(
                    f"| {int(r.num_sampled)} | {r.dpsgd_noise:g} | {cell(r)} "
                    f"| {eps(r.eps_sample_max)} | {eps(r.eps_sample_mean)} | {int(r.n_seeds)} |"
                )
            out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    results = ROOT / "results" / ("quick" if args.quick else "")
    df = load_runs(results)
    if df.empty:
        raise SystemExit(f"no runs under {results}")
    df.to_csv(results / "runs.csv", index=False)
    s = summarise(df)
    for exp, part in s.groupby("exp"):
        part.to_csv(results / f"summary_{exp}.csv", index=False)
    (results / "tables.md").write_text(tables(s) + "\n", encoding="utf-8")
    comp = (
        df.groupby("exp")["wall_seconds"]
        .agg(["count", "sum"])
        .rename(columns={"count": "runs", "sum": "seconds"})
    )
    comp["hours"] = comp["seconds"] / 3600
    comp.to_csv(results / "compute.csv")
    print(comp)
    print(tables(s))


if __name__ == "__main__":
    main()
