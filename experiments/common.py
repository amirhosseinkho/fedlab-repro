"""Shared driver for the experiment scripts.

A config YAML has

    base:   settings shared by every run
    groups: list of {name, set: {...}, grid: {key: [values]}, seeds: [...]}
    quick:  {set: {...}, seeds: [...]} applied on top with --quick

Every (group, grid point, seed) becomes one run, written to
``<out-dir>/<group>__<point>__s<seed>.json``. Existing files are skipped, so an
interrupted experiment resumes where it stopped.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from fedlab.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def _fmt(v: Any) -> str:
    return str(v).replace(".", "p").replace("-", "m")


def expand(spec: dict[str, Any], quick: bool) -> list[tuple[str, dict[str, Any], list[int]]]:
    base = dict(spec.get("base", {}))
    q = spec.get("quick", {}) if quick else {}
    runs = []
    for group in spec["groups"]:
        if quick and group.get("skip_in_quick"):
            continue
        grid = group.get("grid", {})
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
            point = dict(zip(keys, values, strict=True))
            cfg = {**base, **group.get("set", {}), **point, **q.get("set", {})}
            name = group["name"] + (
                "__" + "_".join(f"{k}{_fmt(v)}" for k, v in point.items()) if point else ""
            )
            seeds = q.get("seeds", group.get("seeds", spec.get("seeds", [0])))
            runs.append((name, cfg, list(seeds)))
    return runs


def parse_args(name: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=f"Experiment {name.upper()}")
    ap.add_argument("--config", default=str(ROOT / "configs" / f"{name}.yaml"))
    ap.add_argument("--seed", default=None, help="comma-separated seeds; overrides the config")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quick", action="store_true", help="MNIST, few clients and rounds, CPU")
    ap.add_argument(
        "--time-rounds",
        type=int,
        default=0,
        help="only time N rounds per config and project the total",
    )
    ap.add_argument("--only", default=None, help="run only names containing this substring")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--front",
        action="store_true",
        help="run seeds 1 and 2 for the Pareto-front points of the seed-0 runs (E3)",
    )
    return ap.parse_args()


def main(name: str, front: Callable[[Path], set[str]] | None = None) -> None:
    """``front(out_dir)`` returns the run names on the Pareto fronts (E3 only)."""
    args = parse_args(name)
    spec = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    runs = expand(spec, args.quick)
    if args.only:
        runs = [r for r in runs if args.only in r[0]]
    if args.seed is not None:
        seeds = [int(s) for s in args.seed.split(",")]
        runs = [(n, c, seeds) for n, c, _ in runs]
    default_out = ROOT / "results" / ("quick" if args.quick else "") / name
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.front:
        if front is None:
            raise SystemExit(f"--front is not defined for {name}")
        names = front(out_dir)
        print("Pareto-front points:", ", ".join(sorted(names)), flush=True)
        runs = [(n, c, [1, 2]) for n, c, _ in runs if n in names]
    if args.device:
        runs = [(n, {**c, "device": args.device}, s) for n, c, s in runs]
    if args.time_rounds:
        time_runs(runs, args.time_rounds, out_dir)
        return
    total = sum(len(s) for _, _, s in runs)
    done = 0
    t0 = time.time()
    for run_name, cfg, seeds in runs:
        for seed in seeds:
            done += 1
            out = out_dir / f"{run_name}__s{seed}.json"
            if out.exists():
                continue
            res = run_experiment({**cfg, "seed": seed})
            write_compact(out, res)
            print(
                f"[{done}/{total}] {run_name} s{seed}: acc {res['final_acc']:.4f} "
                f"(last3 {res['final_acc_last3']:.4f}) {res['wall_seconds']:.0f}s "
                f"| elapsed {(time.time() - t0) / 60:.1f} min",
                flush=True,
            )


def time_runs(runs: list, n_rounds: int, out_dir: Path) -> None:
    """Run every distinct config for ``n_rounds`` rounds and project the total."""
    rows, total = [], 0.0
    for run_name, cfg, seeds in runs:
        res = run_experiment({**cfg, "rounds": n_rounds, "eval_every": 10**9, "seed": seeds[0]})
        per_round = (res["wall_seconds"] - res["eval_seconds"]) / n_rounds
        per_eval = res["eval_seconds"] / max(res["evals"], 1)
        n_evals = cfg["rounds"] // cfg.get("eval_every", 5) + 2
        projected = len(seeds) * (
            cfg["rounds"] * per_round + n_evals * per_eval + res["data_seconds"]
        )
        total += projected
        rows.append(
            {
                "run": run_name,
                "seeds": len(seeds),
                "rounds": cfg["rounds"],
                "sec_per_round": per_round,
                "sec_per_eval": per_eval,
                "data_seconds": res["data_seconds"],
                "projected_seconds": projected,
            }
        )
        print(
            f"{run_name}: {per_round:.2f} s/round, eval {per_eval:.2f} s -> {projected / 60:.1f} min",
            flush=True,
        )
    print(f"projected total: {total / 3600:.2f} h", flush=True)
    (out_dir / "timing.json").write_text(
        json.dumps({"rows": rows, "total_hours": total / 3600}, indent=1)
    )


def write_compact(path: Path, res: dict[str, Any]) -> None:
    """Write a run log without the bulky per-round byte table.

    Bytes per round are constant for a given configuration, so the first round's
    row and the totals are kept.
    """
    res = dict(res)
    per_round = res.pop("comm_per_round")
    strip = [{k: v for k, v in r.items() if k != "round"} for r in per_round]
    if all(r == strip[0] for r in strip):
        res["comm_round1"] = per_round[0] if per_round else None
        res["comm_rounds"] = len(per_round)
    else:  # not constant: keep the full table
        res["comm_per_round"] = per_round
    path.write_text(json.dumps(res, default=float), encoding="utf-8")


if __name__ == "__main__":
    sys.exit("import this module from run_e*.py")
