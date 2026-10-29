"""Experiment E3; see configs/e3.yaml.

Seed 0 runs every grid point. ``--front`` then adds seeds 1 and 2 for the
points on the Pareto fronts of the seed-0 runs:

* no DP: final accuracy vs total bytes (up + down);
* central DP (z > 0): final accuracy vs epsilon (Poisson-assumed accounting);
* DP-SGD: final accuracy vs the largest client epsilon.
"""

from __future__ import annotations

import json
from pathlib import Path

from common import main


def pareto(points: list[tuple[float, float, str]]) -> set[str]:
    """Names of the points not dominated by another (smaller x, larger y)."""
    front, best = set(), float("-inf")
    for _x, y, name in sorted(points, key=lambda p: (p[0], -p[1])):
        if y > best:
            front.add(name)
            best = y
    return front


def front(out_dir: Path) -> set[str]:
    groups: dict[str, list[tuple[float, float, str]]] = {"nodp": [], "central": [], "dpsgd": []}
    for f in sorted(out_dir.glob("*__s0.json")):
        r = json.loads(f.read_text())
        name = f.name.removesuffix("__s0.json")
        group = name.split("__")[0]
        last = r["history"][-1]
        acc = r["final_acc_last3"]
        if group == "nodp":
            groups[group].append((r["total_bytes_raw"], acc, name))
        elif group == "central" and r["config"]["central_dp_noise"] > 0:
            groups[group].append((last["eps_central_poisson"], acc, name))
        elif group == "dpsgd":
            groups[group].append((last["eps_sample_max"], acc, name))
    return set().union(*(pareto(p) for p in groups.values()))


if __name__ == "__main__":
    main("e3", front)
