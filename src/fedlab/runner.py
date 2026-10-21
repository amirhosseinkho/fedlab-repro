"""Run one federated experiment from a config dict and write its logs."""

from __future__ import annotations

import logging
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import flwr
import numpy as np
import opacus
import torch
import torch.nn.functional as F
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import (
    DifferentialPrivacyServerSideFixedClipping,
    FedAvg,
    FedProx,
    Strategy,
)

from fedlab.clients import ClientSettings, make_client_app
from fedlab.comm import CommLog
from fedlab.data import FederatedData, build_federated_data
from fedlab.grid import SequentialGrid
from fedlab.models import build_model, num_params
from fedlab.privacy import DELTA, central_epsilon, flower_accountant
from fedlab.strategies import Scaffold
from fedlab.utils import get_device, set_global_seed, write_json

DEFAULTS: dict[str, Any] = {
    "dataset": "cifar10",
    "partition": "dirichlet",
    "alpha": 0.5,
    "num_clients": 100,
    "fraction": 0.1,
    "rounds": 100,
    "local_epochs": 1,
    "batch_size": 50,
    "lr": 0.05,
    "algorithm": "fedavg",
    "mu": 0.0,
    "server_lr": 1.0,
    "central_dp_noise": None,  # None = no central DP; 0.0 = clipping only
    "central_dp_clip": 1.0,
    "dpsgd_noise": None,  # None = no sample-level DP
    "dpsgd_clip": 1.0,
    "eval_every": 5,
    "seed": 0,
    "device": "auto",
    "max_train": None,
    "max_test": None,
    "femnist_subset_seed": 0,
}


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: FederatedData) -> tuple[float, float]:
    model.eval()
    correct, loss, n = 0, 0.0, 0
    for xb, yb in data.test_batches():
        out = model(xb)
        loss += float(F.cross_entropy(out, yb, reduction="sum"))
        correct += int((out.argmax(1) == yb).sum())
        n += len(yb)
    return correct / n, loss / n


def build_strategy(cfg: dict[str, Any], num_sampled: int) -> Strategy:
    central = cfg["central_dp_noise"] is not None
    kw: dict[str, Any] = {
        "fraction_train": cfg["fraction"],
        "fraction_evaluate": 0.0,  # evaluation is centralised on the test set
        "min_train_nodes": 1,
        "min_available_nodes": cfg["num_clients"],
        # DP-FedAvg averages clipped updates uniformly (equal client weights).
        "weighted_by_key": "dp-weight" if central else "num-examples",
    }
    algo = cfg["algorithm"]
    strategy: Strategy
    if algo == "fedavg":
        strategy = FedAvg(**kw)
    elif algo == "fedprox":
        strategy = FedProx(proximal_mu=cfg["mu"], **kw)
    elif algo == "scaffold":
        strategy = Scaffold(server_lr=cfg["server_lr"], **kw)
    else:
        raise ValueError(algo)
    if central:
        if algo != "fedavg":
            raise ValueError("central DP is only combined with FedAvg (DP-FedAvg)")
        z = float(cfg["central_dp_noise"])
        strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy,
            noise_multiplier=z,
            clipping_norm=float(cfg["central_dp_clip"]),
            num_sampled_clients=num_sampled,
            accountant=flower_accountant(cfg["num_clients"]) if z > 0 else None,
        )
    return strategy


def run_experiment(cfg_in: dict[str, Any], out_path: str | Path | None = None) -> dict[str, Any]:
    cfg = {**DEFAULTS, **cfg_in}
    logging.getLogger("flwr").setLevel(logging.WARNING)
    set_global_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    t0 = time.time()
    data = build_federated_data(
        cfg["dataset"],
        cfg["partition"],
        cfg["num_clients"],
        seed=int(cfg["seed"]),
        device=device,
        alpha=cfg["alpha"],
        femnist_subset_seed=cfg["femnist_subset_seed"],
        max_train=cfg["max_train"],
        max_test=cfg["max_test"],
    )
    result = run_on_data(cfg, data, device)
    result["data_seconds"] = time.time() - t0 - result["wall_seconds"]
    if out_path is not None:
        write_json(out_path, result)
    return result


def run_on_data(
    cfg_in: dict[str, Any],
    data: FederatedData,
    device: torch.device,
    model_fn: Callable[[], torch.nn.Module] | None = None,
    return_arrays: bool = False,
) -> dict[str, Any]:
    """Run the federation on already-partitioned ``data``.

    ``model_fn`` overrides the dataset's model (unit tests use a tiny one).
    """
    cfg = {**DEFAULTS, **cfg_in, "num_clients": data.num_clients}
    logging.getLogger("flwr").setLevel(logging.WARNING)
    # Seed after data preparation so model init and sampling do not depend on
    # how much randomness partitioning consumed.
    set_global_seed(int(cfg["seed"]))
    model_fn = model_fn or (lambda: build_model(data.name))
    t0 = time.time()
    n_clients = data.num_clients
    num_sampled = max(int(n_clients * cfg["fraction"]), 1)
    settings = ClientSettings(
        algorithm=cfg["algorithm"],
        lr=float(cfg["lr"]),
        local_epochs=int(cfg["local_epochs"]),
        batch_size=int(cfg["batch_size"]),
        seed=int(cfg["seed"]),
        dpsgd_noise=cfg["dpsgd_noise"],
        dpsgd_clip=float(cfg["dpsgd_clip"]),
    )
    comm = CommLog()
    grid = SequentialGrid(make_client_app(data, settings, device, model_fn), n_clients, comm)
    strategy = build_strategy(cfg, num_sampled)
    model = model_fn().to(device)
    eval_model = model_fn().to(device)
    initial = ArrayRecord({k: Array(v.cpu().numpy()) for k, v in model.state_dict().items()})
    history: list[dict[str, Any]] = []
    rounds = int(cfg["rounds"])
    eval_time = [0.0]

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
        if server_round % int(cfg["eval_every"]) and server_round not in (0, rounds):
            return None
        eval_model.load_state_dict(
            {k: v.to(device) for k, v in arrays.to_torch_state_dict().items()}
        )
        acc, loss = evaluate(eval_model, data)
        row: dict[str, Any] = {
            "round": server_round,
            "acc": acc,
            "loss": loss,
            "bytes_up": sum(v for r, v in comm.up_raw.items() if r <= server_round),
            "bytes_down": sum(v for r, v in comm.down_raw.items() if r <= server_round),
            "time": time.time() - t0,
        }
        if cfg["central_dp_noise"] is not None and server_round > 0:
            z = float(cfg["central_dp_noise"])
            for s in ("no-amplification", "poisson"):
                row[f"eps_central_{s}"] = central_epsilon(
                    z, server_round, num_sampled, n_clients, s
                )
        if cfg["dpsgd_noise"] is not None and settings.log:
            last: dict[int, float] = {}
            for rec in settings.log:
                last[rec["client"]] = rec["epsilon"]
            row["eps_sample_max"] = max(last.values())
            row["eps_sample_mean_participants"] = float(np.mean(list(last.values())))
        history.append(row)
        return MetricRecord({"accuracy": acc, "loss": loss})

    app = ServerApp()

    @app.main()
    def main(grid_: Any, context: Context) -> None:
        res = strategy.start(
            grid=grid_,
            initial_arrays=initial,
            num_rounds=rounds,
            train_config=ConfigRecord({}),
            evaluate_fn=evaluate_fn,
        )
        final["arrays"] = res.arrays

    final: dict[str, ArrayRecord] = {}

    app(grid, Context(run_id=1, node_id=0, node_config={}, state=RecordDict(), run_config={}))
    wall = time.time() - t0

    result: dict[str, Any] = {
        "config": cfg,
        "history": history,
        "final_acc": history[-1]["acc"],
        "final_acc_last3": float(np.mean([h["acc"] for h in history[-3:]])),
        "total_bytes_raw": comm.total("raw"),
        "total_bytes_serialized": comm.total("serialized"),
        "comm_per_round": comm.per_round(),
        "num_params": num_params(model),
        "num_sampled": num_sampled,
        "client_sizes": [data.client_size(i) for i in range(n_clients)],
        "wall_seconds": wall,
        "eval_seconds": eval_time[0],
        "evals": len(history),
        "device": str(device)
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "flwr": flwr.__version__,
            "opacus": opacus.__version__,
        },
    }
    if isinstance(strategy, DifferentialPrivacyServerSideFixedClipping):
        spent = strategy.privacy_spent()
        result["flower_accountant_epsilon"] = None if spent is None else spent.epsilon
        result["delta"] = DELTA
    if cfg["dpsgd_noise"] is not None:
        last = {}
        for rec in settings.log:
            last[rec["client"]] = rec["epsilon"]
        result["eps_sample_per_client"] = last
        result["delta"] = DELTA
    if return_arrays:
        result["initial_arrays"] = initial
        result["final_arrays"] = final["arrays"]
        result["grid"] = grid
        result["strategy"] = strategy
    return result
