"""Flower ``ClientApp`` that runs local training for every algorithm.

One train handler covers:

* FedAvg: plain minibatch SGD for E local epochs.
* FedProx: SGD on ``loss + mu/2 * ||w - w_t||^2``, with ``mu`` read from the
  ``"proximal-mu"`` key that Flower's ``FedProx`` strategy puts in the config.
  Flower's docstring example adds ``(mu/2) * ||w - w_t||`` (not squared); this
  file uses the squared norm of Li et al. (2020), Eq. (2).
* SCAFFOLD (Karimireddy et al. 2020, Alg. 1, option II): every local step uses
  ``g - c_i + c``; afterwards ``c_i+ = c_i - c + (x - y) / (K * lr)``. ``c_i``
  lives in the node's ``context.state``; the reply carries the new weights and
  ``delta c_i``.
* DP-SGD (sample-level DP): Opacus ``PrivacyEngine`` with Poisson sampling,
  per-sample clipping and Gaussian noise; the RDP accountant history persists
  in ``context.state``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedlab.data import FederatedData
from fedlab.models import build_model
from fedlab.utils import derive_seed

SCAFFOLD_C = "scaffold-c"
SCAFFOLD_DC = "scaffold-dc"
SCAFFOLD_CI = "scaffold-ci"
DPSGD_HISTORY = "dpsgd-history"


@dataclass
class ClientSettings:
    algorithm: str  # fedavg | fedprox | scaffold
    lr: float
    local_epochs: int
    batch_size: int
    seed: int
    dpsgd_noise: float | None = None  # None = no sample-level DP
    dpsgd_clip: float = 1.0
    log: list[dict] = field(default_factory=list)  # per-call client records


def _state_dict(record: ArrayRecord, device: torch.device) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((k, t.to(device)) for k, t in record.to_torch_state_dict().items())


def _record(tensors: dict[str, torch.Tensor]) -> ArrayRecord:
    return ArrayRecord({k: Array(v.detach().cpu().numpy()) for k, v in tensors.items()})


def make_client_app(
    data: FederatedData,
    settings: ClientSettings,
    device: torch.device,
    model_fn: Callable[[], torch.nn.Module] | None = None,
) -> ClientApp:
    app = ClientApp()
    model_fn = model_fn or (lambda: build_model(data.name))
    model = model_fn().to(device)

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        cfg = msg.content["config"]
        server_round = int(cfg["server-round"])
        pid = int(context.node_config["partition-id"])
        gen = torch.Generator().manual_seed(derive_seed(settings.seed, "client", pid, server_round))
        x = _state_dict(msg.content["arrays"], device)
        n = data.client_size(pid)
        extra: dict[str, float] = {}

        if settings.dpsgd_noise is not None:
            y, steps, loss, hist = _train_dpsgd(
                data, pid, x, settings, gen, context, device, server_round, model_fn
            )
            reply = RecordDict({"arrays": _record(y)})
        else:
            mu = float(cfg.get("proximal-mu", 0.0))
            corr = None
            if settings.algorithm == "scaffold":
                c = _state_dict(msg.content[SCAFFOLD_C], device)
                ci = (
                    _state_dict(context.state[SCAFFOLD_CI], device)
                    if SCAFFOLD_CI in context.state
                    else OrderedDict((k, torch.zeros_like(v)) for k, v in c.items())
                )
                corr = {k: c[k] - ci[k] for k in c}
            model.load_state_dict(x)
            steps, loss = local_sgd(model, data, pid, settings, gen, x_global=x, mu=mu, corr=corr)
            y = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())
            reply = RecordDict({"arrays": _record(y)})
            if settings.algorithm == "scaffold":
                assert corr is not None
                ci_new = {k: ci[k] - c[k] + (x[k] - y[k]) / (steps * settings.lr) for k in x}
                context.state[SCAFFOLD_CI] = _record(ci_new)
                reply[SCAFFOLD_DC] = _record({k: ci_new[k] - ci[k] for k in x})

        # L2 norm of the (unclipped) model update, used to choose the central-DP clip.
        extra["update-norm"] = float(
            torch.sqrt(sum(((y[k] - x[k]).double() ** 2).sum() for k in x)).item()
        )
        reply["metrics"] = MetricRecord(
            {"num-examples": n, "dp-weight": 1.0, "train-loss": loss, "steps": steps, **extra}
        )
        rec = {"round": server_round, "client": pid, "n": n, "steps": steps, "loss": loss, **extra}
        if settings.dpsgd_noise is not None:
            # epsilon is computed from this history at evaluation time (runner)
            rec["dp_history"] = tuple(hist)
        settings.log.append(rec)
        return Message(reply, reply_to=msg)

    return app


def local_sgd(
    model: torch.nn.Module,
    data: FederatedData,
    pid: int,
    settings: ClientSettings,
    gen: torch.Generator,
    x_global: dict[str, torch.Tensor] | None = None,
    mu: float = 0.0,
    corr: dict[str, torch.Tensor] | None = None,
) -> tuple[int, float]:
    """Run E epochs of minibatch SGD; return (number of steps, mean loss)."""
    model.train()
    params = dict(model.named_parameters())
    opt = torch.optim.SGD(model.parameters(), lr=settings.lr)
    total = torch.zeros((), device=next(model.parameters()).device)
    seen = 0
    steps = 0
    for xb, yb in data.train_batches(pid, settings.batch_size, settings.local_epochs, gen):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        total += loss.detach() * len(yb)
        seen += len(yb)
        if mu > 0.0:
            assert x_global is not None
            prox = sum(((p - x_global[k]) ** 2).sum() for k, p in params.items())
            loss = loss + 0.5 * mu * prox
        loss.backward()
        if corr is not None:
            for k, p in params.items():
                p.grad.add_(corr[k])
        opt.step()
        steps += 1
    return steps, float(total.item() / max(seen, 1))


def _train_dpsgd(
    data: FederatedData,
    pid: int,
    x: dict[str, torch.Tensor],
    settings: ClientSettings,
    gen: torch.Generator,
    context: Context,
    device: torch.device,
    server_round: int,
    model_fn: Callable[[], torch.nn.Module],
) -> tuple[dict[str, torch.Tensor], int, float, list[tuple[float, float, int]]]:
    """DP-SGD with Opacus's clipping, noise and RDP accountant.

    ``make_private`` wraps the model and optimizer and registers the accountant
    hook with ``sample_rate = 1 / len(loader)``. Batches are drawn here on the
    device with the same rule as Opacus's ``UniformWithReplacementSampler``
    (each sample independently with probability ``sample_rate``,
    ``int(1 / sample_rate)`` batches per epoch), which avoids a per-item
    DataLoader over GPU tensors.
    """
    from opacus import PrivacyEngine

    assert settings.algorithm == "fedavg", "DP-SGD is only combined with FedAvg here"
    model = model_fn().to(device)
    model.load_state_dict(x)
    xs, ys = data.client_tensors(pid)
    n = len(ys)
    # Only its length matters: it sets Opacus's sample rate 1 / len(loader).
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.arange(n)), batch_size=settings.batch_size
    )
    opt = torch.optim.SGD(model.parameters(), lr=settings.lr)
    engine = PrivacyEngine(accountant="rdp")
    if DPSGD_HISTORY in context.state:
        hist = context.state[DPSGD_HISTORY]
        engine.accountant.history = [
            (float(z), float(q), int(s))
            for z, q, s in zip(hist["noise"], hist["rate"], hist["steps"], strict=True)
        ]
    noise_gen = torch.Generator(device=device).manual_seed(
        derive_seed(settings.seed, "dpnoise", pid, server_round)
    )
    dp_model, dp_opt, _ = engine.make_private(
        module=model,
        optimizer=opt,
        data_loader=loader,
        noise_multiplier=float(settings.dpsgd_noise),
        max_grad_norm=settings.dpsgd_clip,
        poisson_sampling=True,
        noise_generator=noise_gen,
    )
    sample_rate = 1 / len(loader)
    batches_per_epoch = int(1 / sample_rate)
    dp_model.train()
    total, seen, steps = 0.0, 0, 0
    for _ in range(settings.local_epochs):
        for _ in range(batches_per_epoch):
            mask = torch.rand(n, generator=gen) < sample_rate
            b = mask.nonzero(as_tuple=False).reshape(-1).to(device)
            xb, yb = xs[b], ys[b]
            dp_opt.zero_grad(set_to_none=True)
            out = dp_model(data.augment(xb, gen))
            if len(yb) == 0:
                # Empty Poisson batch: Opacus still takes a noise-only step.
                loss = out.sum() * 0.0
            else:
                loss = F.cross_entropy(out, yb)
                total += float(loss.detach()) * len(yb)
                seen += len(yb)
            loss.backward()
            dp_opt.step()
            steps += 1
    hist = [(float(z), float(q), int(k)) for z, q, k in engine.accountant.history]
    context.state[DPSGD_HISTORY] = ConfigRecord(
        {
            "noise": [h[0] for h in hist],
            "rate": [h[1] for h in hist],
            "steps": [h[2] for h in hist],
        }
    )
    y = OrderedDict((k, v.detach().clone()) for k, v in dp_model._module.state_dict().items())
    # Free the per-sample gradients and Opacus's hooks now. Otherwise they stay
    # reachable through reference cycles until the garbage collector runs, and
    # GPU memory grows past the card's 4 GB within a few rounds (D22).
    dp_opt.zero_grad(set_to_none=True)
    dp_model._close()
    return y, steps, total / max(seen, 1), hist


def numpy_state(record: ArrayRecord) -> dict[str, np.ndarray]:
    return {k: v.numpy() for k, v in record.items()}
