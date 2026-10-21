"""Sample-level (Opacus) and user-level (Flower central DP) privacy."""

from __future__ import annotations

import dp_accounting
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import CPU, Tiny, synthetic_data
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant

from fedlab.privacy import DELTA, central_epsilon, sample_level_epsilon
from fedlab.runner import run_on_data


def test_opacus_clips_each_per_sample_gradient() -> None:
    """With batch size 1 and zero noise, the privatised gradient Opacus applies
    is exactly that sample's clipped gradient, so its norm must be <= C."""
    torch.manual_seed(0)
    C = 0.05
    x = torch.randn(16, 4) * 10
    y = torch.randint(0, 2, (16,))
    model = Tiny()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dp_model, dp_opt, dp_loader = PrivacyEngine(accountant="rdp").make_private(
        module=model,
        optimizer=opt,
        data_loader=loader,
        noise_multiplier=0.0,
        max_grad_norm=C,
        poisson_sampling=False,
    )
    norms_before, norms_after = [], []
    for xb, yb in dp_loader:
        dp_opt.zero_grad()
        F.cross_entropy(dp_model(xb), yb).backward()
        norms_before.append(
            torch.sqrt(sum(p.grad_sample.pow(2).sum() for p in dp_model.parameters())).item()
        )
        dp_opt.pre_step()  # clip + (zero) noise + scale, without the update
        norms_after.append(
            torch.sqrt(sum(p.grad.pow(2).sum() for p in dp_model.parameters())).item()
        )
    assert max(norms_before) > C  # the test exercises actual clipping
    assert max(norms_after) <= C * (1 + 1e-5)
    for b, a in zip(norms_before, norms_after, strict=True):
        assert a == pytest.approx(min(b, C), rel=1e-4)


def test_sample_level_epsilon_matches_opacus() -> None:
    acc = RDPAccountant()
    for _ in range(30):
        acc.step(noise_multiplier=1.1, sample_rate=0.1)
    for _ in range(20):
        acc.step(noise_multiplier=0.8, sample_rate=0.05)
    assert sample_level_epsilon(acc.history, DELTA) == acc.get_epsilon(delta=DELTA)


def test_central_epsilon_matches_dp_accounting() -> None:
    z, T, m, N = 1.0, 40, 10, 100
    ref = dp_accounting.rdp.RdpAccountant()
    ref.compose(dp_accounting.GaussianDpEvent(z), T)
    assert central_epsilon(z, T, m, N, "no-amplification") == pytest.approx(
        ref.get_epsilon(DELTA), rel=1e-12
    )
    ref = dp_accounting.rdp.RdpAccountant()
    ref.compose(dp_accounting.PoissonSampledDpEvent(m / N, dp_accounting.GaussianDpEvent(z)), T)
    assert central_epsilon(z, T, m, N, "poisson") == pytest.approx(
        ref.get_epsilon(DELTA), rel=1e-12
    )


def test_central_epsilon_monotone() -> None:
    eps_t = [central_epsilon(1.0, t, 10, 100, "poisson") for t in (10, 50, 100)]
    eps_z = [central_epsilon(z, 50, 10, 100, "poisson") for z in (0.5, 1.0, 1.5)]
    assert eps_t == sorted(eps_t) and eps_z == sorted(eps_z, reverse=True)
    assert central_epsilon(0.0, 10, 10, 100, "poisson") == float("inf")


def test_reported_central_epsilon_equals_flower_accountant(tiny_cfg) -> None:
    data = synthetic_data([6, 6, 6, 6])
    cfg = {
        **tiny_cfg,
        "rounds": 3,
        "fraction": 0.5,
        "central_dp_noise": 1.2,
        "central_dp_clip": 0.5,
    }
    res = run_on_data(cfg, data, CPU, lambda: Tiny())
    assert res["flower_accountant_epsilon"] == pytest.approx(
        res["history"][-1]["eps_central_no-amplification"], rel=1e-12
    )
    assert res["flower_accountant_epsilon"] == pytest.approx(
        central_epsilon(1.2, 3, 2, 4, "no-amplification")
    )


def test_central_dp_clips_updates(tiny_cfg) -> None:
    """With zero noise, the aggregated update is an average of updates whose
    norm is at most C, so its norm is at most C as well."""
    data = synthetic_data([6, 6, 6])
    C = 1e-3
    cfg = {**tiny_cfg, "rounds": 1, "central_dp_noise": 0.0, "central_dp_clip": C, "lr": 1.0}
    res = run_on_data(cfg, data, CPU, lambda: Tiny(), return_arrays=True)
    d = [
        res["final_arrays"][k].numpy() - res["initial_arrays"][k].numpy()
        for k in res["final_arrays"]
    ]
    assert np.sqrt(sum((v.astype(np.float64) ** 2).sum() for v in d)) <= C * (1 + 1e-4)


def test_dpsgd_client_epsilon_grows_with_rounds(tiny_cfg) -> None:
    data = synthetic_data([20, 20])
    cfg = {**tiny_cfg, "rounds": 3, "batch_size": 5, "dpsgd_noise": 1.0, "dpsgd_clip": 1.0}
    res = run_on_data(cfg, data, CPU, lambda: Tiny())
    eps = [h["eps_sample_max"] for h in res["history"] if "eps_sample_max" in h]
    assert len(eps) == 3 and eps == sorted(eps) and eps[0] < eps[-1]
    # every client ran 4 Poisson steps per round at q = 1/4 (20 samples, batch 5)
    acc = RDPAccountant()
    for _ in range(12):
        acc.step(noise_multiplier=1.0, sample_rate=0.25)
    assert res["eps_sample_per_client"][0] == pytest.approx(acc.get_epsilon(delta=DELTA))
