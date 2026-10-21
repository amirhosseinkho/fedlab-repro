"""FedAvg, FedProx and SCAFFOLD against their defining equations."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import CPU, Tiny, synthetic_data

from fedlab.models import CNNMnist, num_params
from fedlab.runner import run_on_data


def _arrays(rec) -> dict[str, np.ndarray]:
    return {k: v.numpy() for k, v in rec.items()}


def test_mnist_cnn_has_paper_parameter_count() -> None:
    # McMahan et al. (2017), Sec. 3: "1,663,370 total parameters".
    assert num_params(CNNMnist()) == 1_663_370


def test_fedavg_one_client_one_step_equals_centralised_sgd(tiny_cfg) -> None:
    data = synthetic_data([8])  # one client, one full batch -> exactly one SGD step
    cfg = {**tiny_cfg, "rounds": 1, "batch_size": 8}
    res = run_on_data(cfg, data, CPU, model_fn=lambda: Tiny(), return_arrays=True)

    model = Tiny()
    model.load_state_dict(
        {k: torch.from_numpy(v) for k, v in _arrays(res["initial_arrays"]).items()}
    )
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    x, y = data.client_tensors(0)
    loss = F.cross_entropy(model(data.augment(x, torch.Generator())), y)
    loss.backward()
    opt.step()
    got = _arrays(res["final_arrays"])
    for k, v in model.state_dict().items():
        np.testing.assert_allclose(got[k], v.numpy(), rtol=0, atol=1e-6)


def test_fedprox_with_mu_zero_equals_fedavg(tiny_cfg) -> None:
    data = synthetic_data([10, 7, 13])
    cfg = {**tiny_cfg, "rounds": 3, "batch_size": 4}
    a = run_on_data({**cfg, "algorithm": "fedavg"}, data, CPU, lambda: Tiny(), return_arrays=True)
    b = run_on_data(
        {**cfg, "algorithm": "fedprox", "mu": 0.0}, data, CPU, lambda: Tiny(), return_arrays=True
    )
    for k, v in _arrays(a["final_arrays"]).items():
        np.testing.assert_array_equal(v, _arrays(b["final_arrays"])[k])
    c = run_on_data(
        {**cfg, "algorithm": "fedprox", "mu": 1.0}, data, CPU, lambda: Tiny(), return_arrays=True
    )
    assert any(
        not np.array_equal(v, _arrays(c["final_arrays"])[k])
        for k, v in _arrays(a["final_arrays"]).items()
    )


def test_fedprox_step_has_proximal_gradient() -> None:
    from fedlab.clients import ClientSettings, local_sgd

    data = synthetic_data([6])
    model = Tiny()
    start = copy.deepcopy(model.state_dict())
    anchor = {k: v + 0.3 for k, v in start.items()}  # w_t differs from the start point
    mu, lr = 0.5, 0.1
    s = ClientSettings("fedprox", lr=lr, local_epochs=1, batch_size=6, seed=0)
    local_sgd(model, data, 0, s, torch.Generator(), x_global=anchor, mu=mu)

    ref = Tiny()
    ref.load_state_dict(start)
    x, y = data.client_tensors(0)
    F.cross_entropy(ref(data.augment(x, torch.Generator())), y).backward()
    for k, p in ref.named_parameters():
        expected = start[k] - lr * (p.grad + mu * (start[k] - anchor[k]))
        np.testing.assert_allclose(model.state_dict()[k].numpy(), expected.numpy(), atol=1e-6)


def _softmax_grad(w: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Analytic gradient of mean cross-entropy for a linear softmax model."""
    logits = x @ w.T + b
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    p[np.arange(len(y)), y] -= 1.0
    return p.T @ x / len(y), p.mean(0)


def test_scaffold_control_variates_follow_paper(tiny_cfg) -> None:
    """Two rounds, two clients, one local step each (K = 1), compared with
    Algorithm 1 (option II) evaluated by hand with analytic gradients."""
    data = synthetic_data([5, 3], seed=3)
    lr, rounds = 0.5, 2
    cfg = {**tiny_cfg, "algorithm": "scaffold", "rounds": rounds, "batch_size": 8, "lr": lr}
    res = run_on_data(cfg, data, CPU, model_fn=lambda: Tiny(zero=True), return_arrays=True)

    feats = [
        data.augment(data.client_tensors(i)[0], torch.Generator())
        .flatten(1)
        .numpy()
        .astype(np.float64)
        for i in range(2)
    ]
    labels = [data.client_tensors(i)[1].numpy() for i in range(2)]
    w, b = np.zeros((2, 4)), np.zeros(2)
    c_w, c_b = np.zeros((2, 4)), np.zeros(2)
    ci = [(np.zeros((2, 4)), np.zeros(2)) for _ in range(2)]
    N, K = 2, 1
    for _ in range(rounds):
        ys, dcs = [], []
        for i in range(2):
            gw, gb = _softmax_grad(w, b, feats[i], labels[i])
            yw = w - lr * (gw - ci[i][0] + c_w)  # y <- y - lr (g - c_i + c)
            yb = b - lr * (gb - ci[i][1] + c_b)
            new_ci = (ci[i][0] - c_w + (w - yw) / (K * lr), ci[i][1] - c_b + (b - yb) / (K * lr))
            dcs.append((new_ci[0] - ci[i][0], new_ci[1] - ci[i][1]))
            ci[i] = new_ci
            ys.append((yw, yb))
        w = w + np.mean([yw - w for yw, _ in ys], axis=0)  # eta_g = 1
        b = b + np.mean([yb - b for _, yb in ys], axis=0)
        c_w = c_w + sum(d[0] for d in dcs) / N
        c_b = c_b + sum(d[1] for d in dcs) / N
    # With K = 1, option II gives c_i = gradient at the last server model.
    got = _arrays(res["final_arrays"])
    np.testing.assert_allclose(got["fc.weight"], w, atol=1e-5)
    np.testing.assert_allclose(got["fc.bias"], b, atol=1e-5)
    server_c = _arrays(res["strategy"].c)
    np.testing.assert_allclose(server_c["fc.weight"], c_w, atol=1e-5)
    np.testing.assert_allclose(server_c["fc.bias"], c_b, atol=1e-5)
    for i in range(2):
        ctx = res["grid"].contexts[res["grid"].get_node_ids()[i]]
        stored = _arrays(ctx.state["scaffold-ci"])
        np.testing.assert_allclose(stored["fc.weight"], ci[i][0], atol=1e-5)


@pytest.mark.parametrize("algorithm, factor", [("fedavg", 1), ("scaffold", 2)])
def test_round_bytes_match_model_size(tiny_cfg, algorithm, factor) -> None:
    data = synthetic_data([4, 4, 4, 4])
    cfg = {**tiny_cfg, "algorithm": algorithm, "fraction": 0.5, "rounds": 3}
    res = run_on_data(cfg, data, CPU, model_fn=lambda: Tiny(), return_arrays=True)
    model_bytes = sum(p.numel() * p.element_size() for p in Tiny().parameters())
    for row in res["comm_per_round"]:
        assert row["messages"] == 2
        assert row["down_raw"] == 2 * factor * model_bytes
        assert row["up_raw"] == 2 * factor * model_bytes
    assert res["total_bytes_raw"] == 3 * 2 * 2 * factor * model_bytes
