"""Privacy accounting for the two DP levels used in this repo.

(a) Sample-level DP (DP-SGD inside each client, Opacus). Each client keeps
    the history of its Opacus ``RDPAccountant`` (noise multiplier, sample rate,
    steps) in its Flower ``context.state`` across the rounds it is sampled in;
    its epsilon at ``delta`` is recomputed from that history by Opacus.

(b) User-level central DP (DP-FedAvg, McMahan et al. 2018) with Flower's
    ``DifferentialPrivacyServerSideFixedClipping``. Two epsilons are reported,
    both computed by Flower's ``RdpAccountant`` (which wraps Google's
    dp-accounting):

    * ``no_amplification``: every round is a Gaussian mechanism on the full
      population (no sampling amplification). This is the accountant Flower
      1.41 allows inside the strategy and it is valid for our simulation.
    * ``poisson``: subsampled Gaussian with q = m / N, the accounting of the
      DP-FedAvg paper. It assumes Poisson client sampling; our simulation draws
      exactly m clients without replacement (``random.sample`` in Flower's
      ``sample_nodes``), so this number is reported for comparison only.

No epsilon is computed by hand anywhere in this repo.
"""

from __future__ import annotations

import functools

import numpy as np
from flwr.supercore.privacy_accounting import (
    GaussianPrivacyEvent,
    NeighboringRelation,
    PrivacyConfig,
    SamplingMethod,
)
from flwr.supercore.privacy_accounting.rdp_accountant import RdpAccountant
from opacus.accountants import RDPAccountant as OpacusRDP
from opacus.accountants.analysis import rdp as opacus_rdp

DELTA = 1e-5


def sample_level_epsilon(history: list[tuple[float, float, int]], delta: float = DELTA) -> float:
    """Epsilon of one client's DP-SGD history, as Opacus's RDP accountant computes it.

    Same arithmetic as ``opacus.accountants.RDPAccountant.get_privacy_spent``
    (sum of ``compute_rdp`` over the history, then ``get_privacy_spent``), but
    the RDP of one step is cached per (sample rate, noise multiplier): Opacus
    recomputes it from scratch, which dominated the cost of DP-SGD runs.
    """
    if not history:
        return 0.0
    rdp = sum(_rdp_one_step(float(q), float(z)) * int(k) for z, q, k in history)
    eps, _ = opacus_rdp.get_privacy_spent(orders=OpacusRDP.DEFAULT_ALPHAS, rdp=rdp, delta=delta)
    return float(eps)


@functools.cache
def _rdp_one_step(q: float, noise_multiplier: float) -> np.ndarray:
    return opacus_rdp.compute_rdp(
        q=q, noise_multiplier=noise_multiplier, steps=1, orders=OpacusRDP.DEFAULT_ALPHAS
    )


def flower_accountant(
    population: int, sampling: str = "no-amplification", delta: float = DELTA
) -> RdpAccountant:
    method = {
        "no-amplification": SamplingMethod.NO_AMPLIFICATION,
        "poisson": SamplingMethod.POISSON,
    }[sampling]
    return RdpAccountant(
        PrivacyConfig(
            target_delta=delta,
            population_size=population,
            neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
            sampling_method=method,
        )
    )


def central_epsilon(
    noise_multiplier: float,
    rounds: int,
    sample_size: int,
    population: int,
    sampling: str,
    delta: float = DELTA,
) -> float:
    """User-level epsilon after ``rounds`` releases of DP-FedAvg."""
    if noise_multiplier <= 0:
        return float("inf")
    acc = flower_accountant(population, sampling, delta)
    acc.compose(GaussianPrivacyEvent(noise_multiplier, sample_size, population), count=rounds)
    return acc.get_privacy_spent().epsilon
