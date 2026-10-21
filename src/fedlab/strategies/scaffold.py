"""SCAFFOLD (Karimireddy et al., ICML 2020) as a Flower message-based strategy.

Flower 1.41.0 ships no SCAFFOLD strategy (``flwr/serverapp/strategy`` has
FedAvg, FedProx, FedAdam, ... but no control-variate method), so it is written
here on top of Flower's ``FedAvg`` (sampling and message construction are
inherited unchanged).

Server side of Algorithm 1, with global step size ``server_lr`` (eta_g):

    x <- x + eta_g * (1/|S|) * sum_{i in S} (y_i - x)
    c <- c + (1/N)   * sum_{i in S} delta c_i

Downlink carries ``x`` and ``c``; uplink carries ``y_i`` and ``delta c_i``, so
each direction moves twice the model size (counted by ``fedlab.comm``).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from fedlab.clients import SCAFFOLD_C, SCAFFOLD_DC


class Scaffold(FedAvg):
    def __init__(self, *, server_lr: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.server_lr = server_lr
        self.c: ArrayRecord | None = None
        self._x: ArrayRecord | None = None
        self._num_nodes = 0

    def summary(self) -> None:
        super().summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if self.c is None:
            self.c = ArrayRecord({k: Array(np.zeros_like(v.numpy())) for k, v in arrays.items()})
        self._x = arrays
        self._num_nodes = len(list(grid.get_node_ids()))
        messages = list(super().configure_train(server_round, arrays, config, grid))
        for msg in messages:
            msg.content[SCAFFOLD_C] = self.c
        return messages

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid, _ = self._check_and_log_replies(replies, is_train=True, validate=False)
        if not valid:
            return None, None
        assert self._x is not None and self.c is not None
        x = {k: v.numpy() for k, v in self._x.items()}
        c = {k: v.numpy() for k, v in self.c.items()}
        dy = {k: np.zeros_like(v) for k, v in x.items()}
        dc = {k: np.zeros_like(v) for k, v in x.items()}
        for msg in valid:
            y_i = msg.content["arrays"]
            dc_i = msg.content[SCAFFOLD_DC]
            for k in x:
                dy[k] += y_i[k].numpy() - x[k]
                dc[k] += dc_i[k].numpy()
        m = len(valid)
        new_x = {k: x[k] + self.server_lr * dy[k] / m for k in x}
        new_c = {k: c[k] + dc[k] / self._num_nodes for k in c}
        self.c = ArrayRecord({k: Array(v.astype(c[k].dtype)) for k, v in new_c.items()})
        metrics = self.train_metrics_aggr_fn([msg.content for msg in valid], self.weighted_by_key)
        return ArrayRecord({k: Array(v.astype(x[k].dtype)) for k, v in new_x.items()}), metrics
