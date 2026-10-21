"""Exact communication accounting from the Flower messages that are exchanged.

Every ``Array`` inside a message's ``ArrayRecord`` s is counted:

* ``raw`` bytes: ``prod(shape) * itemsize`` from the Array's own dtype/shape
  fields, i.e. the tensor payload;
* ``serialized`` bytes: Flower's ``ArrayRecord.count_bytes()``, the length of
  the ``np.save`` buffers Flower actually ships plus the key strings.

Config and metric records (a few scalars) are not counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from flwr.app import RecordDict


def recorddict_bytes(content: RecordDict) -> tuple[int, int]:
    """Return ``(raw, serialized)`` byte counts of all arrays in ``content``."""
    raw = 0
    serialized = 0
    for record in content.array_records.values():
        for arr in record.values():
            raw += int(np.prod(arr.shape, dtype=np.int64)) * np.dtype(arr.dtype).itemsize
        serialized += record.count_bytes()
    return raw, serialized


@dataclass
class CommLog:
    """Per-round uplink (client -> server) and downlink (server -> client) bytes."""

    down_raw: dict[int, int] = field(default_factory=dict)
    up_raw: dict[int, int] = field(default_factory=dict)
    down_serialized: dict[int, int] = field(default_factory=dict)
    up_serialized: dict[int, int] = field(default_factory=dict)
    messages: dict[int, int] = field(default_factory=dict)

    def add(self, server_round: int, content: RecordDict, uplink: bool) -> None:
        raw, ser = recorddict_bytes(content)
        if uplink:
            self.up_raw[server_round] = self.up_raw.get(server_round, 0) + raw
            self.up_serialized[server_round] = self.up_serialized.get(server_round, 0) + ser
        else:
            self.down_raw[server_round] = self.down_raw.get(server_round, 0) + raw
            self.down_serialized[server_round] = self.down_serialized.get(server_round, 0) + ser
            self.messages[server_round] = self.messages.get(server_round, 0) + 1

    def total(self, kind: str = "raw") -> int:
        if kind == "raw":
            return sum(self.down_raw.values()) + sum(self.up_raw.values())
        return sum(self.down_serialized.values()) + sum(self.up_serialized.values())

    def per_round(self) -> list[dict[str, int]]:
        rounds = sorted(set(self.down_raw) | set(self.up_raw))
        return [
            {
                "round": r,
                "down_raw": self.down_raw.get(r, 0),
                "up_raw": self.up_raw.get(r, 0),
                "down_serialized": self.down_serialized.get(r, 0),
                "up_serialized": self.up_serialized.get(r, 0),
                "messages": self.messages.get(r, 0),
            }
            for r in rounds
        ]
