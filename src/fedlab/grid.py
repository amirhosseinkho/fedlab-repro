"""An in-process, sequential implementation of Flower's ``Grid``.

Flower's simulation engine needs Ray, and flwr 1.41.0 does not install Ray on
Windows with Python 3.13 (its metadata excludes ``sys_platform == 'win32'``).
This Grid replaces only the transport: Flower's own ``ServerApp``, strategies,
``Message`` / ``RecordDict`` types and ``ClientApp`` are used unchanged. Each
message is deep-copied on the way to and from a client (so a client never
shares memory with the server), every node keeps a persistent ``Context``
(``context.state`` is how SCAFFOLD clients keep their control variate), and the
clients run one after another in the order the strategy addressed them.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

from flwr.app import Context, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.serverapp import Grid
from flwr.supercore.run import Run
from flwr.supercore.task_identity import TaskIdentity

from fedlab.comm import CommLog

# Flower reserves node ID 1 for the server; clients start after it.
FIRST_CLIENT_NODE_ID = SUPERLINK_NODE_ID + 1


class SequentialGrid(Grid):
    def __init__(
        self,
        client_app: ClientApp,
        num_nodes: int,
        comm: CommLog | None = None,
        run_config: dict | None = None,
    ) -> None:
        self.client_app = client_app
        self.comm = comm if comm is not None else CommLog()
        self._node_ids = list(range(FIRST_CLIENT_NODE_ID, FIRST_CLIENT_NODE_ID + num_nodes))
        self.contexts = {
            nid: Context(
                run_id=1,
                node_id=nid,
                node_config={
                    "partition-id": nid - FIRST_CLIENT_NODE_ID,
                    "num-partitions": num_nodes,
                },
                state=RecordDict(),
                run_config=dict(run_config or {}),
            )
            for nid in self._node_ids
        }
        self._run = Run.create_empty(1)
        # Normally set by Flower's runtime for the ServerApp process; Message()
        # reads these when the strategy builds its messages.
        TaskIdentity.run_id = 1
        TaskIdentity.node_id = SUPERLINK_NODE_ID
        TaskIdentity.task_id = 1
        self._counter = 0
        self._replies: dict[str, Message] = {}

    # --- Grid API ---------------------------------------------------------
    def set_run(self, run: Run) -> None:
        self._run = run

    @property
    def run(self) -> Run:
        return self._run

    def create_message(
        self,
        content: RecordDict,
        message_type: str,
        dst_node_id: int,
        group_id: str,
        ttl: float | None = None,
    ) -> Message:
        return Message(content, dst_node_id, message_type, ttl=ttl, group_id=group_id)

    def get_node_ids(self) -> Iterable[int]:
        return list(self._node_ids)

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        ids = []
        for msg in messages:
            reply = self._deliver(msg)
            self._replies[msg.metadata.message_id] = reply
            ids.append(msg.metadata.message_id)
        return ids

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        return [self._replies.pop(i) for i in message_ids if i in self._replies]

    def send_and_receive(
        self, messages: Iterable[Message], *, timeout: float | None = None
    ) -> Iterable[Message]:
        return self.pull_messages(list(self.push_messages(messages)))

    # --- transport ----------------------------------------------------------
    def _deliver(self, msg: Message) -> Message:
        self._counter += 1
        msg.metadata.__dict__["_message_id"] = f"msg-{self._counter}"
        server_round = int(_config_value(msg.content, "server-round", 0))
        incoming = copy.deepcopy(msg)
        self.comm.add(server_round, incoming.content, uplink=False)
        reply = self.client_app(incoming, self.contexts[msg.metadata.dst_node_id])
        reply.metadata.__dict__["_message_id"] = f"reply-{self._counter}"
        reply = copy.deepcopy(reply)
        if not reply.has_error():
            self.comm.add(server_round, reply.content, uplink=True)
        return reply


def _config_value(content: RecordDict, key: str, default: int) -> int:
    for record in content.config_records.values():
        if key in record:
            return int(record[key])
    return default
