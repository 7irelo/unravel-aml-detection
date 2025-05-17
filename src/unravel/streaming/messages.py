"""Wire formats. JSON cannot carry NaN, so missing feature values travel as ``null``.

``tx.raw`` is validated against a closed schema that has no label field, and unexpected fields are
rejected rather than ignored. This makes it structurally impossible for a label to reach the engine
through the stream.
"""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict

from unravel.features.engine import TxEvent

TOPIC_RAW = "tx.raw"
TOPIC_FEATURES = "tx.features"
TOPIC_SCORES = "tx.scores"
TOPIC_ALERTS = "alerts"
TOPIC_DLQ = "dlq"
ALL_TOPICS = (TOPIC_RAW, TOPIC_FEATURES, TOPIC_SCORES, TOPIC_ALERTS, TOPIC_DLQ)


class RawTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tx_id: str
    order_key: int
    minute: int
    src: str
    dst: str
    from_bank: str
    to_bank: str
    amount_received: float
    receiving_currency: str
    amount_paid: float
    payment_currency: str
    payment_format: str

    def to_event(self) -> TxEvent:
        return TxEvent(**self.model_dump())


def encode_event(event: TxEvent) -> bytes:
    return json.dumps(event._asdict(), separators=(",", ":")).encode()


def decode_event(payload: bytes) -> TxEvent:
    return RawTransaction.model_validate_json(payload).to_event()


def encode_vector(tx_id: str, minute: int, vector: list[float]) -> bytes:
    body: dict[str, Any] = {
        "tx_id": tx_id,
        "minute": minute,
        "v": [None if math.isnan(x) else x for x in vector],
    }
    # ``repr`` round-trips floats exactly, and json.dumps uses it, so parity survives the wire.
    return json.dumps(body, separators=(",", ":"), allow_nan=False).encode()


def decode_vector(payload: bytes) -> tuple[str, int, list[float]]:
    body = json.loads(payload)
    return body["tx_id"], body["minute"], [math.nan if x is None else x for x in body["v"]]
