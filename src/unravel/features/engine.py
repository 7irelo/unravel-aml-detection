"""The single feature implementation, shared by offline backfill and the live consumer.

Parity between training and live scoring comes from there being only one code path. The rules that
keep it honest:

* Event time only. Nothing reads a wall clock.
* A transaction's features are computed *before* it updates state, so a vector depends only on
  strictly earlier events in the canonical order ``(minute, order_key)``.
* All state is plain data, so it can be snapshotted and restored exactly.
* No labels, transaction IDs or account identifiers enter the feature vector.
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Iterable
from typing import NamedTuple

WINDOWS: tuple[int, ...] = (60, 360, 1440, 5760)  # 1h, 6h, 24h, 96h, in minutes
W_1H, W_6H, W_24H, W_96H = range(4)
WINDOW_LABELS = ("1h", "6h", "24h", "96h")

PAYMENT_FORMATS = ("ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire")
_FORMAT_CODE = {name: float(i) for i, name in enumerate(PAYMENT_FORMATS)}

#: Bounds on graph work per event. A hit sets ``graph_truncated`` rather than failing silently.
MAX_INTERSECT = 256
MAX_FAN_SCAN = 32
MAX_BFS_NODES = 128
MAX_CYCLE_HOPS = 6

NAN = float("nan")


class TxEvent(NamedTuple):
    """One transaction as seen on ``tx.raw``. There is deliberately no label field."""

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


class _Flow:
    """Exact sliding-window aggregates over one account's flows in a single direction."""

    __slots__ = ("events", "heads", "last_advance", "nbrs", "rounds", "sum_c_log_c", "sums")

    def __init__(self) -> None:
        self.events: list[tuple[int, float, int, bool]] = []
        self.heads = [0, 0, 0, 0]
        self.sums = [0.0, 0.0, 0.0, 0.0]
        self.rounds = 0  # round-amount events inside the 24h window
        self.nbrs: tuple[dict[int, int], ...] = ({}, {}, {}, {})
        self.sum_c_log_c = 0.0  # supports O(1) entropy over the 24h window
        self.last_advance = -1

    def advance(self, now: int) -> None:
        if now == self.last_advance:
            return
        self.last_advance = now
        events = self.events
        size = len(events)
        for k in range(4):
            head = self.heads[k]
            if head >= size:
                continue
            cutoff = now - WINDOWS[k]
            if events[head][0] > cutoff:
                continue
            nbrs = self.nbrs[k]
            total = self.sums[k]
            while head < size and events[head][0] <= cutoff:
                _, usd, peer, is_round = events[head]
                total -= usd
                count = nbrs[peer]
                if k == W_24H:
                    self.sum_c_log_c -= count * math.log(count)
                    if count > 1:
                        self.sum_c_log_c += (count - 1) * math.log(count - 1)
                    if is_round:
                        self.rounds -= 1
                if count == 1:
                    del nbrs[peer]
                else:
                    nbrs[peer] = count - 1
                head += 1
            self.heads[k] = head
            self.sums[k] = total
        oldest = self.heads[W_96H]
        if oldest > 512 and oldest * 2 > size:
            del events[:oldest]
            self.heads = [h - oldest for h in self.heads]

    def add(self, minute: int, usd: float, peer: int, is_round: bool) -> None:
        self.events.append((minute, usd, peer, is_round))
        for k in range(4):
            nbrs = self.nbrs[k]
            count = nbrs.get(peer, 0)
            if k == W_24H:
                if count:
                    self.sum_c_log_c -= count * math.log(count)
                self.sum_c_log_c += (count + 1) * math.log(count + 1)
                if is_round:
                    self.rounds += 1
            nbrs[peer] = count + 1
            self.sums[k] += usd

    def count(self, k: int) -> int:
        return len(self.events) - self.heads[k]

    def entropy_24h(self) -> float:
        n = self.count(W_24H)
        if n == 0:
            return 0.0
        return max(0.0, math.log(n) - self.sum_c_log_c / n)


class _Account:
    __slots__ = (
        "first_minute",
        "flow_in",
        "flow_out",
        "last_currency",
        "last_format",
        "last_in_minute",
        "last_in_usd",
        "last_out_minute",
        "mean_in",
        "mean_out",
        "n_in",
        "n_out",
        "pairs_out",
        "var_in",
        "var_out",
    )

    def __init__(self, minute: int) -> None:
        self.flow_out = _Flow()
        self.flow_in = _Flow()
        self.first_minute = minute
        self.n_out = 0
        self.n_in = 0
        self.mean_out = 0.0
        self.var_out = 0.0
        self.mean_in = 0.0
        self.var_in = 0.0
        self.last_out_minute = -1
        self.last_in_minute = -1
        self.last_in_usd = 0.0
        self.last_format = -1.0
        self.last_currency = ""
        self.pairs_out: dict[int, int] = {}


def _flow_names(prefix: str) -> list[str]:
    names: list[str] = []
    for label in WINDOW_LABELS:
        names += [f"{prefix}_cnt_{label}", f"{prefix}_sum_{label}", f"{prefix}_distinct_{label}"]
    return names


TRANSACTION_FEATURES = [
    "log_usd",
    "log_paid_over_received",
    "payment_format",
    "cross_currency",
    "cross_bank",
    "self_transfer",
    "minute_of_day",
    "amount_round_100",
    "amount_round_1000",
]
VELOCITY_FEATURES = (
    _flow_names("src_out") + _flow_names("src_in") + _flow_names("dst_in") + _flow_names("dst_out")
)
HISTORY_FEATURES = [
    "src_out_entropy_24h",
    "dst_in_entropy_24h",
    "src_out_round_ratio_24h",
    "dst_in_round_ratio_24h",
    "pair_count_prior",
    "pair_is_new",
    "reverse_pair_count_prior",
    "src_mins_since_out",
    "src_mins_since_in",
    "dst_mins_since_in",
    "dst_mins_since_out",
    "src_age_mins",
    "dst_age_mins",
    "src_lifetime_out",
    "dst_lifetime_in",
    "src_amount_z",
    "dst_amount_z",
    "src_format_switch",
    "src_currency_switch",
    "src_forward_ratio_6h",
    "src_forward_ratio_24h",
    "src_forward_ratio_96h",
    "dst_forward_ratio_24h",
    "amount_over_last_in",
]
GRAPH_FEATURES = [
    "two_paths_6h",
    "two_paths_24h",
    "two_paths_96h",
    "reciprocal_96h",
    "cycle3_paths_96h",
    "cycle_min_hops_96h",
    "shared_senders_96h",
    "dst_downstream_reach_96h",
    "src_upstream_reach_96h",
    "graph_truncated",
]

FEATURE_NAMES: list[str] = (
    TRANSACTION_FEATURES + VELOCITY_FEATURES + HISTORY_FEATURES + GRAPH_FEATURES
)
CATEGORICAL_FEATURES = ["payment_format"]

#: Feature sets for the tree model arms in the protocol.
FEATURE_SETS: dict[str, list[str]] = {
    "T1": TRANSACTION_FEATURES,
    "T2": TRANSACTION_FEATURES + VELOCITY_FEATURES + HISTORY_FEATURES,
    "T3": FEATURE_NAMES,
}


def _intersection_size(a: dict[int, int], b: dict[int, int]) -> tuple[int, bool]:
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    truncated = len(small) > MAX_INTERSECT
    hits = 0
    for scanned, key in enumerate(small):
        if scanned >= MAX_INTERSECT:
            break
        if key in large:
            hits += 1
    return hits, truncated


class FeatureEngine:
    """Consume events in canonical order and emit one feature vector per event."""

    def __init__(self, fx_rates: dict[str, float]) -> None:
        self.fx_rates = dict(fx_rates)
        self.ids: dict[str, int] = {}
        self.accounts: list[_Account] = []
        self.events_seen = 0
        self.truncated_events = 0
        self.last_minute = -1

    # -- state helpers ----------------------------------------------------------------------------

    def _account(self, name: str, minute: int) -> tuple[int, _Account]:
        index = self.ids.get(name)
        if index is None:
            index = len(self.accounts)
            self.ids[name] = index
            self.accounts.append(_Account(minute))
        return index, self.accounts[index]

    def _usd(self, amount: float, currency: str) -> float:
        rate = self.fx_rates.get(currency)
        return amount * rate if rate is not None else NAN

    # -- graph features ---------------------------------------------------------------------------

    def _cycle_min_hops(self, start: int, target: int, now: int) -> tuple[float, bool]:
        """Shortest path ``dst -> ... -> src`` over 96h edges, which this transaction would close."""
        frontier = [start]
        seen = {start}
        expanded = 0
        for hops in range(1, MAX_CYCLE_HOPS):
            next_frontier: list[int] = []
            for node in frontier:
                if expanded >= MAX_BFS_NODES:
                    return NAN, True
                expanded += 1
                flow = self.accounts[node].flow_out
                flow.advance(now)
                nbrs = flow.nbrs[W_96H]
                if target in nbrs:
                    return float(hops + 1), False
                for peer in nbrs:
                    if peer not in seen:
                        seen.add(peer)
                        next_frontier.append(peer)
            if not next_frontier:
                return NAN, False
            frontier = next_frontier
        return NAN, False

    def _graph_features(
        self, src_id: int, dst_id: int, src: _Account, dst: _Account, now: int
    ) -> list[float]:
        truncated = False
        paths: list[float] = []
        for k in (W_6H, W_24H, W_96H):
            hits, cut = _intersection_size(src.flow_out.nbrs[k], dst.flow_in.nbrs[k])
            truncated |= cut
            paths.append(float(hits))

        reciprocal = float(src.flow_in.nbrs[W_96H].get(dst_id, 0))
        cycle3, cut = _intersection_size(dst.flow_out.nbrs[W_96H], src.flow_in.nbrs[W_96H])
        truncated |= cut

        if src_id == dst_id:
            cycle_hops, cut = NAN, False
        else:
            cycle_hops, cut = self._cycle_min_hops(dst_id, src_id, now)
        truncated |= cut

        # Other senders into dst that also pay something src pays: the bipartite signature.
        shared = 0.0
        src_targets = src.flow_out.nbrs[W_96H]
        if len(src_targets) >= 1 and len(dst.flow_in.nbrs[W_96H]) >= 2:
            for scanned, peer in enumerate(dst.flow_in.nbrs[W_96H]):
                if scanned >= MAX_FAN_SCAN:
                    truncated = True
                    break
                if peer == src_id:
                    continue
                other = self.accounts[peer].flow_out
                other.advance(now)
                hits, _ = _intersection_size(other.nbrs[W_96H], src_targets)
                if hits:
                    shared += 1.0

        # How far money can travel onward from dst, and where it may have come from before src.
        downstream = 0.0
        for scanned, peer in enumerate(dst.flow_out.nbrs[W_96H]):
            if scanned >= MAX_FAN_SCAN:
                truncated = True
                break
            flow = self.accounts[peer].flow_out
            flow.advance(now)
            downstream += len(flow.nbrs[W_96H])
        upstream = 0.0
        for scanned, peer in enumerate(src.flow_in.nbrs[W_96H]):
            if scanned >= MAX_FAN_SCAN:
                truncated = True
                break
            flow = self.accounts[peer].flow_in
            flow.advance(now)
            upstream += len(flow.nbrs[W_96H])

        if truncated:
            self.truncated_events += 1
        return [
            *paths,
            reciprocal,
            float(cycle3),
            cycle_hops,
            shared,
            downstream,
            upstream,
            1.0 if truncated else 0.0,
        ]

    # -- the one public operation -----------------------------------------------------------------

    def process(self, event: TxEvent) -> list[float]:
        """Return the feature vector for ``event``, then fold the event into state."""
        now = event.minute
        if now < self.last_minute:
            raise ValueError(
                f"out-of-order event {event.tx_id}: minute {now} after {self.last_minute}"
            )
        self.last_minute = now

        src_id, src = self._account(event.src, now)
        dst_id, dst = self._account(event.dst, now)
        for flow in (src.flow_out, src.flow_in, dst.flow_in, dst.flow_out):
            flow.advance(now)

        paid_usd = self._usd(event.amount_paid, event.payment_currency)
        received_usd = self._usd(event.amount_received, event.receiving_currency)
        usd = paid_usd if paid_usd == paid_usd else received_usd
        safe_usd = usd if usd == usd and usd > 0 else 0.0
        log_usd = math.log1p(safe_usd)
        log_ratio = (
            math.log(paid_usd / received_usd)
            if paid_usd == paid_usd
            and received_usd == received_usd
            and paid_usd > 0
            and received_usd > 0
            else NAN
        )
        format_code = _FORMAT_CODE.get(event.payment_format, NAN)
        cents = round(event.amount_paid * 100)
        round_100 = cents % 10_000 == 0 and cents > 0
        round_1000 = cents % 100_000 == 0 and cents > 0

        vector: list[float] = [
            log_usd,
            log_ratio,
            format_code,
            1.0 if event.payment_currency != event.receiving_currency else 0.0,
            1.0 if event.from_bank != event.to_bank else 0.0,
            1.0 if src_id == dst_id else 0.0,
            float(now % 1440),
            1.0 if round_100 else 0.0,
            1.0 if round_1000 else 0.0,
        ]

        for flow in (src.flow_out, src.flow_in, dst.flow_in, dst.flow_out):
            events_len = len(flow.events)
            heads = flow.heads
            sums = flow.sums
            nbrs = flow.nbrs
            for k in range(4):
                total = sums[k]
                vector += [
                    float(events_len - heads[k]),
                    math.log1p(total) if total > 0.0 else 0.0,
                    float(len(nbrs[k])),
                ]

        src_out_24 = src.flow_out.count(W_24H)
        dst_in_24 = dst.flow_in.count(W_24H)
        pair_count = src.pairs_out.get(dst_id, 0)
        reverse_count = dst.pairs_out.get(src_id, 0)

        def forward(account: _Account, k: int) -> float:
            incoming = account.flow_in.sums[k]
            return account.flow_out.sums[k] / incoming if incoming > 1e-9 else NAN

        vector += [
            src.flow_out.entropy_24h(),
            dst.flow_in.entropy_24h(),
            src.flow_out.rounds / src_out_24 if src_out_24 else 0.0,
            dst.flow_in.rounds / dst_in_24 if dst_in_24 else 0.0,
            float(pair_count),
            0.0 if pair_count else 1.0,
            float(reverse_count),
            float(now - src.last_out_minute) if src.last_out_minute >= 0 else NAN,
            float(now - src.last_in_minute) if src.last_in_minute >= 0 else NAN,
            float(now - dst.last_in_minute) if dst.last_in_minute >= 0 else NAN,
            float(now - dst.last_out_minute) if dst.last_out_minute >= 0 else NAN,
            float(now - src.first_minute),
            float(now - dst.first_minute),
            float(src.n_out),
            float(dst.n_in),
            (log_usd - src.mean_out) / math.sqrt(src.var_out / src.n_out)
            if src.n_out >= 2 and src.var_out > 1e-12
            else NAN,
            (log_usd - dst.mean_in) / math.sqrt(dst.var_in / dst.n_in)
            if dst.n_in >= 2 and dst.var_in > 1e-12
            else NAN,
            1.0 if src.last_format >= 0 and format_code != src.last_format else 0.0,
            1.0 if src.last_currency and event.payment_currency != src.last_currency else 0.0,
            forward(src, W_6H),
            forward(src, W_24H),
            forward(src, W_96H),
            forward(dst, W_24H),
            safe_usd / src.last_in_usd if src.last_in_usd > 1e-9 else NAN,
        ]
        vector += self._graph_features(src_id, dst_id, src, dst, now)

        # Only now does this transaction become part of history.
        src.flow_out.add(now, safe_usd, dst_id, round_100)
        dst.flow_in.add(now, safe_usd, src_id, round_100)
        src.pairs_out[dst_id] = pair_count + 1

        src.n_out += 1
        delta = log_usd - src.mean_out
        src.mean_out += delta / src.n_out
        src.var_out += delta * (log_usd - src.mean_out)
        dst.n_in += 1
        delta = log_usd - dst.mean_in
        dst.mean_in += delta / dst.n_in
        dst.var_in += delta * (log_usd - dst.mean_in)

        src.last_out_minute = now
        src.last_format = format_code if format_code == format_code else -1.0
        src.last_currency = event.payment_currency
        dst.last_in_minute = now
        dst.last_in_usd = safe_usd
        self.events_seen += 1
        return vector

    def process_all(self, events: Iterable[TxEvent]) -> list[list[float]]:
        return [self.process(event) for event in events]

    # -- snapshots --------------------------------------------------------------------------------

    def snapshot(self) -> bytes:
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore(blob: bytes) -> FeatureEngine:
        # Snapshots are only ever read from this system's own state directory.
        engine = pickle.loads(blob)
        if not isinstance(engine, FeatureEngine):
            raise TypeError("snapshot does not contain a FeatureEngine")
        return engine


def vectors_equal(a: list[float], b: list[float], tol: float = 1e-9) -> bool:
    """Compare feature vectors, treating two NaNs as equal."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if x != x or y != y:
            if not (x != x and y != y):
                return False
        elif abs(x - y) > tol * max(1.0, abs(x), abs(y)):
            return False
    return True
