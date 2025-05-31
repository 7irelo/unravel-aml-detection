"""Multi-PNA with edge updates, for edge classification on a directed multigraph.

Follows Egressy et al. (AAAI 2024): reverse message passing, port numbering and ego IDs on top of
PNA, with edge-update MLPs. Two evaluation protocols share this code:

* **P1, paper-compatible.** Neighbours may come from anywhere in the snapshot, including edges later
  than the target. This matches how IBM's loader samples and supports published comparisons.
* **P2, strictly causal.** Every target gets its own disjoint subgraph containing only strictly
  earlier edges. Separate copies are needed because a node seen by two targets at different times
  must not share one state. This estimates what a live system can do.

Port numbers are computed inside the sampled subgraph, from its time-ordered edges, so under P2 they
can only reflect the past.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from torch import nn
from torch_geometric.nn import PNAConv

from unravel.config import DatasetPaths
from unravel.eval.metrics import pr_auc
from unravel.eval.splits import TEST, TRAIN, VAL_A, VAL_B, Split
from unravel.features.engine import PAYMENT_FORMATS
from unravel.features.fx import load_fx

Protocol = Literal["P1", "P2"]


@dataclass(frozen=True)
class GnnConfig:
    arm: str = "G3"
    protocol: Protocol = "P2"
    hidden: int = 64
    layers: int = 2
    fanout: tuple[int, ...] = (20, 10)
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 20
    negatives_per_positive: int = 10
    positive_weight: float = 6.0
    patience: int = 5
    ports: bool = True
    ego: bool = True
    reverse: bool = True
    edge_updates: bool = True
    extra_columns: tuple[str, ...] = field(default_factory=tuple)
    seed: int = 0


@dataclass
class EdgeTable:
    """All transactions as integer-coded arrays, sorted by canonical time."""

    src: np.ndarray
    dst: np.ndarray
    minute: np.ndarray
    attributes: np.ndarray
    y: np.ndarray
    partition: np.ndarray
    origin: int
    n_nodes: int
    tx_id: np.ndarray

    def __post_init__(self) -> None:
        self.adjacency = _Adjacency(self.src, self.dst, self.n_nodes)


class _Adjacency:
    """Per-node edge lists, sorted by edge index, which is also time order."""

    def __init__(self, src: np.ndarray, dst: np.ndarray, n_nodes: int) -> None:
        nodes = np.concatenate([src, dst])
        edges = np.concatenate([np.arange(len(src)), np.arange(len(src))])
        self_loop_copy = np.concatenate([np.zeros(len(src), bool), src == dst])
        nodes, edges = nodes[~self_loop_copy], edges[~self_loop_copy]
        order = np.lexsort((edges, nodes))
        self.edges = edges[order]
        counts = np.bincount(nodes, minlength=n_nodes)
        self.starts = np.r_[0, np.cumsum(counts)]

    def incident(self, node: int) -> np.ndarray:
        return self.edges[self.starts[node] : self.starts[node + 1]]


def edge_attributes(
    usd: np.ndarray,
    payment_format: np.ndarray,
    minute: np.ndarray,
    cross_currency: np.ndarray,
    origin: int,
) -> np.ndarray:
    """Transaction attributes. Absolute time is excluded so test values aren't out of range."""
    return np.column_stack(
        [
            np.log1p(np.nan_to_num(usd)),
            cross_currency.astype(np.float32),
            *[(payment_format == i).astype(np.float32) for i in range(len(PAYMENT_FORMATS))],
            np.sin(2 * np.pi * (minute % 1440) / 1440),
            np.cos(2 * np.pi * (minute % 1440) / 1440),
        ]
    ).astype(np.float32)


def load_edges(paths: DatasetPaths, extra_columns: tuple[str, ...] = ()) -> EdgeTable:
    split = Split.load(paths.split)
    fx = load_fx(paths.fx)
    tx = pl.read_parquet(paths.transactions)
    labels = pl.read_parquet(paths.labels).select("tx_id", "is_laundering")
    joined = tx.join(labels, on="tx_id", how="left", maintain_order="left")
    accounts = pl.concat([joined["src"], joined["dst"]]).unique(maintain_order=True)
    code = {name: i for i, name in enumerate(accounts.to_list())}
    src = np.fromiter((code[a] for a in joined["src"].to_list()), np.int64, joined.height)
    dst = np.fromiter((code[a] for a in joined["dst"].to_list()), np.int64, joined.height)
    usd = joined.with_columns(
        (pl.col("amount_paid") * pl.col("payment_currency").replace_strict(fx, default=0.0)).alias(
            "usd"
        )
    )["usd"].to_numpy()
    fmt_code = {name: i for i, name in enumerate(PAYMENT_FORMATS)}
    fmt = np.array([fmt_code.get(f, -1) for f in joined["payment_format"].to_list()])
    minute = joined["minute"].to_numpy()
    cross = (joined["payment_currency"] != joined["receiving_currency"]).to_numpy()
    attributes = edge_attributes(usd, fmt, minute, cross, split.origin_minute)
    if extra_columns:
        from unravel.models.dataset import FeatureReader

        extra = FeatureReader(paths.features).matrix(list(extra_columns))
        extra = np.nan_to_num(extra, nan=-1.0, posinf=0.0, neginf=0.0)
        train = minute < split.val_start
        scale = np.maximum(extra[train].std(axis=0), 1e-6)
        attributes = np.c_[attributes, (extra - extra[train].mean(axis=0)) / scale]
    return EdgeTable(
        src=src,
        dst=dst,
        minute=minute,
        attributes=attributes,
        y=joined["is_laundering"].to_numpy().astype(np.int64),
        partition=split.assign(minute),
        origin=split.origin_minute,
        n_nodes=len(code),
        tx_id=joined["tx_id"].to_numpy(),
    )


@dataclass
class Subgraph:
    """A disjoint batch. ``edge_*`` are message edges; targets index into local node IDs."""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    target_src: torch.Tensor
    target_dst: torch.Tensor
    target_attr: torch.Tensor
    y: torch.Tensor
    #: Edge-table index of each message edge, and its target. Used only by leakage tests.
    message_edge_ids: np.ndarray
    message_owner: np.ndarray


def _port_numbers(
    src_local: np.ndarray, dst_local: np.ndarray, order: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Number each edge by neighbour order, per node and direction, using the earliest edge per pair.

    Parallel edges share a port, which is how the method distinguishes repeat payments to one peer
    from payments to several peers.
    """

    def assign(owner: np.ndarray, peer: np.ndarray) -> np.ndarray:
        ports = np.zeros(len(owner), dtype=np.float32)
        if len(owner) == 0:
            return ports
        keys = owner.astype(np.int64) * (n_nodes + 1) + peer
        unique, inverse = np.unique(keys, return_inverse=True)
        earliest = np.full(len(unique), np.iinfo(np.int64).max)
        np.minimum.at(earliest, inverse, order)
        pair_owner = unique // (n_nodes + 1)
        ranking = np.lexsort((earliest, pair_owner))
        rank = np.zeros(len(unique), dtype=np.int64)
        boundaries = np.r_[True, pair_owner[ranking][1:] != pair_owner[ranking][:-1]]
        positions = np.arange(len(unique))
        group_start = np.maximum.accumulate(np.where(boundaries, positions, 0))
        rank[ranking] = positions - group_start
        ports = rank[inverse].astype(np.float32)
        return ports

    return assign(dst_local, src_local), assign(src_local, dst_local)


class NeighbourSampler:
    """Build disjoint per-target subgraphs under the chosen protocol."""

    def __init__(
        self, table: EdgeTable, config: GnnConfig, snapshot_end: int, seed: int = 0
    ) -> None:
        self.table = table
        self.config = config
        self.snapshot_end = snapshot_end  # exclusive edge index visible under P1
        self.rng = np.random.default_rng(seed)

    def _candidates(self, node: int, target: int) -> np.ndarray:
        edges = self.table.adjacency.incident(node)
        if self.config.protocol == "P2":
            # Strictly earlier minutes only: same-minute edges have no reliable order.
            cutoff = self.table.minute[target]
            limit = np.searchsorted(self.table.minute[edges], cutoff, side="left")
            return edges[:limit]
        limit = np.searchsorted(edges, self.snapshot_end, side="left")
        visible = edges[:limit]
        return visible[visible != target]

    def _sample(self, edges: np.ndarray, count: int) -> np.ndarray:
        if len(edges) <= count:
            return edges
        if self.config.protocol == "P2":
            return edges[-count:]  # the most recent past is the most relevant
        return self.rng.choice(edges, count, replace=False)

    def batch(self, targets: np.ndarray) -> Subgraph:
        table = self.table
        message_edges: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        for owner, target in enumerate(targets.tolist()):
            frontier = {int(table.src[target]), int(table.dst[target])}
            seen: set[int] = set(frontier)
            collected: list[np.ndarray] = []
            for fanout in self.config.fanout:
                next_frontier: set[int] = set()
                for node in frontier:
                    chosen = self._sample(self._candidates(node, target), fanout)
                    if len(chosen):
                        collected.append(chosen)
                        for peer in np.r_[table.src[chosen], table.dst[chosen]].tolist():
                            if peer not in seen:
                                seen.add(peer)
                                next_frontier.add(peer)
                frontier = next_frontier
                if not frontier:
                    break
            if collected:
                unique = np.unique(np.concatenate(collected))
                message_edges.append(unique)
                owners.append(np.full(len(unique), owner, dtype=np.int64))
        edge_ids = np.concatenate(message_edges) if message_edges else np.zeros(0, dtype=np.int64)
        owner_ids = np.concatenate(owners) if owners else np.zeros(0, dtype=np.int64)

        # Nodes are local to their target: the same account under two targets is two nodes.
        batch_size = len(targets)
        shift = table.n_nodes
        keys = np.concatenate(
            [
                owner_ids * shift + table.src[edge_ids],
                owner_ids * shift + table.dst[edge_ids],
                np.arange(batch_size) * shift + table.src[targets],
                np.arange(batch_size) * shift + table.dst[targets],
            ]
        )
        unique_nodes, inverse = np.unique(keys, return_inverse=True)
        m = len(edge_ids)
        src_local = inverse[:m]
        dst_local = inverse[m : 2 * m]
        target_src = inverse[2 * m : 2 * m + batch_size]
        target_dst = inverse[2 * m + batch_size :]

        n_nodes = len(unique_nodes)
        ego = np.zeros((n_nodes, 1), dtype=np.float32)
        if self.config.ego:
            ego[target_src] = 1.0
            ego[target_dst] = 1.0

        attributes = table.attributes[edge_ids]
        if self.config.ports:
            in_port, out_port = _port_numbers(src_local, dst_local, edge_ids, n_nodes)
            ports = np.log1p(np.c_[in_port, out_port])
            attributes = np.c_[attributes, ports]
        target_attr = table.attributes[targets]
        return Subgraph(
            x=torch.from_numpy(ego),
            edge_index=torch.from_numpy(np.vstack([src_local, dst_local]).astype(np.int64)),
            edge_attr=torch.from_numpy(attributes.astype(np.float32)),
            target_src=torch.from_numpy(target_src.astype(np.int64)),
            target_dst=torch.from_numpy(target_dst.astype(np.int64)),
            target_attr=torch.from_numpy(target_attr),
            y=torch.from_numpy(table.y[targets].astype(np.float32)),
            message_edge_ids=edge_ids,
            message_owner=owner_ids,
        )


def _mlp(inputs: int, hidden: int, outputs: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, outputs)
    )


class MultiPna(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        edge_inputs: int,
        target_inputs: int,
        degree_histogram: torch.Tensor,
        config: GnnConfig,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden
        self.node_encoder = nn.Linear(1, hidden)
        self.edge_encoder = nn.Linear(edge_inputs, hidden)
        options = {
            "aggregators": ["mean", "min", "max", "std"],
            "scalers": ["identity", "amplification"],
            "deg": degree_histogram,
            "edge_dim": hidden,
            "towers": 1,
            "pre_layers": 1,
            "post_layers": 1,
            "divide_input": False,
        }
        self.forward_convs = nn.ModuleList(
            PNAConv(hidden, hidden, **options) for _ in range(config.layers)
        )
        self.reverse_convs = (
            nn.ModuleList(PNAConv(hidden, hidden, **options) for _ in range(config.layers))
            if config.reverse
            else None
        )
        merge_in = hidden * (2 if config.reverse else 1)
        self.merges = nn.ModuleList(_mlp(merge_in, hidden, hidden) for _ in range(config.layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(config.layers))
        self.edge_updates = (
            nn.ModuleList(_mlp(hidden * 3, hidden, hidden) for _ in range(config.layers))
            if config.edge_updates
            else None
        )
        self.target_encoder = nn.Linear(target_inputs, hidden)
        self.classifier = _mlp(hidden * 3, hidden, 1, config.dropout)

    def forward(self, batch: Subgraph) -> torch.Tensor:
        h = self.node_encoder(batch.x)
        edges = self.edge_encoder(batch.edge_attr)
        source, destination = batch.edge_index[0], batch.edge_index[1]
        reverse_index = batch.edge_index.flip(0)
        for k in range(self.config.layers):
            parts = [self.forward_convs[k](h, batch.edge_index, edge_attr=edges)]
            if self.reverse_convs is not None:
                parts.append(self.reverse_convs[k](h, reverse_index, edge_attr=edges))
            h = h + self.norms[k](self.merges[k](torch.cat(parts, dim=-1))).relu()
            if self.edge_updates is not None and edges.numel():
                edges = edges + self.edge_updates[k](
                    torch.cat([h[source], h[destination], edges], dim=-1)
                )
        target = self.target_encoder(batch.target_attr)
        readout = torch.cat([h[batch.target_src], h[batch.target_dst], target], dim=-1)
        return self.classifier(readout).squeeze(-1)


def degree_histogram(table: EdgeTable, mask: np.ndarray, cap: int = 2000) -> torch.Tensor:
    degree = np.bincount(table.dst[mask], minlength=table.n_nodes)
    return torch.from_numpy(np.bincount(np.minimum(degree, cap)).astype(np.int64))


def _sequential(data: np.ndarray, size: int) -> list[np.ndarray]:
    return [data[i : i + size] for i in range(0, len(data), size)]


def predict(
    model: MultiPna, sampler: NeighbourSampler, targets: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    """Return probabilities. Training-mode behaviour such as dropout is switched off."""
    model.eval()
    scores = np.zeros(len(targets), dtype=np.float64)
    cursor = 0
    with torch.no_grad():
        for chunk in _sequential(targets, batch_size):
            scores[cursor : cursor + len(chunk)] = (
                torch.sigmoid(model(sampler.batch(chunk))).cpu().numpy()
            )
            cursor += len(chunk)
    return scores


@dataclass
class TrainResult:
    model: MultiPna
    best_epoch: int
    best_val_pr_auc: float
    history: list[dict[str, float]]
    seconds: float


def train_gnn(
    table: EdgeTable,
    config: GnnConfig,
    *,
    val_negatives: int = 20_000,
    log: bool = True,
) -> TrainResult:
    """Fit on train-window edges. Model selection uses validation A only."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()

    train_mask = table.partition == TRAIN
    train_end = int(np.flatnonzero(train_mask).max()) + 1
    val_mask = table.partition == VAL_A
    val_end = int(np.flatnonzero(val_mask).max()) + 1
    train_sampler = NeighbourSampler(table, config, train_end, config.seed)
    val_sampler = NeighbourSampler(table, config, val_end, config.seed + 1)

    positives = np.flatnonzero(train_mask & (table.y == 1))
    negatives = np.flatnonzero(train_mask & (table.y == 0))
    val_pos = np.flatnonzero(val_mask & (table.y == 1))
    val_neg = np.flatnonzero(val_mask & (table.y == 0))
    val_neg = rng.choice(val_neg, min(val_negatives, len(val_neg)), replace=False)
    val_targets = np.sort(np.r_[val_pos, val_neg])

    probe = train_sampler.batch(positives[:4] if len(positives) >= 4 else negatives[:4])
    model = MultiPna(
        probe.edge_attr.shape[1],
        probe.target_attr.shape[1],
        degree_histogram(table, train_mask),
        config,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.positive_weight))

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        n_negative = min(len(negatives), config.negatives_per_positive * len(positives))
        epoch_targets = np.r_[positives, rng.choice(negatives, n_negative, replace=False)]
        rng.shuffle(epoch_targets)
        total_loss = 0.0
        for chunk in _sequential(epoch_targets, config.batch_size):
            batch = train_sampler.batch(chunk)
            optimiser.zero_grad()
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            total_loss += float(loss.detach()) * len(chunk)
        val_scores = predict(model, val_sampler, val_targets)
        val_score = pr_auc(table.y[val_targets], val_scores)
        row = {
            "epoch": float(epoch),
            "loss": total_loss / max(len(epoch_targets), 1),
            "val_a_subsample_pr_auc": val_score,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        if log:
            print(json.dumps(row), flush=True)
        if val_score > best_score:
            best_score, best_epoch = val_score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model, best_epoch, best_score, history, time.perf_counter() - started)


def score_partitions(
    table: EdgeTable, result: TrainResult, config: GnnConfig, codes: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Score whole partitions with the visibility each protocol allows."""
    output = {}
    ends = {
        VAL_A: int(np.flatnonzero(np.isin(table.partition, (TRAIN, VAL_A))).max()) + 1,
        VAL_B: int(np.flatnonzero(np.isin(table.partition, (TRAIN, VAL_A, VAL_B))).max()) + 1,
        TEST: len(table.src),
    }
    for code in codes:
        targets = np.flatnonzero(table.partition == code)
        sampler = NeighbourSampler(table, config, ends[code], config.seed + 2)
        output[code] = predict(result.model, sampler, targets)
    return output


def save_model(result: TrainResult, config: GnnConfig, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), directory / "model.pt")
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "best_epoch": result.best_epoch,
                "best_val_pr_auc": result.best_val_pr_auc,
                "history": result.history,
                "seconds": result.seconds,
                "device": "cpu",
            },
            indent=2,
        )
    )
