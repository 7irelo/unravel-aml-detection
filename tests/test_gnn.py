"""Leakage guarantees of the sampler, and the expressivity claims behind the architecture."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from unravel.eval.metrics import pr_auc
from unravel.eval.splits import TEST, TRAIN, VAL_A
from unravel.models.gnn import (
    EdgeTable,
    GnnConfig,
    MultiPna,
    NeighbourSampler,
    _port_numbers,
    degree_histogram,
    predict,
    score_partitions,
    train_gnn,
)


def table_from(edges: list[tuple[int, int, int]], y: list[int] | None = None) -> EdgeTable:
    src = np.array([e[0] for e in edges])
    dst = np.array([e[1] for e in edges])
    minute = np.array([e[2] for e in edges])
    return EdgeTable(
        src=src,
        dst=dst,
        minute=minute,
        attributes=np.ones((len(edges), 3), dtype=np.float32),
        y=np.array(y) if y is not None else np.zeros(len(edges), dtype=np.int64),
        partition=np.zeros(len(edges), dtype=np.int64),
        origin=0,
        n_nodes=int(max(src.max(), dst.max())) + 1,
        tx_id=np.array([f"t{i}" for i in range(len(edges))]),
    )


def _random_table(seed: int = 0, n: int = 3000, nodes: int = 200) -> EdgeTable:
    rng = np.random.default_rng(seed)
    minutes = np.sort(rng.integers(0, 5000, n))
    return table_from(
        [(int(rng.integers(nodes)), int(rng.integers(nodes)), int(m)) for m in minutes]
    )


# -- causality ------------------------------------------------------------------------------------


def test_p2_never_uses_an_edge_at_or_after_the_target_minute() -> None:
    table = _random_table()
    sampler = NeighbourSampler(table, GnnConfig(protocol="P2"), len(table.src))
    targets = np.random.default_rng(1).choice(len(table.src), 200, replace=False)
    batch = sampler.batch(targets)
    assert len(batch.message_edge_ids) > 0
    target_minutes = table.minute[targets][batch.message_owner]
    assert np.all(table.minute[batch.message_edge_ids] < target_minutes)


def test_p2_excludes_same_minute_edges() -> None:
    table = table_from([(0, 2, 1), (0, 1, 5), (1, 2, 5), (2, 3, 5)])
    batch = NeighbourSampler(table, GnnConfig(protocol="P2"), 4).batch(np.array([2]))
    assert set(batch.message_edge_ids.tolist()) == {0}


def test_p1_can_see_future_edges_within_the_snapshot_but_not_past_it() -> None:
    """P1 is deliberately non-causal inside the snapshot, as in the published protocol."""
    table = table_from([(0, 1, 1), (1, 2, 2), (2, 3, 3), (1, 4, 4)])
    sampler = NeighbourSampler(table, GnnConfig(protocol="P1"), 3)
    batch = sampler.batch(np.array([0]))
    assert 1 in batch.message_edge_ids  # a later edge, inside the snapshot
    assert 3 not in batch.message_edge_ids  # outside the snapshot
    assert 0 not in batch.message_edge_ids  # never the target itself


def test_mutating_the_future_cannot_change_a_p2_prediction() -> None:
    table = _random_table(n=800, nodes=40)
    config = GnnConfig(protocol="P2", hidden=8, fanout=(5, 3), dropout=0.0)
    histogram = degree_histogram(table, np.ones(len(table.src), dtype=bool))
    sampler = NeighbourSampler(table, config, len(table.src))
    targets = np.arange(400, 420)
    torch.manual_seed(0)
    probe = sampler.batch(targets)
    model = MultiPna(probe.edge_attr.shape[1], probe.target_attr.shape[1], histogram, config)
    before = predict(model, sampler, targets)

    table.attributes[500:] = 99.0  # everything later than all the targets
    table.y[500:] = 1
    after = predict(model, NeighbourSampler(table, config, len(table.src)), targets)
    assert np.array_equal(before, after)


def test_labels_are_never_message_inputs() -> None:
    table = _random_table(n=500, nodes=30)
    config = GnnConfig(protocol="P2", hidden=8, fanout=(5, 3))
    sampler = NeighbourSampler(table, config, len(table.src))
    targets = np.arange(300, 340)
    a = sampler.batch(targets)
    table.y[:] = 1
    b = NeighbourSampler(table, config, len(table.src)).batch(targets)
    assert torch.equal(a.edge_attr, b.edge_attr)
    assert torch.equal(a.x, b.x)


def test_each_target_gets_its_own_copy_of_a_shared_node() -> None:
    """A node seen by two targets at different times must not share state between them."""
    table = table_from([(0, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4)])
    batch = NeighbourSampler(table, GnnConfig(protocol="P2"), 4).batch(np.array([2, 3]))
    assert batch.target_src[0] != batch.target_src[1]


# -- ports and structure --------------------------------------------------------------------------


def test_port_numbers_rank_neighbours_and_share_across_parallel_edges() -> None:
    #  node 0 pays node 1 twice, then node 2: out-ports 0, 0, 1.
    src = np.array([0, 0, 0])
    dst = np.array([1, 1, 2])
    in_ports, out_ports = _port_numbers(src, dst, np.arange(3), 3)
    assert out_ports.tolist() == [0, 0, 1]
    assert in_ports.tolist() == [0, 0, 0]


def test_port_order_follows_time_not_node_id() -> None:
    src = np.array([0, 0])
    dst = np.array([9, 1])
    _, out_ports = _port_numbers(src, dst, np.arange(2), 10)
    assert out_ports.tolist() == [0, 1]  # node 9 was paid first


def _cycle_vs_chain_data() -> tuple[EdgeTable, np.ndarray]:
    """Two graphs identical to a plain GNN's local view: edges either close a 3-cycle or don't."""
    edges: list[tuple[int, int, int]] = []
    labels: list[int] = []
    targets: list[int] = []
    node = 0
    minute = 0
    for i in range(240):
        a, b, c, d = node, node + 1, node + 2, node + 3
        node += 4
        closes = i % 2 == 0
        edges += [(a, b, minute), (b, c, minute + 1)]
        labels += [0, 0]
        edges.append((c, a if closes else d, minute + 2))
        labels.append(1 if closes else 0)
        targets.append(len(edges) - 1)
        minute += 3
    return table_from(edges, labels), np.array(targets)


def _fit_cycle_task(config: GnnConfig) -> float:
    table, targets = _cycle_vs_chain_data()
    torch.manual_seed(0)
    sampler = NeighbourSampler(table, config, len(table.src), 0)
    histogram = degree_histogram(table, np.ones(len(table.src), dtype=bool))
    probe = sampler.batch(targets[:4])
    model = MultiPna(probe.edge_attr.shape[1], probe.target_attr.shape[1], histogram, config)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    train, test = targets[:160], targets[160:]
    batch = sampler.batch(train)
    model.train()
    for _ in range(120):
        optimiser.zero_grad()
        loss = loss_fn(model(batch), batch.y)
        loss.backward()
        optimiser.step()
    return pr_auc(table.y[test], predict(model, sampler, test))


def test_ego_ids_let_the_model_detect_cycles_a_plain_gnn_cannot() -> None:
    """The theory behind the paper: without ego IDs, a node can't recognise itself in a loop."""
    multi = GnnConfig(protocol="P2", hidden=16, fanout=(5, 5), dropout=0.0)
    plain = replace(multi, ego=False, ports=False, reverse=False)
    assert _fit_cycle_task(multi) > 0.95
    assert _fit_cycle_task(plain) < 0.8


# -- training end to end --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def planted() -> EdgeTable:
    """Positive edges carry a signal in their attributes and in their local structure."""
    rng = np.random.default_rng(0)
    n = 6000
    src = rng.integers(0, 300, n)
    dst = rng.integers(0, 300, n)
    y = (rng.random(n) < 0.02).astype(np.int64)
    attributes = rng.normal(size=(n, 4)).astype(np.float32)
    attributes[y == 1, 0] += 3.0
    partition = np.searchsorted([3600, 4500, 5250], np.arange(n), side="right")
    return EdgeTable(
        src=src,
        dst=dst,
        minute=np.arange(n) // 2,
        attributes=attributes,
        y=y,
        partition=partition,
        origin=0,
        n_nodes=300,
        tx_id=np.array([f"t{i}" for i in range(n)]),
    )


def test_training_learns_and_scores_held_out_partitions(planted: EdgeTable) -> None:
    config = GnnConfig(
        hidden=16, fanout=(5, 3), epochs=4, batch_size=128, negatives_per_positive=5, patience=4
    )
    result = train_gnn(planted, config, val_negatives=400, log=False)
    assert result.best_val_pr_auc > 0.3
    assert len(result.history) >= 1
    scores = score_partitions(planted, result, config, (VAL_A, TEST))
    test_y = planted.y[planted.partition == TEST]
    assert pr_auc(test_y, scores[TEST]) > 5 * test_y.mean()
    assert np.all(planted.partition[planted.partition == TRAIN] == TRAIN)
