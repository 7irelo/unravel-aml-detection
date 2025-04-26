"""Plain-English meaning of each feature, used for SHAP reason codes and drift alerts."""

from __future__ import annotations

from unravel.features.engine import (
    FEATURE_NAMES,
    GRAPH_FEATURES,
    TRANSACTION_FEATURES,
    WINDOW_LABELS,
)

_FIXED: dict[str, str] = {
    "log_usd": "Size of this transaction in USD",
    "log_paid_over_received": "Gap between value paid and value received, after FX conversion",
    "payment_format": "Payment format used",
    "cross_currency": "Paid and received in different currencies",
    "cross_bank": "Sender and receiver are at different banks",
    "self_transfer": "Sender and receiver are the same account",
    "minute_of_day": "Time of day",
    "amount_round_100": "Amount is a round multiple of 100",
    "amount_round_1000": "Amount is a round multiple of 1,000",
    "src_out_entropy_24h": "How evenly the sender spread payments across recipients over 24h",
    "dst_in_entropy_24h": "How evenly the receiver's funds came from different senders over 24h",
    "src_out_round_ratio_24h": "Share of the sender's payments over 24h with round amounts",
    "dst_in_round_ratio_24h": "Share of the receiver's inflows over 24h with round amounts",
    "pair_count_prior": "Earlier payments from this sender to this receiver",
    "pair_is_new": "First payment seen between these two accounts",
    "reverse_pair_count_prior": "Earlier payments in the opposite direction",
    "src_mins_since_out": "Time since the sender's previous payment",
    "src_mins_since_in": "Time since the sender last received funds",
    "dst_mins_since_in": "Time since the receiver last received funds",
    "dst_mins_since_out": "Time since the receiver last sent funds",
    "src_age_mins": "How long the sender has been visible in the data",
    "dst_age_mins": "How long the receiver has been visible in the data",
    "src_lifetime_out": "Total payments made by the sender so far",
    "dst_lifetime_in": "Total payments received by the receiver so far",
    "src_amount_z": "How unusual this amount is for the sender",
    "dst_amount_z": "How unusual this amount is for the receiver",
    "src_format_switch": "Sender changed payment format since their last payment",
    "src_currency_switch": "Sender changed currency since their last payment",
    "src_forward_ratio_6h": "Share of the sender's 6h inflows already paid onward",
    "src_forward_ratio_24h": "Share of the sender's 24h inflows already paid onward",
    "src_forward_ratio_96h": "Share of the sender's 96h inflows already paid onward",
    "dst_forward_ratio_24h": "Share of the receiver's 24h inflows already paid onward",
    "amount_over_last_in": "Amount relative to the sender's most recent inflow",
    "two_paths_6h": "Intermediaries linking sender to receiver within 6h (scatter-gather)",
    "two_paths_24h": "Intermediaries linking sender to receiver within 24h (scatter-gather)",
    "two_paths_96h": "Intermediaries linking sender to receiver within 96h (scatter-gather)",
    "reciprocal_96h": "Receiver paid the sender within the past 96h (2-hop cycle)",
    "cycle3_paths_96h": "Routes back from receiver to sender through one intermediary (3-hop cycle)",
    "cycle_min_hops_96h": "This payment closes a cycle of funds of this length",
    "shared_senders_96h": "Other senders into this receiver paying the same recipients (bipartite)",
    "dst_downstream_reach_96h": "How widely funds can travel onward from the receiver",
    "src_upstream_reach_96h": "How widely funds reached the sender from further upstream",
    "graph_truncated": "Graph features hit their work limit for this transaction",
}

_FLOW = {
    "src_out": "Sender's outgoing",
    "src_in": "Sender's incoming",
    "dst_in": "Receiver's incoming",
    "dst_out": "Receiver's outgoing",
}
_METRIC = {
    "cnt": "payment count",
    "sum": "payment value",
    "distinct": "number of counterparties",
}


def describe(name: str) -> str:
    """Return a one-line description, raising if a feature has no entry."""
    if name in _FIXED:
        return _FIXED[name]
    prefix, metric, window = name.rsplit("_", 2)
    if prefix in _FLOW and metric in _METRIC and window in WINDOW_LABELS:
        return f"{_FLOW[prefix]} {_METRIC[metric]} over {window}"
    raise KeyError(f"feature {name!r} has no registry entry")


def family(name: str) -> str:
    """Group a feature for ablation reports."""
    if name in TRANSACTION_FEATURES:
        return "transaction"
    if name in GRAPH_FEATURES:
        return "graph"
    if name.rsplit("_", 2)[0] in _FLOW and name.rsplit("_", 2)[1] in _METRIC:
        return "velocity_diversity"
    if "round" in name:
        return "round_amounts"
    if "forward" in name or name == "amount_over_last_in":
        return "pass_through"
    return "behaviour"


def all_described() -> dict[str, str]:
    return {name: describe(name) for name in FEATURE_NAMES}
