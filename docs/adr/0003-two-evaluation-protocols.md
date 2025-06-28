# 3. Two evaluation protocols

**Status:** Accepted

**Context.** IBM's Multi-GNN code builds its test graph from all edges, and its loader samples with no time constraint. A test edge can therefore draw on edges later than itself. That's a fair description of classifying a finished batch, but a live stream can't do it.

**Decision.**
- **P1, paper-compatible:** neighbours from anywhere in the snapshot. Used only for comparison with published results.
- **P2, strictly causal:** each target gets its own disjoint subgraph of strictly earlier edges. Headline system claims use this.

Tree features are past-only under both.

**Consequences.** The gap between P1 and P2 is a result in itself. P2 costs more compute, because subgraphs can't be shared across targets. Comparing P2 numbers directly against published GNN numbers would be unfair to P2, so reports always name the protocol.
