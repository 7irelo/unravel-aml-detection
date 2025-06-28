# 5. Single-partition ingestion

**Status:** Accepted for this system. A partitioned design is described below.

**Context.** A transaction's features need state from both its sender and its receiver, which would hash to different partitions. The parity test also needs a strict total order.

**Decision.** `tx.raw` has one partition and one consumer. It's deterministic, simple, and far above the average live rate.

**If volume ever required scale-out,** the design would be:
1. Split each transaction into two per-account events, keyed by account.
2. Compute side features in per-partition state stores.
3. Join the two sides by `tx_id` with a bounded wait.
4. Handle graph features separately, as they need cross-partition reads. Use a shared graph store with slightly stale reads, or a periodic batch.

Step 4 is the real cost. It gives up exact causal order across accounts, so offline training would need to simulate the same staleness to keep parity.
