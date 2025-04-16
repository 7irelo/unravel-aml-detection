# Evaluation protocol

Fixed before any model was trained. Later changes go in `protocol-changes.md`, with a reason.

Measured facts here come from `reports/HI-Small/data_audit.md`.

## Data

IBM AMLworld HI-Small: 5,078,345 transactions, 5,177 laundering (1 in 981), of which 3,209 (62%) belong to 370 named pattern instances.

### The tail

The file runs for 18 days, not the 10 stated in the paper. Days 10 to 17 hold 1,108 transactions, about 60% of them laundering: normal activity stops while laundering patterns play out to completion. This is a generator artefact with no real-world counterpart.

It falls entirely inside the test window under IBM's split, and contains 655 of the 1,611 test positives. Those transactions are very easy to rank, so any metric that includes them is inflated, including published results that use this split.

**Every test and validation report therefore gives two figures:**

- **All test days:** comparable with published numbers.
- **Full-volume days only:** days with at least 5% of median daily volume. This is the honest estimate, and the headline for capacity metrics.

## Splits

| Partition | Window | Transactions | Laundering | Use |
|---|---|---|---|---|
| Train | Sep 1 to Sep 6 | 3,248,921 | 2,530 | Fitting |
| Validation A | Sep 7 | 482,751 | 497 | Early stopping, hyperparameters, thresholds |
| Validation B | Sep 8 | 482,773 | 539 | Calibration only |
| Test | Sep 9 to Sep 18 | 863,900 | 1,611 | Final evaluation only |

Day boundaries come from a transcription of IBM's search, pinned by a test against their loop. It targets 60/20/20 by transaction count, and the result is 64/19/17. Only 2 test days have full volume, and one of them is a low-volume Saturday-like day, so daily capacity results are reported per day with the actual n.

## Protocols

| | P1, paper-compatible | P2, strictly causal |
|---|---|---|
| Purpose | Comparison with published work | Estimate of live performance |
| GNN neighbours | Anywhere in the snapshot, including later edges | Strictly earlier minutes only |
| Tree features | Past-only | Past-only |

## Arms

| Arm | Description |
|---|---|
| B0 | Random; rank by amount |
| B1 | Handwritten monitoring rules, thresholds set by judgement |
| T1 | LightGBM, per-transaction fields only |
| T2 | T1 plus rolling account features |
| T3 | T2 plus graph features |
| T3x | T3 with XGBoost |
| T3-nofmt | T3 without payment format, as a generator-artefact check |
| G1 | PNA |
| G2 | Multi-PNA + EU under P1 |
| G3 | Multi-PNA + EU under P2 |
| G4 | G3 plus T2 features on edges |

## Metrics

- **Ranking:** PR-AUC, alongside the base rate.
- **Published comparison:** minority F1 at a threshold chosen on validation A.
- **Capacity:** precision, recall and the recall ceiling at k ∈ {50, 100, 200, 500} per day, in transaction and case units, plus pattern hit rate and value captured. Also precision at a fixed 0.02% of volume.
- **Online regret:** a rolling-threshold policy against the offline top-k oracle.
- **Calibration:** log loss, Brier score, Brier skill and reliability on quantile bins. Fitted on validation B only.
- **Pattern recall:** per named pattern and for "natural" laundering. This is the direct test of whether GNNs catch what hand-built features miss.

## Statistics

- Five seeds per arm, reported as mean ± standard deviation, matching the papers. Any smaller count is stated.
- Confidence intervals from a cluster bootstrap by account-day.

## Test discipline

All development uses validation. Every test evaluation appends to `reports/HI-Small/test_ledger.jsonl`, and the write-up reports the count.

## Known risks to validity

- **ACH:** 87% of laundering is ACH, and Wire and Reinvestment have none. A model can lean on this, and it may not carry over to real data. `T3-nofmt` measures the dependence.
- **Boundaries:** 150 of 370 pattern instances straddle a split boundary, so early hops can appear in training and later hops in test.
- **Window length:** gather-scatter instances have a median span of about 150 hours, longer than the 96-hour window.
- **Synthetic data:** results describe AMLworld, and may not transfer to real transactions.
