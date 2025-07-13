# Unravel

An anti-money-laundering monitoring system: a streaming feature pipeline, tree and graph neural network detection on IBM's AMLworld dataset, and an LLM investigator that drafts cited case files for a human analyst to approve.

Nothing is filed automatically. There's no code path that could.

<!-- RESULTS:START -->
## Results

On IBM AMLworld HI-Small, using LightGBM with 5 seeds on a strict temporal split. Full results are in [docs/results.md](docs/results.md).

| Features | Minority F1 %, all test days | Minority F1 %, full-volume days |
|---|---|---|
| Transaction fields only | 21.5 ± 0.5 | 13.4 ± 0.3 |
| + account history | 71.6 ± 0.9 | 60.8 ± 0.7 |
| + graph features | 73.4 ± 1.4 | 61.6 ± 1.6 |

The all-days column uses the same day boundaries as published work, where the best tree baseline reaches 63.2 and the best GNN 68.2. Those figures aren't directly comparable: features and sampling differ. The file ends in a laundering-only tail that inflates any metric including it, so full-volume days are the honest estimate.

- **At 100 alerts a day:** 99.8% precision and 20.9% recall, against a recall ceiling of 20.9%, on just 2 full-volume test days.
- **The main limitation:** the model recalls 85% of laundering in named patterns but only 2% of "natural" laundering, such as paying employees, which makes up 39% of positives on those days.
- **Parity:** all 5,078,345 feature vectors read back from Redpanda were identical to offline, across real consumer crashes.
- **Causal evaluation costs real performance:** the same Multi-PNA on the same budget scores F1 38.6 ± 3.3 when neighbours may include later edges, as in published protocols, and 30.7 ± 2.0 with strictly earlier edges (3 seeds each, full-volume days).
- **GNNs:** CPU-budget runs only. They trail the trees and don't reproduce published results, so a GPU run is needed for a fair comparison.
- **LLM narratives:** the agent and evaluation are built and tested with a scripted client. Quality on a real model is not yet measured, as it needs an API key.

<!-- RESULTS:END -->

## How it fits together

```text
 AMLworld CSV ──► ingest ──► canonical Parquet (labels held separately)
                                │
              ┌─────────────────┴──────────────────┐
              │ OFFLINE                            │ ONLINE
              ▼                                    ▼
    backfill runner                         replayer (labels stripped)
              │                                    │ tx.raw (Redpanda)
              └───────► FeatureEngine ◄────── feature consumer ── snapshots + offsets
                      (one implementation)         │ tx.features
              │                                    ▼
              ▼                             scorer ── calibrator ── capacity queue
    training and evaluation                        │ alerts
              │                                    ▼
              └──► versioned model ──►      Postgres (transactions, alerts, cases)
                                                   │
                                                   ▼
                               LangGraph investigator (read-only tools, as-of cutoff)
                                 gather ► detect ► draft ► verify ► revise (bounded)
                                                   │
                                                   ▼ interrupt
                                    Analyst review: approve, edit or reject

  Prometheus + Grafana: throughput, latency, alert rate, threshold
```

## What makes it credible

**Training and live scoring use the same features, and a test proves it.** There's one implementation, `FeatureEngine`, used by both the backfill and the Kafka consumer. The parity suite replays everything through the broker and compares every vector, including after a `kill -9`, and it includes a deliberately skewed engine to prove it can fail.

**No future leakage, tested.** Features are computed before a transaction updates state, and property tests confirm that changing later events can't change earlier vectors. The GNN's causal mode gives each target its own subgraph of strictly earlier edges. Published GNN protocols let test edges see later edges, so this project evaluates both ways and reports the gap.

**The evaluation protocol came first.** Splits, arms and metrics were [committed](docs/protocol.md) before any model trained, and every look at the test set leaves an entry in a ledger.

**The audit changed the metrics.** The last 8 of the file's 18 days are about 60% laundering: a generator artefact that holds 41% of test positives and inflates any metric that includes it. Every result here is reported with and without that tail.

**Precision at analyst capacity, with honest limits.** "Top 100 alerts a day" is reported with its recall ceiling, in transaction and case units, and against what a live policy can achieve without seeing the rest of the day.

**The LLM computes no facts.** Tools compute totals and detect structures. Each claim is structured data citing transaction IDs, and a deterministic verifier checks that IDs exist and were retrieved in that run, that numbers recompute, and that cited edges form the claimed shape. The agent's database role can't read labels, enforced by Postgres grants.

**Declining is a valid result.** Most alerts are false positives, so "close" is a first-class outcome and evaluation includes false-positive alerts.

## Quick start

No data, GPU, or API key needed. This runs the full flow on a synthetic fixture:

```bash
make demo
# or, without make:
docker compose -f deploy/compose.yaml --profile demo up --build -d
```

Then open the review UI at http://localhost:8501. The demo's investigator is a scripted template rather than an LLM, so it's useful for seeing the flow but says nothing about narrative quality.

To develop:

```bash
make install
make check      # lint, types and tests; no external services needed
```

To reproduce results on real data, and to use a real LLM or GPU, see the [runbook](docs/runbook.md).

## Repository

| Path | Contents |
|---|---|
| `src/unravel/features/` | The one feature implementation, and its registry of plain-English meanings |
| `src/unravel/eval/` | Metrics, cluster bootstrap, splits and the test ledger |
| `src/unravel/models/` | Tree arms, calibration and the Multi-PNA GNN with causal sampling |
| `src/unravel/streaming/` | Replayer, consumer, scorer and the parity check |
| `src/unravel/alerts/` | Online capacity queue |
| `src/unravel/agent/` | Investigator graph, tools, detectors, claims and verifier |
| `src/unravel/agent_eval/` | Case sampling, mutation tests and judge calibration |
| `src/unravel/monitoring/` | PSI drift monitoring |
| `src/unravel/api/`, `apps/review_ui/` | Analyst review |
| `deploy/`, `migrations/` | Compose stack, dashboards and schema |
| `docs/` | [Protocol](docs/protocol.md), [results](docs/results.md), [model card](docs/model-card.md), [evaluation card](docs/eval-card.md), [runbook](docs/runbook.md), [decision records](docs/adr) |

## Data and references

- Altman et al., [Realistic Synthetic Financial Transactions for Anti-Money Laundering Models](https://arxiv.org/abs/2306.16424), NeurIPS 2023. The [dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) is under CDLA-Sharing-1.0 and isn't redistributed here.
- Egressy et al., [Provably Powerful Graph Neural Networks for Directed Multigraphs](https://arxiv.org/abs/2306.11586), AAAI 2024.

The data is synthetic, and results may not carry over to real transactions. This isn't a compliance product.

## License

MIT
