# Runbook

## Start and stop

| Goal | Command |
|---|---|
| Full demo on synthetic data, no credentials | `make demo` |
| Stop, keeping data | `make demo-down` |
| Stop and delete data | `make clean-demo` |
| Just the broker and database | `docker compose -f deploy/compose.yaml --profile core up -d redpanda postgres` |

| Service | Address |
|---|---|
| Review UI | http://localhost:8501 |
| Review API docs | http://localhost:8000/docs |
| Redpanda Console | http://localhost:8088 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Postgres | localhost:55432 |
| Kafka | localhost:19092 |

Without `make`, use `docker compose -f deploy/compose.yaml --profile demo up --build -d`.

## Reproducing the results

```bash
make install
make data NAME=HI-Small      # about 500 MB
make features NAME=HI-Small  # about 25 minutes
make trees NAME=HI-Small     # about 2 hours
uv run unravel report
```

Peak memory is about 4 GB for the feature engine and training. Features are read from disk on demand.

## Common problems

**The consumer stopped or crashed.** Restart it. It resumes from the latest snapshot in `artifacts/<dataset>/consumer_state.pkl`, re-emits anything after that point, and downstream readers upsert on `tx_id`. Re-emitted vectors are byte-identical, and `unravel stream-parity` will confirm it.

**Messages are landing in `dlq`.** Each carries a reason. `schema:` means a producer sent a field that isn't allowed, which is deliberate protection against labels reaching the engine. `out-of-order` means an event arrived behind the event-time clock, so check the producer's ordering.

**Parity failed.** Treat it as a release blocker: live scores no longer match what the model was trained on. `reports/<dataset>/parity.json` names the worst feature. Look for anything in the engine that reads outside the event, such as a wall clock or environment variable.

**A drift alert fired.** Check whether one feature or many moved.
- One feature, especially going null or constant, usually means a pipeline bug.
- Many features together suggest the population changed, so review calibration before trusting probabilities. A calibrator fitted at one base rate is wrong at another.

**Alert volume collapsed or the queue is full by morning.** The admission threshold comes from recent days' top scores, so a sudden shift in score distribution takes a few days to wash through. Check `unravel_alert_threshold` in Grafana.

**A case shows `incomplete`.** The agent was refused, ran out of budget, or couldn't pass verification after its revisions. The reason is on the case. An analyst can still reject it, but an incomplete case with no draft can't be approved.

**LLM spend stopped a run.** `SpendGuard` raises once the budget set by `UNRAVEL_LLM_BUDGET_USD` is used. Affected cases are marked `incomplete`. Raise the limit deliberately if needed.

**Docker's disk fills up.** A full replay writes several GB of topic data into Docker's virtual disk, which doesn't shrink by itself when containers go. Keep at least 30 GB free, and use `docker compose -f deploy/compose.yaml --profile demo down -v` when finished.

**A port is taken.** Change the host side of the mapping in `deploy/compose.yaml`, and the matching default in `src/unravel/api/settings.py`.

## Using a real LLM

```bash
export ANTHROPIC_API_KEY=...
UNRAVEL_LLM=anthropic make demo             # live investigator in the stack
uv run unravel agent-eval --budget 50       # evaluation on validation alerts
# agent-eval also writes reports/<dataset>/gold_sheet.csv: blind drafts for human rating.
uv run unravel gold-sheet                   # same sheet from template drafts, to try the rubric
```

## Running GNNs properly

CPU runs are smoke tests. For real results use a GPU, and `pip install` the CUDA build of torch.

```bash
uv run unravel train-gnn --arm G3 --protocol P2 --epochs 100 --fanout 100,100 \
    --negatives-per-positive 100 --batch-size 8192
```

To reproduce published numbers, first run IBM's own code at a pinned commit, in its own environment.
