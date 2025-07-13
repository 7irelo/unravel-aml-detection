# Protocol changes

Changes relative to the original plan, with reasons. All of these were made before any model was trained.

| Change | Reason |
|---|---|
| Longest feature window is 96h, not 72h | Cycle, fan and scatter-gather instances run up to about 96h in the raw data. |
| Reports split out full-volume days from the tail | The audit found days 10 to 17 are about 60% laundering, an artefact that inflates metrics. |
| Test window is 10 calendar days, of which 2 are full-volume | The file spans 18 days rather than 10, which moves IBM's boundaries. |
| Validation B is the later validation day rather than a share of validation transactions by count | The two coincide here: the two validation days have equal volume. |

Changes made after training began, all before the test set was evaluated, unless stated:

| Change | Reason |
|---|---|
| Hyperparameter search reduced to 10 trials and 400 rounds | Limited CPU on the machine used, shared with other work. |
| Bootstrap reduced to 50 resamples for tree arms and 30 for GNN arms | Run time. Made during test evaluation, and affects only interval width. T3 seed 0 and seed 1 were evaluated twice as a result, which the ledger shows. G2 was also evaluated twice, because its first report failed to write on a full disk. |
| GNN arms use 1 to 3 seeds (the plan asks for 5, or 3 if budget forces it), 8 epochs, hidden size 48, neighbours (15, 5) and 20 negatives per positive, on CPU | No GPU available. These are budget runs, so they don't reproduce published GNN results and shouldn't be read as the limit of the method. |
| GNN arms are calibrated on seed 0 only | Scoring on CPU is slow. Ranking metrics are unaffected. |
| HI-Medium was not run | It needs about 9 GB of free disk, which wasn't available. |
| Drift alert level uses a multiple of the median daily PSI, not the worst day | With the worst day, weekends inflated levels so far that dropping a payment format went unnoticed. The cost is more false alarms on weekends. |
| Pattern detectors ignore recurring relationships and other payment formats | Normal activity around busy accounts swamped the shapes. Chosen on validation. |
| The published GNN reproduction with IBM's own code was not run | It needs a GPU: the paper reports about 8 hours per PNA run on a V100. |
| G2 and G3 gained seeds 1 and 2, and all GNN arms gained a calibration run | The claim about P1 and P2 shouldn't rest on one seed. It moved the full-volume gap from 42 vs 29 to 38.6 vs 30.7. These runs add 8 entries to the test ledger, with no settings changed. |
