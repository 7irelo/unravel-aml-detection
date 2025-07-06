# Model card: Unravel detector

Performance figures are in [`results.md`](results.md), generated from `reports/`.

## What it does

Scores each transaction for likely money laundering. Scores rank a capacity-limited analyst queue. A high score is a reason for a person to look, not a finding.

## Model

LightGBM on 91 past-only features: transaction fields, rolling account velocity and counterparty diversity, round-amount ratios, pass-through behaviour, and bounded graph structure such as two-paths and cycles. Account and bank identifiers are excluded. Trained on negatives downsampled with prior correction, then calibrated on a later, separate window.

## Data

IBM AMLworld HI-Small: synthetic, agent-based, labelled by the generator. It contains no personal data.

## Intended use

A portfolio demonstration of AML monitoring methods. **Not for real compliance decisions.**

## Known limitations

- **Synthetic data.** Performance on real transactions is unknown. The authors themselves warn it may not translate one-to-one.
- **Generator artefacts.** 87% of laundering is ACH, and Wire and Reinvestment have none. Models lean on this, and `T3-nofmt` measures the dependence. The file also ends in a laundering-only tail, which inflates metrics that include it.
- **Few test days.** Only 2 test days have full volume, so daily capacity figures are indicative.
- **Long patterns.** Gather-scatter instances often run longer than the 96-hour window, so their full shape can't be seen by the features.
- **Label delay ignored.** Training assumes labels are known immediately. Real investigations take weeks.
- **Calibration depends on the base rate.** A calibrator fitted at one prevalence is wrong at another, so probabilities need monitoring and refitting.
- **Bounded graph work.** About 5% of events hit a work limit, flagged in the `graph_truncated` feature.

## Explainability

Each alert stores TreeSHAP reason codes in plain language. These explain why the model scored a transaction highly. They don't establish why the behaviour happened or show intent.

## Monitoring

PSI per feature and on scores, against a training reference, with levels calibrated from normal day-to-day variation. Injection scenarios are in `results.md`.
