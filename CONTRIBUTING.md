# Working rules

- **Results come from scripts.** No hand-typed metrics in docs. Every number traces to a file in `reports/`.
- **The protocol comes first.** `docs/protocol.md` was fixed before models were trained. Later changes go in `docs/protocol-changes.md` with a reason.
- **Test data is recorded.** All development uses validation. Each test evaluation leaves an entry in `reports/<dataset>/test_ledger.jsonl`, and the write-up reports the count.
- **Parity is non-negotiable.** Features have one implementation, `FeatureEngine`. If a change needs a separate batch path, redesign it.
- **Nothing is filed.** No code path may file a report. The agent's database role can't read labels or write anything.
- **Secrets stay out of git.** The repository is public.
- **Small commits** with Conventional Commit messages, each with passing checks.
- `make check` needs no data, GPU, broker or LLM key, and must stay that way.
