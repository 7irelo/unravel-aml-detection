# Evaluation card: Unravel investigator

## What it does

For a high-risk alert, a LangGraph agent gathers evidence through read-only tools, identifies a pattern or declines to, and drafts a case file in which every claim cites transaction IDs. An analyst decides what happens next. Nothing is filed.

## Safeguards built in, and how each is tested

| Safeguard | Enforced by | Test |
|---|---|---|
| No access to labels | Postgres grants; the agent's tools have no label path | `test_agent_role_cannot_read_labels_or_write` |
| No access to the future | Every tool enforces the alert's as-of time | `test_tools_never_show_data_after_the_alert` |
| No fabricated citations | Verifier L1 and L2 | Mutation tests |
| No wrong numbers | Tools compute all figures; verifier L3 and L4 | Mutation tests |
| Pattern claims match their edges | Verifier L6 | Mutation tests |
| No filing path | No such tool, endpoint or status | `test_no_tool_can_file_or_write`, `test_a_case_cannot_be_marked_as_filed` |
| Failure is visible | Refusals, budget exhaustion and failed verification become `incomplete` | Graph tests |
| Bounded cost | `SpendGuard`, tool-round and revision limits | Graph and client tests |

## Measured without an LLM

In [`results.md`](results.md): the rule baseline's pattern accuracy on real alerts, and the verifier's detection rate per seeded error type. The scripted client exercises the flow end to end. It's a fixed template and not an AI, so its narrative quality says nothing about an LLM's.

## Not yet measured

This needs an API key and human time, and is one command away (`docs/runbook.md`):

- Citation support: whether cited transactions really support each claim.
- LLM pattern accuracy against the rule baseline.
- Rate of correctly declining false positives, and of missing laundering.
- Refusal rate, cost per case, and run-to-run consistency.
- Judge agreement with human ratings, using weighted kappa, and judge detection of seeded errors.

The API request shape is tested against a fake SDK client, including the rules on sampling parameters, forced tool choice and thinking blocks. It hasn't yet been run against the live API.

## Known limits

- **Omissions and framing.** The verifier checks what's stated, not what's left out. A draft can pass every check and still mislead by dropping contrary evidence, which is why the judge and human review exist.
- **Swapping a citation for another retrieved transaction with matching values** can pass deterministic checks, and only a judge or human would notice.
- **Pattern ground truth covers 62% of laundering** in HI-Small. The rest has no typology to score against.
- **Narrative format** borrows the who/what/when/where/why structure, and doesn't claim conformance with any regulator's format.
- **Rubric ratings** will come from a non-compliance-officer, so they measure narrative quality rather than regulatory adequacy.
