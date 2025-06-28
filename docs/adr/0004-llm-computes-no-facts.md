# 4. Facts come from code, and the LLM interprets

**Status:** Accepted

**Context.** A regulated narrative can't rest on totals an LLM added up, or on citations it may have invented.

**Decision.** Tools compute all numbers and detect all structures. Claims are structured data with cited transaction IDs and machine-checkable values. A deterministic verifier checks existence, retrieval in this run, numbers, prose figures, timing and pattern shape. Only whether evidence supports a qualitative claim goes to a judge.

LangGraph handles orchestration, checkpoints and the human pause, and the official `anthropic` SDK is called directly from the model node. This keeps control of current API rules: no sampling parameters, no forced `tool_choice`, explicit effort, and unchanged replay of thinking blocks.

**Consequences.** Hallucinated IDs and wrong totals can't reach an analyst unflagged. The verifier doesn't catch misleading framing or omissions, which is why the judge and human review remain.
