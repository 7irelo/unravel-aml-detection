# 6. Postgres rather than a graph database

**Status:** Accepted

**Context.** The investigator needs bounded neighbourhood queries of at most 3 hops inside a time window.

**Decision.** Postgres with `(src, minute)` and `(dst, minute)` indexes, and breadth-first expansion in the application. It also holds alerts, cases, audit events and LangGraph checkpoints, so there's one less service to run. Grants enforce that the agent's role can't read labels.

**Revisit if** neighbourhood queries become a measured bottleneck.
