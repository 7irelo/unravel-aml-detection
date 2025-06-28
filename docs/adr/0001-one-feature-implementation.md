# 1. One feature implementation for offline and online

**Status:** Accepted

**Context.** Skew between training features and live features is where many fraud systems quietly fail. The usual setup has a batch version in SQL or Spark and a separate streaming version, which drift apart.

**Decision.** `FeatureEngine.process(event)` is the only implementation. Offline backfill and the Kafka consumer both feed it events in the same canonical order. Features are computed before an event updates state, using event time only.

**Consequences.**
- Parity comes from construction, and the suite in `tests/test_streaming.py` verifies it, including a deliberately skewed engine that must fail.
- Backfill runs one event at a time in Python, which is slower than a columnar batch job. That's an acceptable cost at this scale, and this is why the consumer is Python, not Go (see ADR 2).
