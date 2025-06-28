# 2. Python consumer, not Go

**Status:** Accepted

**Context.** The brief allowed a Go or Python consumer. Average HI-Small volume is about 6 transactions per second.

**Decision.** Python. A Go consumer would need a second feature implementation, reintroducing the risk ADR 1 removes, and the volume doesn't call for it.

**Revisit if** measured throughput falls below what's needed. Compile hot paths with Numba first, since this keeps one implementation.
