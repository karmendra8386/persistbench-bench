# The verified-memory-write rule

*A transferable design rule from PersistAIBench, for anyone building
agents that write to long-term memory.*

## The rule

> **Verify a memory write against current state — never against the
> observation that produced it.**

An agent's memory write is a claim about the world ("the refund of
4,200 cents was issued", "the order shipped"). If you check that claim
against the tool responses the agent saw, you have checked nothing
when the tool response itself was wrong: a silently corrupted or stale
response *is* the supporting observation, so the false claim verifies
cleanly and enters long-term memory as "checked" fact — where later
sessions trust it more, not less, because it carries a verification
mark. The only check that catches this class of failure compares the
write against an authoritative state query at write time.

## Why the weaker guard fails — measured, not hypothetical

PersistAIBench compares six write policies under five seeded fault
conditions (design: `docs/design-freeze.md`). The two guards:

- **M3, evidence-contract write.** Each claim is stored only with a
  supporting tool observation, and marked `verified` when one exists.
- **M4, state-verified write.** Each asserted fact is checked against
  the database at write time. Where state holds a value, the write is
  **corrected** to it; facts state does not hold are **rejected**.
  Every falsified field is accounted for in the write log as
  corrected-or-rejected.

Week 2 pilot (live models, `docs/results.md`), formation rate — the
share of stored writes asserting a DB-contradicted fact:

| Fault | M3 (evidence contract) | M4 (state-verified) |
|---|---|---|
| explicit_error | 0.000 | 0.000 |
| empty_partial | 0.083 | 0.000 |
| silent_wrong_value | **0.333** | **0.000** |
| stale_value | **0.417** | **0.000** |

The pattern is exactly the design prediction, first asserted in the
scripted-agent tests and now measured with a live model: the evidence
contract works when the failure is *visible* in the observation
(explicit errors, empty responses), and fails when the failure is
*silent* — because under silent/stale faults the corrupted response is
itself the evidence. M4's formation rate was 0.000 in every fault ×
model cell of the pilot (0 false writes in 60). Just as important,
under `no_fault` M4 stored everything and matched the unguarded
policy's store: the guard does not destroy legitimate writes (a
pre-registered falsification check — a guard that mangles true writes
is not viable).

## Adopting it — a checklist

1. **Treat every agent memory write as untrusted input.** The writer
   is the component that just consumed possibly-corrupt tool output.
2. **Separate "observed" from "true" in the write path.** If your
   verification step reads the same responses the agent read, it is a
   formatting check, not a truth check.
3. **Verify against state at write time, not read time.** A false
   memory corrected on read still propagates to every consumer that
   doesn't re-verify; a write that never lands propagates nowhere.
4. **Correct where state holds a value; reject only what state cannot
   support.** Rejection alone throws away the true facts mixed into a
   partially false write and teaches nothing; correction preserves the
   write's utility and leaves an audit trail.
5. **Log every corrected/rejected field per write.** The accounting is
   what makes the guard auditable — and the correction rate is itself
   a monitoring signal for upstream tool corruption.
6. **Test the guard on clean traffic.** Measure what it does to true
   writes under no-fault conditions before trusting its fault numbers.

## Limits, honestly

- **M4 needs a queryable source of truth at write time.** Where none
  exists, an evidence contract (M3) is still worth having — it stops
  the visible-error class completely in this benchmark — but it must
  not be *represented* as protection against silent corruption. That
  misrepresentation is the failure mode this rule exists to prevent.
- **Verification is not free.** State checks cost a query (and latency)
  per write; the accompanying paper's framing — the *cost* of verified
  memory writes — is the trade-off to budget, weighed against the
  downstream cost of acting on a false memory across sessions.
- **Pilot scale.** These numbers come from a 360-cell pilot (n = 12–30
  writes per cell) with one fully-measurable model arm; a second arm's
  write generation failed for unrelated reasons and is reported as
  such in `docs/results.md`. Treat the M3/M4 contrast as a strong
  directional result with a mechanism behind it, not a final effect
  size.
