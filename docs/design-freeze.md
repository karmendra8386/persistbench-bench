# PersistBench — Design Freeze (Week 1)

**Paper:** "Fail-Plausible, Then Persistent: Cross-Session Propagation of Silent Agent Failures Through Long-Term Memory — and the Cost of Verified Memory Writes"
**Author:** Karmendra Pandey
**Status:** FROZEN at Week 1 (2026-10-23), per the approved outline. Changes after this freeze require a dated amendment appended to this file and to the pre-registration draft — never a silent edit.
**Scope of Week 1:** design freeze, pre-registration draft, and the deterministic harness (world, chains, fault injection, scorer, scripted agents). No LLM runs. Live-model memory policies M2–M4 are built in Week 2; Week 1 freezes their interface (Section 5).

---

## 1. Research questions and hypotheses (pre-registered)

| Question | What it asks | Hypothesis |
|---|---|---|
| **RQ1 Formation** | How often does the Session-1 memory write falsify a failed or silently-wrong outcome — recording success, or a wrong value, as fact? | **H1:** False-Memory Formation Rate above 20% for explicit failures under the default self-summary policy, and higher for silent wrong values than for explicit failures. |
| **RQ2 Propagation** | Does a corrupted Session-1 memory change behaviour on different, dependent tasks in Sessions 2–4? | **H2:** Self-written memory after a corrupted Session 1 performs worse than no memory at all on dependent Sessions 2–4 tasks. Memory becomes net-negative. |
| **RQ3 Guard** | Which write-time guard works: disclosure instruction, evidence contract at write, or state-verified writes? | **H3:** Disclosure helps modestly, the evidence contract helps strongly, and state-verified writes eliminate fault-derived false memories by construction — at a measurable cost in tokens and in legitimate-memory utility. |
| **RQ4 Persistence** | How long does a false memory survive, and does contradictory evidence correct it? | **H4:** False memories persist for at least 3 sessions and survive one contradictory verified observation in a majority of cases, unless writes tag provenance and confidence. |
| **RQ5 Economics** | What do verified writes cost, and what do they save downstream? | **H5:** Verified writes cost under 15% more tokens per Session-1 task but reduce cost per correct downstream task. |

**Headline contrast (H2):** M1 (self-written memory) vs M0 (no memory), paired by chain, in the corrupted case — with M5 (oracle truth) as the ceiling showing how much headroom memory could have provided.

**What counts as a false memory:** a memory write that asserts, as fact, an outcome or value that database ground truth contradicts — a failed task recorded as completed, a silently wrong value recorded as the true value, a partial result recorded as complete, or a stale value recorded as current. The classifier is programmatic for structured writes; a human spot-check (~100 writes; ≥90% agreement with hand labels on a 50-write sample in the Week 2 pilot) validates it for free-form writes.

## 2. Environment spec

A τ-bench-style deterministic task world with **database ground truth**. Every task outcome is checkable against database state; scoring never depends on a judge's opinion.

- **Domain (Week 1 build):** retail-style orders/bookings. Tables: `products` (sku, price, stock), `orders` (status, payment status, address, payment id, tracking, cancellation fee), `payments` (amount, status), `shipments` (tracking → order, address, status).
- **Tools:** `get_order`, `get_product`, `get_shipment`, `charge_payment`, `ship_order`, `update_shipping_address`, `cancel_order`, `refund_order`, `restock`, `create_return_label`. All deterministic; all state transitions recorded in the DB.
- **Ground truth is correct by construction:** the scorer reads the same DB the tools mutate. The public τ-bench retail domain is a comparability check only; the self-built chain suite is the primary evidence (τ-bench airline annotations had known flaws, and the experiment needs chains, not isolated tasks).

### Task chains

12 chains in the Week 1 build, scaling to ~40 before the main run (exact final count frozen in the pre-registration). Each chain:

1. **Session 1 — origin task.** The agent attempts a task in which a fault may be injected at a fixed mid-task step. At session end it writes to memory under the assigned policy.
2. **Sessions 2–4 — three different dependent tasks.** Each later task's correct completion depends on what actually happened in Session 1. The agent begins each session with access to its memory.
3. **Correction event before Session 4.** One contradictory verified observation (a checked fact conflicting with a false memory, if one formed). Whether behaviour revises is the RQ4 correction test.

## 3. Fault conditions (independent variable)

Injected by seeded deterministic tool wrappers at a fixed mid-task step; identical faults for every model and policy.

| # | Condition | What the tool does | What it tests |
|---|---|---|---|
| i | No fault (control) | Returns the correct result; DB effect applied. | Legitimate-memory utility; clean baseline. |
| ii | Explicit error | Returns a visible error; DB effect **not** applied. | Whether an honest failure is still written as success. |
| iii | Empty / partial result | Returns an empty/partial result with no error; DB effect **not** applied. | Whether absence is mistaken for completion. |
| iv | Silent wrong value | DB effect applied correctly, but the response reports a plausible wrong value for one field. | The core fail-plausible case; expected hardest to catch. |
| v | Stale value | DB effect applied correctly, but the response reports the field's pre-call (formerly true) value. | Whether recency is checked at write time. |

## 4. Models and scale (main run; frozen here, executed Weeks 3–4)

- **Models (4):** Qwen2.5-7B-Instruct, Qwen3-8B, Llama-3.1-8B-Instruct, Mistral-7B-Instruct. 4-bit quantisation, vLLM, Kaggle T4. Serving settings reported in full.
- **Scale:** ~40 chains × 6 policies × 5 fault conditions × 3 seeds ≈ 3,600 short sessions.
- **Month-2 transfer arms:** LongMemEval transfer check; Mem0 open-source framework check. Neither replaces the controlled suite.

## 5. Memory policies — Week 1 interface freeze (build in Week 2)

The treatment. Everything except the write policy is held fixed within a chain.

**Frozen interface:** a policy is an object with `name` and `write(trajectory, chain, world) -> dict | None`, returning the memory store (a flat fact dict, or `None` for no memory) that later sessions read. Week 1 implements **M0** (returns `None`) and **M5** (oracle truth read from the DB) plus **M1 for scripted trajectories** (the agent's own claim, verbatim — the scripted stand-in for a self-summary). M2–M4 raise `NotImplementedError` until Week 2; their semantics are frozen here:

| Policy | Write rule | Role |
|---|---|---|
| **M0** | No memory. Nothing carried across sessions. | Floor. The H2 comparison. |
| **M1** | Free-form self-summary. The industry default. | Primary condition. |
| **M2** | Self-summary plus a disclosure instruction to report failures honestly. | Cheapest guard; prompt-only. |
| **M3** | Structured evidence-contract write: status (verified / failed / unknown) plus an evidence pointer for each claim. | Strong guard. |
| **M4** | State-verified write, checked against database ground truth before it is stored. | Strongest guard; eliminates fault-derived false memories by construction, at a price. |
| **M5** | Oracle human-truth memory. | Ceiling. |

Dependent-session agents read memory as a fact dict; with `None` (M0) they must query the world directly.

## 6. Metrics (primary; all programmatic, no LLM judge)

| Metric | Definition | RQ |
|---|---|---|
| **False-Memory Formation Rate** | Share of Session-1 writes that assert a contradicted outcome or value as fact, by fault type and policy. | RQ1 |
| **Cross-Session Propagation Rate** | Share of dependent Sessions 2–4 tasks whose failure is attributable to the false memory, in the paired M1 vs M0 / M5 comparison. | RQ2 |
| **Persistence** | Number of sessions until a false memory stops influencing the agent's actions. | RQ4 |
| **Correction Resistance** | Share of false memories still influencing behaviour in Session 4 after one contradictory verified observation. | RQ4 |
| **Legitimate-Memory Utility** | In the no-fault control, the gain of M1 over M0. Guards (M2–M4) are judged against this: a guard that destroys legitimate utility is not viable. | RQ3 |
| **Cost per Correct Downstream Task** | Total token cost, including the write-time guard, divided by downstream tasks solved correctly, per condition. Failed downstream work counts in the numerator. | RQ5 |

## 7. Statistics plan

Comparisons are **paired by chain** — each policy faces the same tasks and the same seeded faults. Binary paired contrasts use **McNemar's test**; rates and cost ratios carry **bootstrap confidence intervals**. The analysis script is written and frozen before the main run, as part of the pre-registration, so the choice of test cannot follow the results.

## 8. Week 1 exit criterion

The harness grades hand-written/scripted trajectories correctly in **all 5 fault conditions**: the scripted oracle agent scores correct (task success + truthful claim) under no-fault; under each injected fault, the scorer detects the outcome mismatch in the scripted naive agent's claim; and the oracle's claim is never falsified under any fault. Evidence: the pytest suite in `tests/` (see README for the run and counts).

## 9. Contributions (exactly three — frozen)

1. **PersistBench** — the first benchmark for cross-session propagation of silent failures through agent memory.
2. **The measurement** — formation, propagation, and persistence rates across 4 open models × 5 fault types × 6 memory policies, scored programmatically with no model judge for primary metrics.
3. **Verified memory writes as a priced intervention** — a policy-gated write-time guard, evaluated with cost-per-correct-downstream-task accounting and against legitimate-memory utility.

This paper is **not** a new architecture, and the contributions list must not drift into claiming one.
