# PersistBench — Pre-Registration Draft (Week 1)

**Status:** DRAFT — to be frozen, with the analysis script, before the main run (Weeks 3–4). Companion to `docs/design-freeze.md` (frozen 2026-10-23).
**Author:** Karmendra Pandey. Nothing in this document is a result; thresholds are hypotheses to be tested.

## 1. Hypotheses (primary, directional)

- **H1 (formation, RQ1):** False-Memory Formation Rate under M1 (free-form self-summary) is above 20% for explicit failures, and higher for silent wrong values than for explicit failures.
- **H2 (propagation, RQ2) — headline test:** After a corrupted Session 1, M1 memory yields worse dependent-task performance in Sessions 2–4 than M0 (no memory), paired by chain. Memory becomes net-negative.
- **H3 (guard, RQ3):** Guard effectiveness orders as M2 (disclosure) < M3 (evidence contract) < M4 (state-verified); M4 eliminates fault-derived false memories by construction. Guards are also evaluated on cost (tokens) and on Legitimate-Memory Utility loss.
- **H4 (persistence & correction, RQ4):** False memories under M1 persist for at least 3 sessions, and a majority survive one contradictory verified observation unless writes tag provenance/confidence.
- **H5 (economics, RQ5):** M4's verified writes cost under 15% more tokens per Session-1 task than M1, but M4 reduces Cost per Correct Downstream Task relative to M1.

## 2. Conditions

- **Fault conditions (5):** no fault (control); explicit error; empty/partial result; silent wrong value; stale value. Seeded deterministic wrappers, injected at a fixed mid-task step of Session 1; identical faults across models and policies.
- **Memory policies (6):** M0 no memory; M1 free-form self-summary; M2 self-summary + disclosure instruction; M3 evidence-contract write; M4 state-verified write; M5 oracle truth (ceiling control).
- **Models (4):** Qwen2.5-7B-Instruct, Qwen3-8B, Llama-3.1-8B-Instruct, Mistral-7B-Instruct; 4-bit, vLLM, Kaggle T4.
- **Design:** ~40 task chains (12 in the Week 1 build; final count frozen at freeze time) × 6 policies × 5 faults × 3 seeds ≈ 3,600 sessions. Each chain: Session 1 origin task, Sessions 2–4 dependent tasks, one correction event before Session 4.

## 3. Primary metrics

1. False-Memory Formation Rate (per fault type × policy).
2. Cross-Session Propagation Rate (paired M1 vs M0 and M1 vs M5 on dependent tasks).
3. Persistence (sessions until a false memory stops influencing actions).
4. Correction Resistance (Session-4 influence after the correction event).
5. Legitimate-Memory Utility (no-fault control: M1 gain over M0; guard utility loss vs M1).
6. Cost per Correct Downstream Task (per condition; guard token cost included in the numerator).

All primary scoring is programmatic against database end-state. No LLM judge is used for any primary metric. The false-memory classifier is validated by a human spot-check of ~100 writes (pilot gate: ≥90% agreement with hand labels on 50 writes).

## 4. Analysis plan

- Unit of pairing: the task chain. Every policy × fault cell is compared on identical chains and identical seeded faults.
- Binary paired contrasts (dependent-task success; false-memory formation): **McNemar's test**.
- Rates, differences of rates, and cost ratios: **bootstrap 95% confidence intervals** (paired resampling over chains; seed fixed and recorded at freeze time).
- Primary contrast, pre-declared: **M1 vs M0 dependent-task success in the corrupted-fault cells** (H2). Secondary contrasts: M1 vs M5 (headroom), M2/M3/M4 vs M1 (guard effects), formation-rate ordering across fault types (H1), persistence/correction summaries (H4), cost-per-correct ratios (H5).
- The analysis script is written and frozen with this pre-registration, before any main-run data exists. Any analysis added later is labelled exploratory in the manuscript.

## 5. Falsification checks (pre-registered branches)

Any one of these reshapes the paper; each is reported honestly rather than designed away.

1. **Formation ≈ 0% under M1, even for silent faults.** First response (Week 2 pilot): pivot the fault mix toward silent and stale conditions. If formation is still near zero, the paper becomes the well-powered null result: the poisoning threat model does not generalise to benign self-generated outcome memories under these conditions, with boundary conditions measured.
2. **Propagation null.** Memory conditions are indistinguishable downstream. Formation without propagation is a writing quirk, not a reliability problem — and the paper must say so.
3. **Verified writes destroy legitimate utility.** If M4 prevents false memories only by preventing useful memory (Legitimate-Memory Utility collapses vs M1), the guard is not viable and the priced fix fails on its own terms.
4. **Chain suite too easy or too small to discriminate.** If all policies saturate, or differences are within noise at the planned scale, the suite — not the phenomenon — is the finding; the suite must be rebuilt or the claim narrowed.

## 6. Publishable branches (declared in advance)

- **Positive (H1/H2 confirmed):** first controlled demonstration that failed work becomes persistent false memory worse than no memory, plus a priced, policy-gated fix at the write.
- **Negative (formation or propagation ≈ 0 under M1):** a well-powered null with counterfactual memory controls, showing the adversarial poisoning picture does not generalise to benign outcome memories, with boundary conditions.
- **Mixed (judged most likely in the outline):** explicit failures reported honestly while silent wrong values are stored as fact — a dissociation that identifies exactly which fault class needs a write-time guard.

## 7. Exclusions and scope

- Claims are scoped to the measured regime: database-backed retail-style tasks, open 7–8B models at 4-bit quantisation. No claim beyond that regime.
- Month-2 transfer arms (LongMemEval; Mem0 open-source) bound generality; they do not replace the controlled suite.
- No training, no human subjects, no private data.
