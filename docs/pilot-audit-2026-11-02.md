# PersistAIBench — Week-2 pilot audit (2026-11-02)

Written after the pilot results were reviewed and found disappointing.
Every number below is recomputed from `outputs/pilot_cells.jsonl` /
`outputs/pilot_writes.jsonl` by `scripts/pilot_stats.py`
(`outputs/pilot_stats.md`). Verdict up front: **the pilot's one real
signal is genuine but narrow; the pooled headline numbers should never
have been presented as the result.** Nothing here is scaled until the
fixes and gates in `preregistration-amendment-2026-11-02.md` are met.

## What actually holds

Qwen2.5-7B-Instruct, the only functioning arm, on 30 shared origins:

- M1 (naive self-summary) formation **0.700** [Wilson 0.521, 0.833];
  by fault: silent_wrong_value, stale_value, empty_partial all
  **1.000** [0.610, 1.000] (6/6 each), explicit_error 0.500,
  no_fault 0.000. The dissociation is real: honest failures are
  reported half the time; silent faults are *always* stored as fact.
- M3 (evidence contract) halves it: 0.333 headline, **0.500 on
  non-empty writes only** — paired McNemar vs M1: b=11, c=0,
  p≈0.001. M3 also *abstains*: coverage 20/30 (it refuses to store
  claims it cannot evidence — that is a feature, and must be reported
  as coverage, not hidden in the denominator).
- M2 (disclosure prompt) does essentially nothing: 0.667 vs 0.700,
  McNemar b=1, c=0, p=1.0. A prompt asking for honesty does not fix
  formation. That is itself a publishable negative.
- Field-level: 28.7% of M1's stored entries are false (31/108).

## What does not hold — the flaws, ranked

1. **Half the experiment never ran.** Qwen3-8B produced empty stores
   in 89/90 M1–M3 cells (652 parse-log failures): its `<think>`
   reasoning consumed the 256-token budget before any FACTS block.
   Its 0.000s are absence of writes, not absence of false memory.
   **Every pooled number (M1 0.350 etc.) averages a real arm with a
   broken one and is withdrawn.** Per-model reporting only, from now.
   *Fix (shipped):* thinking disabled at the chat template for Qwen3
   (`enable_thinking=False`, Qwen's documented non-thinking mode),
   `strip_thinking()` defence in the backend, action parser and claim
   extractor, and a separate 512-token budget for writes
   (commit after v0.1.0; suite 1,167 passed).
2. **M4 = 0.000 is a tautology, not a finding.** M4 writes DB values;
   the classifier checks writes against the same DB. It cannot score
   otherwise. M4's real measures are utility and cost, neither of
   which the pilot collected (no downstream sessions ran). Even taken
   at face value, 0/30 only bounds formation at ≤0.095 (Clopper–
   Pearson), not zero.
3. **The headline hypotheses are untested.** H2 (propagation),
   H4 (persistence/correction) and H5 (economics) need Sessions 2–4.
   The pilot measured Session-1 formation only. Presenting it as
   "the results" overstated a Week-2 checkpoint as the study.
4. **The classifier gate was weaker than "100%" sounds.** 23 of the
   50 hand-labelled rows were empty/trivial stores — agreement on
   nothing. On the 27 non-empty rows agreement is still 27/27, but
   zero disagreements in 27 only bounds the disagreement rate at
   ≤11.1% (rule of three), and labelling was single-rater, by the
   author. The gate needs a stratified non-empty sample, a second
   rater, and Cohen's kappa (amendment, Gate G3).
5. **Coverage and formation were conflated.** A policy that writes
   nothing scores 0 formation. Every formation number must ship with
   its write coverage beside it (now in `pilot_stats.md`). Qwen2.5
   M3's honest headline is "0.500 formation at 67% coverage", not
   0.333.
6. **The token budget confounded the model comparison.** Even
   Qwen2.5 hit the write cap (its summaries truncate); Qwen3's origin
   runs burned ~5,492 tokens per origin vs Qwen2.5's ~859 (cells
   file, M0 rows) — mostly reasoning that never became an action.
   Cross-model comparisons at one fixed small budget measure budget
   fit as much as memory behaviour. The amendment fixes budgets per
   generation type and reports tokens per cell.
7. **Scale is pilot-scale.** 6 chains, 1 seed, n=30/policy/model.
   The per-fault CIs are ±0.30 wide. Nothing at this n supports a
   cross-model or cross-fault ranking beyond the M1 dissociation.
8. **Novelty needs re-positioning.** Related work has moved:
   AgentPoison (NeurIPS 2024) and MINJA (NeurIPS 2025) poison memory
   *adversarially*; eTAMP poisons web-agent memory from the
   environment; **MemForm (arXiv:2610.14431, under review at ICLR
   2027) benchmarks memory-formation poisoning directly** across five
   interfaces and six LLMs. LoCoMo / LongMemEval / BEAM measure
   recall, not falsity. PersistAIBench's defensible claim is narrower
   and still open: **non-adversarial, self-generated false memories
   from silent tool faults — no attacker — plus their downstream
   propagation/persistence and the priced verified-write fix.**
   Any "first benchmark" phrasing must be scoped to that.

## What changed because of this audit

- Code: Qwen3 thinking fix + think-stripping + write budget (above).
- Analysis: `scripts/pilot_stats.py` — per-model rates, Wilson CIs,
  coverage, non-empty-only rates, field-level rates, paired McNemar,
  gate re-audit. Pooled table demoted in `docs/results.md`.
- Design: amendment dated 2026-11-02 (gates G1–G5, added models,
  seeds, stratified hand-labelling, propagation as the headline).
- Position: pooled pilot numbers withdrawn wherever they were quoted
  as a two-model result (results doc, release record wording stands
  only for the Qwen2.5 arm + gate, with this audit linked).
