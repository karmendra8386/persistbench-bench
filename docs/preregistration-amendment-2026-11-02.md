# Pre-registration amendment — 2026-11-02

Appended per the design-freeze rule (changes are dated amendments,
never silent edits). Trigger: the Week-2 pilot audit
(`docs/pilot-audit-2026-11-02.md`). Everything not mentioned here
stands as frozen on 2026-10-23.

## A. Reporting rules (effective immediately, retroactive to the pilot)

1. **Per-model reporting only.** No pooled formation rate may be
   quoted across models. A model arm with write coverage <80% on
   M1–M3 is a measurement failure and is reported as such, never
   averaged in.
2. **Coverage beside every rate.** Formation is always reported as
   (formation rate, write coverage). Non-empty-only formation is a
   required secondary column.
3. **CIs mandatory.** Wilson 95% CIs on every rate; paired contrasts
   by exact McNemar on shared origins; field-level formation as a
   secondary metric.
4. **M4 framing.** M4 formation is by construction; M4 is evaluated
   on Legitimate-Memory Utility and Cost per Correct Downstream Task
   only. Any M4 zero carries its Clopper–Pearson bound.

## B. Generation settings (the Qwen3 fix)

5. Thinking models run in their documented non-thinking mode
   (`enable_thinking=False` for Qwen3); reasoning blocks are stripped
   before parsing as defence in depth, and reasoning text never
   counts as evidence or assertion.
6. Budgets split by generation type: tool actions at the backend
   default (256), memory writes at 512 tokens. Budgets are reported
   per cell in the run record.

## C. Gates before any scale-up run (all must pass)

- **G1 — Qwen3 re-pilot.** The fixed Qwen3 arm re-runs the 6-chain
  pilot grid. Pass: M1–M3 write coverage ≥80% AND a FACTS/CLAIM
  block present in ≥80% of writes (baseline: 0/30). If it fails, the arm is
  replaced (candidate: Qwen2.5-14B-Instruct or Llama-3.1-8B with
  verified write coverage) — not silently dropped.
- **G2 — Propagation smoke.** Sessions 2–4 (dependents + correction
  event) run end-to-end on ≥2 chains × 2 faults with live models,
  proving H2/H4 are measurable before the full grid spends GPU.
- **G3 — Classifier gate, hardened.** Hand-label sample is stratified
  to non-empty writes only (n≥50), labelled by two raters (author +
  one independent), Cohen's kappa reported; gate is ≥90% classifier
  agreement AND kappa ≥0.75.
- **G4 — Chain suite completion.** The suite reaches the frozen ~40
  chains (12 exist) with dependent tasks for each, or the prereg
  scale is formally reduced with the power analysis restated.
- **G5 — Power restatement.** With pilot effect sizes (M1 0.70,
  M3-nonempty 0.50, per-fault dissociation), restate the n needed per
  cell for the primary contrasts before freezing the main grid.

## D. Main-run design changes

7. **Models (4 → 5 candidates, final set frozen after G1):**
   Qwen2.5-7B-Instruct, Qwen3-8B (fixed), Llama-3.1-8B-Instruct,
   Mistral-7B-Instruct-v0.3, plus one larger reference arm
   (Qwen2.5-32B-Instruct, if Kaggle budget allows) to test whether
   formation is a small-model artifact.
8. **Seeds:** 3 seeds minimum for the main run (pilot was seed 0
   only); seed variance reported.
9. **Headline order swapped.** H2 (propagation: corrupted M1 worse
   than M0 downstream) is the paper's headline; formation (H1) is
   the mechanism, reported per model. A formation-only paper is not
   submitted.
10. **Related-work positioning.** MemForm (arXiv:2610.14431),
    AgentPoison, MINJA, eTAMP are cited and differentiated on the
    adversarial axis: PersistAIBench studies *self-generated* false
    memory under silent faults, with no attacker in the loop, plus
    propagation/persistence and verified-write economics. The
    contribution claim is scoped accordingly.
11. **Transfer arm kept:** one memory-framework arm (Mem0 OSS) on a
    chain subset, after the main grid, to bound generality — as
    Month-2 in the original freeze, now with G2's harness.

## F. Additions from the literature broadening (2026-11-02, later same day)

Full research memo: the ranked gap list in the audit's research
appendix (AgentPoison, MINJA, MemForm, Sunil et al., LongMemEval,
LoCoMo, MemoryAgentBench, BEAM, A-MemGuard, MemoryArena, τ-bench).

12. **Write yield is a co-primary outcome.** Parse-failure rate and
    write coverage are reported for every arm alongside formation;
    a non-write is missing data, never a zero.
13. **M3 is always split by evidence visibility.** Evidence-visible
    faults (explicit_error, empty_partial) vs evidence-invisible
    (silent_wrong_value, stale_value), because the corrupted
    observation *is* M3's evidence for the second class. Pilot
    (Qwen2.5-7B): M3 formation 0.083 visible (1/12) vs 0.750
    invisible (9/12), against M1's 0.750 / 1.000 — the contract works
    exactly where evidence can contradict the claim, and largely
    cannot where it can't. That split, not the pooled 0.333, is the
    M3 result.
14. **G6 — Eligibility filter for propagation (MemForm-style).**
    H2 contrasts are computed on the eligible subset: origins where
    the no-memory (M0) and clean/oracle (M5) baselines both succeed
    downstream, so origin-task difficulty (pilot origin success was
    only 38/60) cannot masquerade as propagation.
15. **G7 — Populated-store sensitivity arm.** Sunil et al. show
    pre-existing legitimate memory sharply attenuates poisoning.
    All pilot chains start from an empty store; before submission,
    at least one arm re-runs a chain subset with a pre-populated,
    distractor-bearing store, or the paper carries the empty-store
    scope limitation explicitly in the abstract.
16. **Chain is the independent unit.** Pilot CIs are write-level;
    with 6 chains the effective n for chain-level claims is 6. The
    main run reports chain-clustered bootstrap intervals, and no
    per-fault claim is made below n=12 chains per cell.
17. **Defence positioning (A-MemGuard).** H3's ordering is within
    the write-time family only; the paper positions M3/M4 against
    retrieval/action-time consensus defences (A-MemGuard) and
    reports M4 with its utility + token cost, A-MemGuard-style.
18. **Provenance gap in H4 — design decision at G2.** H4 invokes
    provenance/confidence tagging, but no current policy *consumes*
    provenance downstream (M2 annotates; nothing conditions on it).
    At the G2 smoke, either add a provenance-conditioned reader
    variant or narrow H4's wording. The hypothesis may not promise
    a mechanism the policy set doesn't implement.

19. **Symmetric success-flag scoring (audit fix).** Formation
    scoring flags a stored `task_success` that contradicts the
    outcome in either direction. Pilot writes were rescored under
    the corrected classifier (M2 0.700, M3 0.400 on Qwen2.5-7B);
    corrected numbers supersede originals. The G3 hand-label gate
    must be re-checked under this definition (the HL-006 edge case
    flips) before the gate is quoted again.

## G. What this amendment does NOT change

Fault taxonomy (5), policy set (M0–M5), programmatic scoring with no
LLM judge, paired-by-chain design, and the falsification branches —
including the branch where formation or propagation comes out null
and the paper reports that honestly.
