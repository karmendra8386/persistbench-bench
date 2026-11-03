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
  pilot grid. Pass: M1–M3 write coverage ≥80% and parse failures
  reduced by ≥90% vs the 652 baseline. If it fails, the arm is
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

## E. What this amendment does NOT change

Fault taxonomy (5), policy set (M0–M5), programmatic scoring with no
LLM judge, paired-by-chain design, and the falsification branches —
including the branch where formation or propagation comes out null
and the paper reports that honestly.
