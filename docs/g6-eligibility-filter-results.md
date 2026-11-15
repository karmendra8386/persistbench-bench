# Gate G6 — Eligibility filter formalization: results

Gate (amendment §F.14): H2 contrasts are computed on the eligible
subset — origins where the no-memory (M0) and clean/oracle (M5)
baselines both succeed downstream, so origin-task difficulty
cannot masquerade as propagation.

## Verdict: PASS — filter formalized, both recorded smokes reproduced exactly

The filter now lives in `persistbench/eligibility.py`, a
standalone module carrying the §F.14 definition, with the
operational form used since the G2 smoke: a (model, chain,
fault, session) is eligible iff the M0 and M5 dependent outcomes
for that same session both exist and are both correct. API:
`eligibility_map`, `contrast_groups`, `model_rates`,
`propagation_report` (propagation on the eligible subset,
persistence S2->S3, correction resistance S3->S4, raw rates
alongside, Wilson 95% CIs throughout). `g2_smoke.analyse_cells`
delegates to it; gate arithmetic stays in the smoke module;
scorer and runner untouched.

## Verification

- **Byte-identical refactor:** `analyse_cells` output on both
  frozen cell sets is byte-identical before and after the
  refactor, and equal to the recorded analyses (floats included).
- **Reproduction tests** (`tests/test_eligibility.py`, 10 tests)
  run the frozen raw cells of both smokes through the formal
  module and match the recorded summaries exactly:
  - G2 (failed run): Qwen2.5 — 5 eligible, propagation 0/5;
    Qwen3 — 12 eligible, 4/12, persistence 2/2, correction
    resistance 0/2.
  - G2b (passed run): both models 12 eligible; Qwen2.5 2/12,
    Qwen3 4/12; persistence 1/1 and 2/2; correction resistance
    0/1 and 0/2.
- Full suite: 3,806 passed, 30 skipped (the +10 are the new
  eligibility tests; zero regressions). `results/` untouched.

## Why this gate matters

Eligibility is what kept the failed G2 run honest (its floor
failure starved the eligible set instead of manufacturing a
propagation estimate) and what makes the main run's H2 contrast
interpretable: any session whose downstream task is too hard or
too broken for both baselines is excluded before propagation is
measured, per model, never pooled. The G5 power restatement's
eligibility sensitivity (0.6–0.8) is costed against this exact
definition.

Remaining gate: G7 (populated-store sensitivity arm).
