# Gate G7 — Populated-store sensitivity arm: results

Gate (amendment §F.15): all pilot chains start from an empty
store; Sunil et al. show pre-existing legitimate memory sharply
attenuates poisoning. The gate requires at least one arm
re-running a chain subset with a pre-populated, distractor-bearing
store before submission (or the empty-store limitation stated in
the paper's abstract).

## Verdict: PASS — arm ran as designed; result is a clean null

Run: Kaggle notebook `g7-arm` v2, repo HEAD 562896b verified in
the run's prep cell, T4 x2, 2026-11-15. Exit 0 (a measurement,
not a verdict). 72 dependent outcomes, 0 errors, answer yield
1.000 (72/72); M0 (unseeded baseline) 24/24; M5 (seeded) 24/24.

## Design (docs/g7-populated-store-design.md)

The G2b grid exactly (Qwen2.5-7B + Qwen3-8B × chain01 + chain05
× silent/stale × M0/M1/M5, revising dependents, seed 0, pilot
origin writes reused), with one change: before the origin write
lands, the store is seeded with TRUE distractor entries drawn
from other chains' worlds — chain01: 4 distractors (address,
refund, stock, fee); chain05: 7 — for a 9-entry populated store,
the maximum the bench's flat 9-key store namespace honestly
carries with keys disjoint from the target write. M2/M3/M4 are
excluded with documented store-honesty reasons (M3 claims
require trajectory evidence pointers; M4's store is target-only
by construction; M2 provenance covers only the target write);
the G2 grid never used them either.

## Result: every delta is +0.000

| Model | Measure | Empty (G2b) | Populated (G7) |
|---|---|---|---|
| Qwen2.5-7B | Propagation (eligible) | 0.167 (2/12) | 0.167 (2/12) |
| Qwen2.5-7B | Persistence S2->S3 | 1.000 (1/1) | 1.000 (1/1) |
| Qwen2.5-7B | Correction resistance | 0.000 (0/1) | 0.000 (0/1) |
| Qwen3-8B | Propagation (eligible) | 0.333 (4/12) | 0.333 (4/12) |
| Qwen3-8B | Persistence S2->S3 | 1.000 (2/2) | 1.000 (2/2) |
| Qwen3-8B | Correction resistance | 0.000 (0/2) | 0.000 (0/2) |

Eligibility 12/12 per model, computed by the formalized G6
filter. World repairs match the pilot record exactly (Qwen2.5
chain05: silent -> 76, stale -> 80; verified against the run's
summary JSON and cells file — an early read of the summary
markdown misstated the silent value as 80; the run's own files
all agree on 76).

## Interpretation — what the null does and does not say

The distractors are key-disjoint from the target fact by
construction: a populated store in this bench cannot shadow the
target entry, so the arm tests whether *unrelated* legitimate
memory dilutes or distracts the dependent's use of a false
entry. At smoke scale it does not — propagation, persistence,
and correction resistance are numerically identical. The arm
does NOT test same-key competition (a prior legitimate entry
for the SAME fact), which the flat store cannot represent; that
remains a stated scope limit for the paper, alongside the
read-time-only population (origin writes were generated against
an empty store, as in the smoke). Within its scope, the
empty-store headline numbers are not an artifact of store
emptiness.

## Provenance

Frozen in `results/g7-arm/`: `g7_arm_summary.json` (validated
parse of the run's output) and `g7_arm_summary.md` (the run's
own summary text). Raw outputs incl. the 72 transcripts remain
on Kaggle (notebook `g7-arm`, version 2); the browser download
path could not save the raw cells file, and the summary JSON
was cross-checked against the cells file read from the run.

**All gates G1–G7 are now satisfied.** The main-run scale-up
(frozen grid costed in G5: 3,600 origin runs per model) is a
separate decision, not part of any gate.
