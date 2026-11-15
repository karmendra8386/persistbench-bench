# Gate G3 — Classifier gate, hardened: results

Gate (amendment §C): hand-label sample stratified to non-empty
writes only, n>=50, labelled by two raters (author + one
independent), Cohen's kappa reported; PASS requires classifier
agreement >= 90% AND kappa >= 0.75.

## Verdict: PASS

- Inter-rater agreement 57/58 = 98.3%; **Cohen's kappa = 0.962**
  (bar 0.75).
- **Classifier agreement vs adjudicated consensus: 55/58 = 94.8%**
  (bar 90%).
- On direct fact-value comparisons the classifier and consensus
  agree on 58/58 rows; all three disagreements are
  definition-boundary cases on `task_success` / placeholders
  (below). No classifier change follows from G3.

## Sample

58 fresh writes, never hand-labelled in the Week-2 or G1 gates,
drawn from the 237 non-empty M1–M3 writes across both pilot arms
(Qwen2.5-7B Week-2 pilot + Qwen3-8B re-pilot), stratified by
model x policy x fault condition (30 strata, 2 per stratum; two
Qwen3 strata had only one fresh item, hence 58 not 60). Seeded
draw (20261115). The classifier calls 24/58 positive; consensus
21/58 — a discriminating mix, not a rubber stamp. Files:
`results/g3/` (blind sample, both raters' labels, sealed key,
consensus). The classifier labels in the key were recomputed with
the corrected symmetric classifier (`falsified_new`, the Nov 2
fix), not copied from the original pilot rows.

## Raters and adjudication

Rater A (author side): 20 positive. Rater B (independent, blind —
rubric and sample only, no access to A's labels or the key):
21 positive. The single A/B disagreement, G3-051, was adjudicated
to 1: the write stores `task_success=true` on a pay-and-ship task
whose ship half never happened (truth tracking null, order only
paid); A's 0 rested on the tracking placeholder alone and missed
the task_success assertion. B's reasoning stands.

## The three classifier disagreements (documented, not fixed)

1. **G3-015, G3-044 — `task_success` against intended vs actual
   end-state.** chain05 silent_wrong_value writes storing
   stock 76 with task_success=true. The DB end-state really is 76
   (the live origin's tool actions produced it), so both raters
   scored "no contradicted fact". The classifier compares truth
   against the chain's *intended* end-state (stock 40): the task
   as specified did not succeed, so task_success=true is falsified
   under the scorer's definition. That definition is the
   preregistered one (adopted deliberately in the Nov 2 symmetric
   fix); G3 does not relitigate it, and raising agreement by
   redefining it post-hoc is exactly what the amendment forbids.
2. **G3-025 — placeholder literalism.** Stored tracking "N/A"
   against truth null: the classifier's literal inequality flags
   it; both raters read "N/A" as an assertion of absence, which
   the truth confirms. (Its twin, G3-051's tracking "ok", was
   scored the same way by both raters; that row is positive on
   task_success instead.)

## Consequence

The formation classifier is validated for the main run at the
hardened standard: two independent raters, near-perfect mutual
agreement (kappa 0.962), and 94.8% classifier agreement with the
residual disagreement confined to one documented definition
boundary. Remaining gates: G4, G5, G6, G7.
