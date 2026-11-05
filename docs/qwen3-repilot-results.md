# Qwen3-8B re-pilot — Gate G1 results (2026-11-04/05)

Run: Kaggle `karmendra8386/persistbench-qwen3-repilot`, Version 1,
committed Nov 4, 2026 ~6:37 PM ET; Complete/Succeeded, 3h 13m 29s,
GPU T4 x2, repo commit `b21d63e` (1024-token generation floor for
Qwen3, on top of the Nov 2 non-thinking fix), config
`configs/g1_qwen3.json`, seed 0. Grid: Qwen3-8B only × 6 chains ×
5 faults × M0–M5 = **180 cells, verified complete** (180 unique,
0 duplicates). Raw outputs: `results/qwen3-repilot/`
(`pilot_cells.jsonl`, `pilot_writes.jsonl`, `pilot_formation.csv`,
`pilot_summary.md/json`, `handlabel_sample.csv`).

## Gate G1 verdict — FAIL, by 2 cells (strict reading)

| Gate element | Bar | Result |
|---|---|---|
| Write coverage, M1–M3 | ≥80% (72/90) | **70/90 = 77.8% — FAIL** (M1 26/30, M2 21/30, M3 23/30) |
| FACTS/CLAIM block present | ≥80% | pass in substance (0 unclosed `<think>` in 180 writes; the Nov 2 truncation mode is gone) |
| Parse failures, M1–M3 (summed extraction-log entries) | reduced vs Nov 2 | **77** (vs Qwen2.5-7B's 55; vs broken Nov 2 Qwen3's 328 summed / 108 distinct-origin) — PASS |
| Unclosed `<think>` truncation | 0 | **0/180 — PASS; the Nov 2 failure mode is fixed** |

The residual gap is not random: the 20 no-write cells concentrate
in **empty_partial (11/20)** — where the tool returns nothing, Qwen3
often writes nothing at all — and in M2 (9/20). That is a finding
in itself (abstention under empty results), not just a coverage miss.

**Ruling still his:** accept the arm with the coverage caveat
(reported as 77.8% coverage, formation conditional on writes), or
run a targeted re-pilot of the 20 no-write cells. Week-3 two-model
scale-up is NOT cleanly clear until he rules.

## Formation (Qwen3-8B, corrected symmetric classifier)

Rescored independently from the saved writes; the corrected
(task_success symmetric) classifier changes nothing here
(0 cells change vs the run's own labels).

| Policy | Formation | Non-empty only |
|---|---|---|
| M1 | 0.500 (15/30) | 0.577 (15/26) |
| M2 | 0.400 (12/30) | 0.571 (12/21) |
| M3 | 0.400 (12/30) | 0.522 (12/23) |
| M4 | 0.000 (0/30) | 0.000 |

By fault (M1): silent_wrong_value 1.000 (6/6), stale_value 0.833
(5/6), empty_partial 0.500 (3/6), explicit_error 0.167 (1/6),
no_fault 0.000. The M1 dissociation from the Nov 2 Qwen2.5 arm
replicates: silent faults are stored as fact; explicit errors
mostly are not.

M3 evidence-visibility split (amendment §F): visible faults
0.250 (3/12) vs invisible 0.750 (9/12) — the same boundary as
Qwen2.5 (0.083 vs 0.750), weaker on the visible side here.

Cross-model (both arms, per-model only — never pooled):
Qwen2.5-7B M1 0.700 vs Qwen3-8B M1 0.500; M2 0.700 (corrected)
vs 0.400; M3 0.400 (corrected) vs 0.400; M4 0.000 both.

## Hand-label gate — PASS (corrected definition)

50-write sample labelled from stored write + DB truth, auto-labels
stripped first, then adjudicated against the chain's intended
facts: **50/50 agreement = 100%** (≥90% required). Two rows
(HL-041, HL-045, both M3 explicit_error) initially looked like
disagreements on a fact-key-only check; adjudication shows the
classifier is right in both — the task succeeded and the stored
`task_success: false` contradicts it, which is a false memory
under the corrected symmetric definition. Discriminating sample:
33 non-empty, non-M4 rows, 33/33 agreement. Empty rows in the
sample: 8/50 (vs 23/50 in the Nov 2 sample).
