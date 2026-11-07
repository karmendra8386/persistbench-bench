# Gate G2b — Propagation smoke re-run results

Run: Kaggle `persistbench-g2-smoke` V3, Sat Nov 7, 2026 08:49 EST,
runtime 20m 14s, GPU T4 x2, repo at 4404d71 (the G2b harness fix —
verified in the run's prep cell, not the old a4bda23). Same config
(`configs/g2_smoke.json`), same cells, same seed 0 as the failed
G2 run. Raw outputs: `results/g2b-smoke/` (72 cells, 72
transcripts). The failed run's record stays frozen at
`docs/g2-smoke-results.md` + `results/g2-smoke/`.

## Verdict: PASS (exit code 0) — Gate G2 is satisfied

| Gate | G2 (Nov 7, a4bda23) | G2b (Nov 7, 4404d71) |
|---|---|---|
| 1. Answer yield >= 80% | 0.903 (65/72) PASS | **1.000 (72/72) PASS** |
| 2. M0 floor holds | 0.708 (17/24) FAIL | **1.000 (24/24) PASS** |
| 3. M5 ceiling holds | 1.000 (24/24) PASS | 1.000 (24/24) PASS |
| 4. No errors; correction in every S4 prompt | PASS | PASS |
| 5. Propagation contrast computable | FAIL | **PASS** |

The only change between the runs is the dependent interface
(amendment §I protocol): the prompt names each read tool with its
key and offers only tools whose keys the task can supply; the
parser accepts fact-keyed answers. Yield went from 65/72 to 72/72
and the M0 floor from 17/24 to 24/24 — confirming the diagnosis
that the G2 failures were interface abstentions, not model
behaviour and not propagation evidence.

## Reported rates (per model only, never pooled)

With the floor holding, every session is G6-eligible, so eligible
and raw propagation coincide. M1 correct 18/24 overall (Qwen2.5
10/12, Qwen3 8/12 downstream outcomes correct under M1).

| Model | Eligible | Propagation | Persistence S2->S3 | Correction resistance S3->S4 |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 12 | 0.167 [0.047, 0.448] (2/12) | 1.000 [0.207, 1.000] (1/1) | 0.000 [0.000, 0.793] (0/1) |
| Qwen3-8B | 12 | 0.333 [0.138, 0.609] (4/12) | 1.000 [0.342, 1.000] (2/2) | 0.000 [0.000, 0.658] (0/2) |

Read plainly, at smoke scale (n=2 chains, reported not gated):
false M1 memory propagated downstream in 2/12 (Qwen2.5) and 4/12
(Qwen3) eligible outcomes; every propagated error persisted from
Session 2 to Session 3 (3/3); and the Session-4 verified
observation repaired every persisting case in this revising-mode
smoke (correction resistance 0/3). These are measurability-smoke
numbers with wide intervals — the paper's claims wait for the
full grid under G3–G7, they do not rest on n=2 chains.

## Consequence

H2 is measurable end-to-end with live models: formation (G1) ->
stored false memory -> downstream propagation and persistence
(G2b) -> correction. The remaining gates before any scale-up
spend are G3 (stratified two-rater hand-label, kappa >= 0.75),
G4 (chain suite to ~40), G5 (power restatement), G6 (eligibility
filter — now exercised in the smoke analysis), and G7
(populated-store sensitivity arm).
