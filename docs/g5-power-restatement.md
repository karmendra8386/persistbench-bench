# Gate G5 — Power restatement: results

Gate (amendment §C): with pilot effect sizes (M1 0.70,
M3-nonempty 0.50, per-fault dissociation), restate the n needed
per cell for the primary contrasts before freezing the main grid.

## Verdict: PASS — the frozen design powers every primary contrast

Computation: `scripts/power_restatement.py` (alpha 0.05
two-sided, power 0.80, normal approximations; chain = independent
unit per §F.16; models are separate arms, never pooled). Frozen
design after G4: 40 chains x 5 faults x 3 seeds = 600 origins per
model per policy, 120 per (model, fault).

| Contrast | Effect (pilot/G2b) | n needed | Delivered | Verdict |
|---|---|---|---|---|
| Formation M1 vs M3, paired (McNemar) | 0.700 vs 0.500 | 79 origins/model | 600 fault-pooled; 120/fault | Powered; per-fault detectable diff 0.162 < 0.200 |
| Formation rates (estimation) | — | — | CI ±0.040 per (model, policy); ±0.089 per (model, policy, fault) | Adequate |
| Propagation M1 vs M5 (eligible) | ~0.25 vs ~0.05 | 47/arm | 288–384 eligible/model (eligibility 0.6–0.8) | Powered with wide margin; rate CI ±0.043–0.050 |
| M3 visibility dissociation | 0.083 vs 0.750 | 5/arm | 240 invisible / 360 visible origins per model | Powered trivially |
| Persistence / correction resistance | conditional | — | ~72–96 propagation events/model, CI ±0.100–0.115 | Secondary: reported with intervals, no power claim |

Unit discipline, stated for the freeze: the fault-pooled
formation contrast is powered even at the strictest unit
(chain-fault pairs, n=200, detectable diff 0.125). Per-fault
formation contrasts are powered at the chain x seed unit (120)
with chain-clustered bootstrap intervals as §F.16 requires; at
the chain-only unit (40) the detectable paired difference would
be 0.280, above the pilot effect — so per-fault claims are
pre-registered to the chain x seed analysis, and no per-fault
claim is made below 12 chains (40 delivered).

## The restated grid cost (the honest number for the scale-up decision)

The original plan's "~3,600 sessions" was the **per-model**
figure, not the total. The frozen grid is 3,600 origin runs per
model (40 chains x 5 faults x 6 policies x 3 seeds), plus the
dependent arm (3 sessions per chain-run on the propagation
policies). Power is per-model, so the model set (§D.7: up to 5
candidates) is a budget choice, not a power requirement: cutting
models or the reference arm costs no power on the remaining arms.
Seeds stay at 3 (§D.8 minimum; at 2 seeds the per-fault paired
contrast drops to the borderline, detectable diff 0.198 vs the
0.200 effect).

Remaining gates: G6 (formalization), G7 (populated-store arm).
No scale-up is launched by this gate — the grid is costed and
powered on paper, and the spend decision stays with the user.
