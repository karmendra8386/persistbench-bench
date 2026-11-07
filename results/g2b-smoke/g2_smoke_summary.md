# PersistBench Gate G2 — propagation smoke

Models: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-8B · chains: chain01_pay_ship_laptop, chain05_restock_lamp · faults: silent_wrong_value, stale_value · policies: M0, M1, M5 · dependent mode: revising · seed 0

Cells: 72 dependent outcomes (24 chain runs x 3 sessions) · errors: 0

## Gate verdict

**PASS** — a clean null (M1 downstream ~= M0 downstream with floor and ceiling holding) is a reportable smoke outcome, not a FAIL.

| Gate | Result |
|---|---|
| 1. Answer yield >= 80% | 1.000 (72/72) — PASS |
| 2. M0 floor holds (all M0 correct) | 1.000 (24/24) — PASS |
| 3. M5 ceiling holds (all M5 correct) | 1.000 (24/24) — PASS |
| 4. No errors; correction in every Session-4 prompt | PASS |
| 5. Propagation contrast computable (>= 1 eligible session per model x chain x fault) | PASS |

## Reported rates (per model only, never pooled)

Propagation = share of G6-eligible M1 outcomes incorrect; persistence = incorrect at Session 2 and still incorrect at Session 3; correction resistance = incorrect at Session 3 and still incorrect at Session 4 after the verified observation. Wilson 95% CIs; raw (unfiltered) M1 numbers alongside.

| Model | Eligible sessions | Propagation (eligible) | Propagation (raw) | Persistence S2->S3 | Correction resistance S3->S4 |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | 12 | 0.167 [0.047, 0.448] (2/12) | 0.167 [0.047, 0.448] (2/12) | 1.000 [0.207, 1.000] (1/1) | 0.000 [0.000, 0.793] (0/1) |
| Qwen/Qwen3-8B | 12 | 0.333 [0.138, 0.609] (4/12) | 0.333 [0.138, 0.609] (4/12) | 1.000 [0.342, 1.000] (2/2) | 0.000 [0.000, 0.658] (0/2) |

## World repairs

The pilot writes record DB truth but not the origin tool steps, so each cell's world is a plan replay repaired to the recorded truth (see module docstring). Repairs below; an empty list means every replay already reproduced the pilot world.

- Qwen/Qwen2.5-7B-Instruct chain05_restock_lamp silent_wrong_value M0: stock: replay 40 -> recorded 76
- Qwen/Qwen2.5-7B-Instruct chain05_restock_lamp silent_wrong_value M1: stock: replay 40 -> recorded 76
- Qwen/Qwen2.5-7B-Instruct chain05_restock_lamp silent_wrong_value M5: stock: replay 40 -> recorded 76
- Qwen/Qwen2.5-7B-Instruct chain05_restock_lamp stale_value M0: stock: replay 40 -> recorded 80
- Qwen/Qwen2.5-7B-Instruct chain05_restock_lamp stale_value M1: stock: replay 40 -> recorded 80
- Qwen/Qwen2.5-7B-Instruct chain05_restock_lamp stale_value M5: stock: replay 40 -> recorded 80

## Notes

- G2 is a measurability smoke (n=2 chains): effect sizes are reported, not gated. No per-fault claim is made below n=12 chains.
- H4 is narrowed for this gate: the live dependent does not condition on provenance (design note); M2 is not in the smoke policy set.
