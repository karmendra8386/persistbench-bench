# Pilot deepened statistics (scripts/pilot_stats.py)

## Per model x policy
| Model | Policy | Formation (Wilson 95%) | Coverage | Formation, non-empty only |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | M1 | 0.700 [0.521, 0.833] | 30/30 | 0.700 [0.521, 0.833] |
| Qwen2.5-7B-Instruct | M2 | 0.667 [0.488, 0.808] | 30/30 | 0.667 [0.488, 0.808] |
| Qwen2.5-7B-Instruct | M3 | 0.333 [0.192, 0.512] | 20/30 | 0.500 [0.299, 0.701] |
| Qwen2.5-7B-Instruct | M4 | 0.000 [0.000, 0.114] | 30/30 | 0.000 [0.000, 0.114] |
| Qwen3-8B | M1 | 0.000 [0.000, 0.114] | 1/30 | 0.000 [0.000, 0.793] |
| Qwen3-8B | M2 | 0.000 [0.000, 0.114] | 0/30 | n/a (no non-empty writes) |
| Qwen3-8B | M3 | 0.000 [0.000, 0.114] | 0/30 | n/a (no non-empty writes) |
| Qwen3-8B | M4 | 0.000 [0.000, 0.114] | 30/30 | 0.000 [0.000, 0.114] |

## Qwen2.5-7B M1 by fault
| Fault | Formation (Wilson 95%) |
|---|---|
| no_fault | 0.000 [0.000, 0.390] |
| explicit_error | 0.500 [0.188, 0.812] |
| empty_partial | 1.000 [0.610, 1.000] |
| silent_wrong_value | 1.000 [0.610, 1.000] |
| stale_value | 1.000 [0.610, 1.000] |

## Paired McNemar (Qwen2.5-7B, shared origins)
- M1_vs_M2: b=1, c=0, exact p=1
- M1_vs_M3: b=11, c=0, exact p=0.000977
- M1_vs_M4: b=21, c=0, exact p=9.54e-07

M4 zero: 0/30, Clopper-Pearson 95% upper 0.095 (and zero by construction — see audit).

## Field-level (Qwen2.5-7B)
- M1: 31/108 stored entries false = 0.287
- M2: 30/106 stored entries false = 0.283
- M3: 10/59 stored entries false = 0.169

## Hand-label gate re-audit
- Overall: 50/50; empty/trivial rows: 23
- Non-empty only: 27/27; rule-of-three 95% upper on disagreement 0.111

## Tokens
- Qwen2.5-7B-Instruct: origin total 154,566, write total 41,387, origin per unique origin 859
- Qwen3-8B: origin total 988,500, write total 56,730, origin per unique origin 5,492
