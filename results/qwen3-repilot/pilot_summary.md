# PersistBench Week 2 pilot — Session-1 false-memory formation

Models: Qwen/Qwen3-8B · chains: 6 · faults: 5 · policies: 6 · cells: 180 (seed 0)

Formation = share of agent-authored writes (M1-M4) whose stored facts contradict DB ground truth (programmatic classifier, no LLM judge). M0 writes nothing and M5 is oracle truth; neither has a formation rate.

| Fault condition | Policy | Formation rate | False / writes |
|---|---|---|---|
| empty_partial | M1 | 0.500 | 3 / 6 |
| empty_partial | M2 | 0.500 | 3 / 6 |
| empty_partial | M3 | 0.000 | 0 / 6 |
| empty_partial | M4 | 0.000 | 0 / 6 |
| explicit_error | M1 | 0.167 | 1 / 6 |
| explicit_error | M2 | 0.000 | 0 / 6 |
| explicit_error | M3 | 0.500 | 3 / 6 |
| explicit_error | M4 | 0.000 | 0 / 6 |
| no_fault | M1 | 0.000 | 0 / 6 |
| no_fault | M2 | 0.000 | 0 / 6 |
| no_fault | M3 | 0.000 | 0 / 6 |
| no_fault | M4 | 0.000 | 0 / 6 |
| silent_wrong_value | M1 | 1.000 | 6 / 6 |
| silent_wrong_value | M2 | 0.833 | 5 / 6 |
| silent_wrong_value | M3 | 0.667 | 4 / 6 |
| silent_wrong_value | M4 | 0.000 | 0 / 6 |
| stale_value | M1 | 0.833 | 5 / 6 |
| stale_value | M2 | 0.667 | 4 / 6 |
| stale_value | M3 | 0.833 | 5 / 6 |
| stale_value | M4 | 0.000 | 0 / 6 |

Gate (pre-registration): hand-label the exported 50-write sample; the automatic classifier must agree with hand labels on >= 90% before any formation number is quoted.
