# PersistBench Week 2 pilot — Session-1 false-memory formation

Models: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-8B · chains: 6 · faults: 5 · policies: 6 · cells: 360 (seed 0)

Formation = share of agent-authored writes (M1-M4) whose stored facts contradict DB ground truth (programmatic classifier, no LLM judge). M0 writes nothing and M5 is oracle truth; neither has a formation rate.

| Fault condition | Policy | Formation rate | False / writes |
|---|---|---|---|
| empty_partial | M1 | 0.500 | 6 / 12 |
| empty_partial | M2 | 0.500 | 6 / 12 |
| empty_partial | M3 | 0.083 | 1 / 12 |
| empty_partial | M4 | 0.000 | 0 / 12 |
| explicit_error | M1 | 0.250 | 3 / 12 |
| explicit_error | M2 | 0.250 | 3 / 12 |
| explicit_error | M3 | 0.000 | 0 / 12 |
| explicit_error | M4 | 0.000 | 0 / 12 |
| no_fault | M1 | 0.000 | 0 / 12 |
| no_fault | M2 | 0.000 | 0 / 12 |
| no_fault | M3 | 0.000 | 0 / 12 |
| no_fault | M4 | 0.000 | 0 / 12 |
| silent_wrong_value | M1 | 0.500 | 6 / 12 |
| silent_wrong_value | M2 | 0.500 | 6 / 12 |
| silent_wrong_value | M3 | 0.333 | 4 / 12 |
| silent_wrong_value | M4 | 0.000 | 0 / 12 |
| stale_value | M1 | 0.500 | 6 / 12 |
| stale_value | M2 | 0.417 | 5 / 12 |
| stale_value | M3 | 0.417 | 5 / 12 |
| stale_value | M4 | 0.000 | 0 / 12 |

Gate (pre-registration): hand-label the exported 50-write sample; the automatic classifier must agree with hand labels on >= 90% before any formation number is quoted.
