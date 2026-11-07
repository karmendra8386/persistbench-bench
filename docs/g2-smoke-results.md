# Gate G2 — Propagation smoke results

Run: Kaggle `persistbench-g2-smoke` V2, Sat Nov 7, 2026 00:09 EST,
runtime 20m 38s, GPU T4 x2, repo at a4bda23. Config
`configs/g2_smoke.json`: Qwen2.5-7B-Instruct + Qwen3-8B x chain01
+ chain05 x silent_wrong_value/stale_value x M0/M1/M5, revising,
seed 0. Raw outputs: `results/g2-smoke/` (72 cells, 72 transcripts).

## Verdict: FAIL (exit code 1)

| Gate | Result |
|---|---|
| 1. Answer yield >= 80% | 0.903 (65/72) — PASS |
| 2. M0 floor holds (all M0 correct) | 0.708 (17/24) — FAIL |
| 3. M5 ceiling holds (all M5 correct) | 1.000 (24/24) — PASS |
| 4. No errors; correction in every S4 prompt | PASS |
| 5. Propagation contrast computable | FAIL |

The Kaggle run itself was clean: 72 cells, 0 errors, both pilot
stores reused, 6 recorded world repairs (Qwen2.5 chain05 stock
40 -> 76 silent / 40 -> 80 stale). The FAIL is in the gate checks,
not the execution.

## Reported rates (per model only, never pooled)

Propagation = share of G6-eligible M1 outcomes incorrect, Wilson
95% CIs; raw M1 alongside.

| Model | Eligible | Propagation (eligible) | Propagation (raw) | Persistence S2->S3 | Correction resistance |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 5 | 0.000 [0.000, 0.434] (0/5) | 0.167 [0.047, 0.448] (2/12) | n/a | n/a |
| Qwen3-8B | 12 | 0.333 [0.138, 0.609] (4/12) | 0.333 [0.138, 0.609] (4/12) | 1.000 [0.342, 1.000] (2/2) | 0.000 [0.000, 0.658] (0/2) |

These are smoke numbers (n=2 chains): reported, not gated, and
not a finding. The eligible set is small *because* the M0 floor
failed — eligibility requires the M0 baseline to succeed.

## Diagnosis of the 7 M0 failures (all Qwen2.5-7B, all believed=None)

No M0 failure is a wrong value. All 7 are abstentions
(`believed=None`, parseable=False, 4–5 invalid replies each), in
two harness-interface classes:

**Class A — the `get_shipment` trap (6/7: chain01, both faults,
S2/S3/S4).** chain01's fact is `tracking`; the only read tool that
discovers tracking from the task's order id is `get_order`.
`get_shipment` is keyed BY tracking — the caller must already know
the answer to call it. The prompt lists the three tools with no
keys. Qwen2.5 chose `get_shipment` (name match: tracking ->
shipment), passed `order_id`, got argument errors, repeated the
identical call until the 5-turn budget ran out. Qwen3 chose
`get_order` on the same prompt and answered correctly at once.

**Class B — answer-format mismatch (1/7: chain05 silent, S4).**
The model answered `{"stock": 76}` — the fact-keyed shape the
ORIGIN agent's claim format uses — instead of `{"answer": 76}`.
The parser rejected it ("unknown or missing tool None"), the model
fetched `get_product`, confirmed stock 76, answered `{"stock": 76}`
again, and exhausted the budget. It had the right value, including
the Session-4 correction, and still scored `believed=None`.

## What this means

The floor failure is a dependent-interface defect, not evidence
about propagation: the memory-reading path (M5, 24/24) and the
answer machinery (yield 90.3%) both work. G2 stays FAIL for this
run. No scale-up follows from it.

## Retry protocol (G2b, recorded in amendment §I)

1. The dependent prompt names each read tool with its key —
   `get_order(order_id)`, `get_product(sku)`,
   `get_shipment(tracking)` — and tools whose key the task cannot
   supply are not offered for that task.
2. The parser accepts the fact-keyed answer form
   `{"<fact>": value}` as an answer, and the invalid-reply error
   states the expected answer shape explicitly.
3. Same config, same cells, same seed. The retry is a harness fix
   re-run, not a new experiment; if the floor still fails, the
   failures are then model behaviour and are reported as such.
