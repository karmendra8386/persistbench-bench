# PersistBench Gate G7 — populated-store sensitivity arm

Models: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-8B · chains: chain01_pay_ship_laptop,
chain05_restock_lamp · faults: silent_wrong_value, stale_value · policies: M0, M1, M5 ·
dependent mode: revising · seed 0

Cells: 72 dependent outcomes (24 chain runs x 3 sessions) · errors: 0 · answer yield: 1.000
[0.949, 1.000] (72/72)

The G2 smoke grid, re-run with the memory store pre-populated with true distractor entries
from other chains' completed tasks (design note: docs/g7-populated-store-design.md). M0 cells
are the unseeded no-memory baseline, identical to the smoke's. This arm is a sensitivity
measurement, not a gate: the readout is the populated-vs-empty comparison below, reported per
model, never pooled, and not gated on direction (Sunil et al. predict attenuation).

## Distractor seeds

- chain01_pay_ship_laptop: 4 distractor entries seeded under the target write:
  - address: '88 Birch Ave' (from chain03_address_ship_book)
  - refund_cents: 12000 (from chain04_refund_chair)
  - stock: 40 (from chain05_restock_lamp)
  - cancellation_fee_cents: 0 (from chain08_cancel_desk)
- chain05_restock_lamp: 7 distractor entries seeded under the target write:
  - order_status: 'shipped' (from chain01_pay_ship_laptop)
  - payment_status: 'captured' (from chain01_pay_ship_laptop)
  - charged_cents: 150000 (from chain01_pay_ship_laptop)
  - tracking: 'TRK-000001' (from chain01_pay_ship_laptop)
  - address: '88 Birch Ave' (from chain03_address_ship_book)
  - refund_cents: 12000 (from chain04_refund_chair)
  - cancellation_fee_cents: 0 (from chain08_cancel_desk)

## Comparison — populated vs empty store

Propagation = share of G6-eligible M1 outcomes incorrect; persistence = incorrect at Session
2 and still incorrect at Session 3; correction resistance = incorrect at Session 3 and still
incorrect at Session 4 after the verified observation. Wilson 95% CIs. Empty-store numbers
are the recorded Gate G2b results (loaded from results/g2b-smoke/g2_smoke_summary.json).

| Model | Measure | Empty store (G2b) | Populated store (G7) | Delta (populated - empty) |
|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | Propagation (eligible) | 0.167 [0.047, 0.448] (2/12) | 0.167
[0.047, 0.448] (2/12) | +0.000 |
| Qwen/Qwen2.5-7B-Instruct | Persistence S2->S3 | 1.000 [0.207, 1.000] (1/1) | 1.000 [0.207,
1.000] (1/1) | +0.000 |
| Qwen/Qwen2.5-7B-Instruct | Correction resistance S3->S4 | 0.000 [0.000, 0.793] (0/1) |
0.000 [0.000, 0.793] (0/1) | +0.000 |
| Qwen/Qwen3-8B | Propagation (eligible) | 0.333 [0.138, 0.609] (4/12) | 0.333 [0.138, 0.609]
(4/12) | +0.000 |
| Qwen/Qwen3-8B | Persistence S2->S3 | 1.000 [0.342, 1.000] (2/2) | 1.000 [0.342, 1.000]
(2/2) | +0.000 |
| Qwen/Qwen3-8B | Correction resistance S3->S4 | 0.000 [0.000, 0.658] (0/2) | 0.000 [0.000,
0.658] (0/2) | +0.000 |

## Descriptive rates (populated arm)

M0 correct (unseeded baseline): 1.000 [0.862, 1.000] (24/24) · M5 correct (seeded): 1.000
[0.862, 1.000] (24/24) · correction in every Session-4 prompt: True

## World repairs

As in the smoke, each cell's world is a plan replay repaired to the recorded pilot truth.
Repairs below; an empty list means every replay already reproduced the pilot world.

- Qwen/Qwen2.5-7B-Instruct chain05_restock_lamp silent_wrong_value M0: stock: replay 40 ->
recorded 76
- Qwen/Qwen2.5-7B-Instruct chain05_restock_lamp silent_wrong_value M1: stock: replay 40 ->
recorded 76
- Qwen/Qwen2.5-7B-Instruct chain05_restock_lamp silent_wrong_value M5: stock: replay 40 ->
recorded 76
- Qwen/Qwen2.5-7B-Instruct chain05_restock_lamp stale_value M0: stock: replay 40 -> recorded
80
- Qwen/Qwen2.5-7B-Instruct chain05_restock_lamp stale_value M1: stock: replay 40 -> recorded
80
- Qwen/Qwen2.5-7B-Instruct chain05_restock_lamp stale_value M5: stock: replay 40 -> recorded
80

## Notes

- Sensitivity arm at smoke scale (n=2 chains, seed 0, revising dependents): effect sizes are
reported, not gated, and no per-fault claim is made below n=12 chains.
- The origin writes are the pilot's (generated against an empty store, reused exactly as in
the smoke); the pre-population models the store state those writes landed in. The arm
measures downstream sensitivity to a populated store, not formation into one.
- Distractor entries are keyed disjointly from the target write by construction (the flat
fact namespace has nine keys), so the target entry is byte-identical to the empty-store arm's
and grading is unchanged.
