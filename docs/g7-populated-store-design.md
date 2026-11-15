# Gate G7 — Populated-store sensitivity arm: design

Gate (preregistration amendment §F.15): Sunil et al. show
pre-existing legitimate memory sharply attenuates poisoning. All
pilot chains start from an empty store; before submission, at
least one arm re-runs a chain subset with a pre-populated,
distractor-bearing store, or the paper carries the empty-store
scope limitation explicitly in the abstract. This note is the
design of that arm. Implementation: `persistbench/g7_arm.py`,
`scripts/g7_arm.py`, `configs/g7_arm.json`, notebook
`kaggle/g7-arm.ipynb`. Tests: `tests/test_g7_arm.py`.

## The arm in one paragraph

The Gate G2 smoke grid, unchanged — Qwen2.5-7B-Instruct +
Qwen3-8B × chain01_pay_ship_laptop + chain05_restock_lamp ×
silent_wrong_value / stale_value × M0/M1/M5, revising dependents,
seed 0, the same reused pilot writes, the same world replay +
repair, the same G6 eligibility and per-model rates — except that
for the M1 and M5 cells the memory store handed to the dependents
is pre-populated with true distractor entries from other chains'
completed tasks. M0 cells have no store and are byte-identical to
the smoke's; they anchor eligibility exactly as before. The
readout is a comparison, per model, of propagation (eligible
subset), persistence S2→S3 and correction resistance S3→S4
between the populated arm and the recorded G2b empty-store run.
The arm is a sensitivity measurement: it is reported, not gated,
and not gated on direction — Sunil et al. predict attenuation,
but either direction is a result.

## Why the seeding looks the way it does

A policy's store in this harness is a **flat fact dict**. Later
sessions read it whole: the live dependent renders every
non-annotation key into its prompt as a `key: value` line, and
the scripted dependent looks facts up by key. There is no
per-entity namespacing and no retrieval step. The entire fact
namespace is eight keys — `order_status`, `payment_status`,
`charged_cents`, `tracking`, `address`, `stock`, `refund_cents`,
`cancellation_fee_cents` — plus `task_success`, which every write
carries.

That format decides what an honest distractor is. A distractor
entry can only occupy a fact key the target chain's write does
not use: anything else would overwrite, or be overwritten by, the
target entry itself — a store collision, not a distractor, and
one that would change what the scorer reads. Seeding is therefore
a merge with the seed keys disjoint from the target write's keys
by construction:

    populated_store = {**seed, **stored_write}

The merge order mirrors accumulation: the prior tasks' entries
were already in the store when this task's write landed. Because
the key sets are disjoint, the target entry surfaces
byte-identically, dependents are graded against the target
chain's truth exactly as in the empty-store arm, and eligibility
(§F.14) is computed identically. `seed_store` raises a reuse
error if a stored write ever carries a seed key — the arm must
never silently shadow the target entry.

## Distractor construction and provenance

Deterministic — no randomness, no sampling. For target chain T,
the excluded keys are T's intended facts plus `task_success`.
Walking the 40-chain suite in order, skipping T, each remaining
fact key is taken from the **first** chain whose intended facts
hold it. Every distractor value is thus another chain's intended
fact: true in that chain's world by construction (it is what that
chain's M5 oracle write stores, and `tests/test_g7_arm.py`
verifies each value against the source world after a clean plan
run). The seed for a target chain is a pure function of the
suite — identical across models, faults and policies; only the
target write differs between cells.

chain01_pay_ship_laptop (target keys: order_status,
payment_status, charged_cents, tracking, task_success) —
4 distractors:

| key | value | source chain |
|---|---|---|
| address | "88 Birch Ave" | chain03_address_ship_book |
| refund_cents | 12000 | chain04_refund_chair |
| stock | 40 | chain05_restock_lamp |
| cancellation_fee_cents | 0 | chain08_cancel_desk |

chain05_restock_lamp (target keys: stock, task_success) —
7 distractors:

| key | value | source chain |
|---|---|---|
| order_status | "shipped" | chain01_pay_ship_laptop |
| payment_status | "captured" | chain01_pay_ship_laptop |
| charged_cents | 150000 | chain01_pay_ship_laptop |
| tracking | "TRK-000001" | chain01_pay_ship_laptop |
| address | "88 Birch Ave" | chain03_address_ship_book |
| refund_cents | 12000 | chain04_refund_chair |
| cancellation_fee_cents | 0 | chain08_cancel_desk |

Populated store sizes: chain01 carries 4 distractors + the
target write's 5 fact entries = **9 entries**; chain05 carries
7 + 2 = **9 entries**. Both sit inside the 8–12-entry band for
the store as a whole. The distractor counts themselves (4 and 7)
are what the flat nine-key namespace honestly carries once the
target's own keys are excluded — see scope limits.

No distractor can mask the falsified entry: falsified fields are
always target-chain fact keys (or `task_success`), and those keys
are excluded from the seed by construction. A distractor also
never sits on the key a dependent session asks about — the asked
facts are the chains' own (tracking for chain01, stock for
chain05), both target keys.

## Per-policy handling

- **M0 (no memory):** nothing to populate. The arm's M0 cells
  are the empty-store baseline, unchanged — deliberately: they
  are half of the G6 eligibility anchor, and keeping them
  identical is what makes the populated-vs-empty comparison a
  one-variable contrast.
- **M1 (self-summary):** seeded. M1's store is the agent's claim
  dict, verbatim; a populated M1 store is exactly a flat dict of
  accumulated claim entries, which is what the merge produces.
- **M5 (oracle truth):** seeded. The distractors are true facts
  (in their source worlds), so the populated M5 store still
  contains no falsehood; the target chain's own entries remain
  the oracle's. M5 stays the ceiling anchor for eligibility.
- **M2 (disclosure): not seeded, excluded from the arm.** M2's
  `_provenance` / `_disclosures` annotate the target write's
  claims; seeded entries would sit in the fact store silently
  uncovered by the annotation, misrepresenting what the policy
  guarantees about its store's contents.
- **M3 (evidence contract): not seeded, excluded from the arm.**
  M3's invariant is that every fact entry corresponds to a
  `_claims` record whose evidence pointer cites an observation
  in the origin trajectory. A seeded entry has no observation in
  this chain's trajectory; writing one into an M3 store would
  fabricate the contract's evidence trail. (The dependent never
  renders `_claims`, so the exclusion is about the store's
  internal honesty, not about hiding entries from the reader.)
- **M4 (state-verified): not seeded, excluded from the arm.**
  M4's store is by construction exactly the target world's
  verified facts — at write time it rejects claimed keys that
  are not chain facts. Foreign entries are precisely what the
  policy exists to keep out; a "populated M4 store" is a
  contradiction in terms within this design.

M2/M3/M4 are not in the G2 grid either, so the exclusion costs
the arm nothing; `configs/g7_arm.json` validation rejects any
policy set other than exactly M0/M1/M5, with the reason stated.

## The comparison readout

`g7_arm_summary.md` / `.json` carry, per model (never pooled):

| Model | Measure | Empty store (G2b) | Populated store (G7) | Delta (populated − empty) |
|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | Propagation (eligible) | 0.167 [0.047, 0.448] (2/12) | measured by the arm run | computed |
| Qwen/Qwen2.5-7B-Instruct | Persistence S2→S3 | 1.000 [0.207, 1.000] (1/1) | measured | computed |
| Qwen/Qwen2.5-7B-Instruct | Correction resistance S3→S4 | 0.000 [0.000, 0.793] (0/1) | measured | computed |
| Qwen/Qwen3-8B | Propagation (eligible) | 0.333 [0.138, 0.609] (4/12) | measured | computed |
| Qwen/Qwen3-8B | Persistence S2→S3 | 1.000 [0.342, 1.000] (2/2) | measured | computed |
| Qwen/Qwen3-8B | Correction resistance S3→S4 | 0.000 [0.000, 0.658] (0/2) | measured | computed |

The empty-store column is loaded at run time from the recorded
`results/g2b-smoke/g2_smoke_summary.json` (config key
`baseline_summary`), never retyped into the arm; if that file is
unavailable the arm still completes and the column reads n/a.
The notebook's exit code reflects completion only (0 = the grid
ran and outputs were written; 2 = reuse failure, the smoke's
fallback contract). There is no PASS/FAIL.

## Scope limits, stated plainly

1. **Distractor load is bounded by the flat namespace.** Nine
   fact keys exist; the target write takes its share, and the
   rest is all a store can honestly hold. The populated stores
   (9 entries) are real multi-task stores, but they are not the
   hundreds-of-entries stores of a long-lived deployment, and
   entries are unattributed — the format has no way to say
   "this tracking number belongs to order O1007". That
   unattributed flatness is itself part of the treatment being
   measured, and it is a property of the benchmark's store
   format, not a shortcut taken by this arm.
2. **The origin write is the pilot's.** It was generated against
   an empty store and is reused exactly as in the smoke; the
   pre-population models the store state that write landed in.
   The arm measures downstream sensitivity to a populated store
   (propagation, persistence, correction), not whether formation
   itself changes when an origin writes into a populated store.
   That second question needs fresh origin generations and is
   out of G7's scope as frozen.
3. **Smoke scale.** n=2 chains, seed 0, revising dependents,
   two faults — the same limits as G2/G2b. The comparison is a
   sensitivity signal with wide intervals, reported as such; it
   discharges §F.15's "at least one arm" requirement, and the
   empty-store limitation does not need to go in the abstract.
