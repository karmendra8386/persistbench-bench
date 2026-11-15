# Gate G4 — Chain suite completion: results

Gate (amendment §C): the suite reaches the frozen ~40 chains with
dependent tasks for each, or the prereg scale is formally reduced
with the power analysis restated.

## Verdict: PASS — 40 chains, no scale reduction needed

`persistbench/chains.py` now defines 40 chains (chains 1–12
byte-unchanged; 13–40 appended). Every chain carries exactly
three dependent tasks (Sessions 2/3/4), Session 4 always a report
on the primary fact.

Suite composition (40): primary_fact — tracking 10, address 7,
charged_cents 6, refund_cents 5, stock 5, cancellation_fee_cents
4, order_status 2, payment_status 1 (no fact dominates; the last
two are new to the suite). Plan lengths: 1-step 21, 2-step 17,
3-step 2, across 14 distinct tool sequences (the original 12 had
7). New families include charge->refund, address-only updates,
update->charge/cancel/refund orderings, and 3-step
charge->address->ship chains.

## Validation

- Full test suite: **3,796 passed, 30 skipped** (baseline 1,235
  passed / 10 skipped; the parametrized chain tests scale with
  the suite; zero regressions). One test edited:
  `test_suite_has_12_chains` -> `test_suite_has_40_chains`.
- Scripted end-to-end sweep: **1,200 runs** (40 chains x 5 fault
  conditions x M0–M5 through `runner.run_chain`, scripted origin)
  — 0 exceptions, 0 failures; all 240 no-fault M0/M5 baseline
  dependent checks correct.

## Design constraints recorded for the paper's methods section

- `intended_facts` must be exactly the facts the plan's tool
  responses report (M3 evidence and the M4 store-equality tests
  pin the set per tool combination).
- A charge-critical step cannot be followed by `ship_order`
  (under empty_partial the unpaid order makes the ship error
  real, breaking M3's task_success contract); charge-critical
  chains pair charge with address updates or place charge last.
- Decision dependents cannot sit on nullable facts (None breaks
  the threshold comparison); decisions live on stock, prepaid
  refund_cents, and non-critical charged_cents only.
- `cancel_order` refuses captured orders, so charge->cancel is
  not a world flow; charge->refund is the valid reversal.
- Tracking and cancellation_fee_cents have no pre-step value, so
  they can be wrong fields but never stale fields.

Remaining gates: G5 (power restatement — now unblocked, with the
full 40-chain suite to cost against), G6, G7.
