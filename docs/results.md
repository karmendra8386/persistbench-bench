# PersistAIBench — results

> **Audit notice (2026-11-02):** this pilot was audited the same day
> (`docs/pilot-audit-2026-11-02.md`). The **pooled column in the table
> below is withdrawn** — it averages one functioning arm with one
> broken arm. Read per-model numbers only, with coverage and CIs, in
> `outputs/pilot_stats.md` (recomputed by `scripts/pilot_stats.py`).
> Reporting rules and scale-up gates are in
> `docs/preregistration-amendment-2026-11-02.md`.

## Week 2 formation pilot (2026-11-02) — the first live-model results

**What this measures.** The Session-1 **false-memory formation rate**:
how often the memory write a model produces asserts, as fact, an outcome
the database contradicts. Formation is scored by the programmatic
classifier (`scorer.classify_write`) against DB ground truth — no LLM
judge. Formation is defined for the agent-authored policies **M1–M4**
only: M0 writes nothing, and M5 is oracle truth by construction.

**Run.** Kaggle `karmendra8386/persistbench-week2-pilot`, Version 1,
Save & Run All 2026-11-02, GPU T4 x2, runtime 2h 5m 19s, repo commit
`86bca21`. Grid: 2 models × 6 chains × 5 fault conditions × 6 policies
= **360 cells, verified complete** (360 unique model × chain × fault ×
policy combinations, no duplicates). The origin trajectory is generated
once per (model, chain, fault) and shared by all six policies — paired
by construction. Raw outputs: `results/pilot/` (`pilot_cells.jsonl`,
`pilot_writes.jsonl`, `pilot_formation.csv`, `pilot_summary.md/json`).

### Formation rate by policy × model (false writes / writes)

| Policy | Qwen2.5-7B-Instruct | Qwen3-8B | Pooled |
|---|---|---|---|
| M1 (self-summary) | **0.700** (21/30) | 0.000 (0/30)\* | **0.350** (21/60) |
| M2 (+ disclosure) | **0.667** (20/30) | 0.000 (0/30)\* | **0.333** (20/60) |
| M3 (evidence contract) | **0.333** (10/30) | 0.000 (0/30)\* | **0.167** (10/60) |
| M4 (state-verified) | **0.000** (0/30) | 0.000 (0/30)\* | **0.000** (0/60) |

\* The Qwen3-8B column is a **measurement failure, not a result** — see
the anomaly note below. Read the Qwen2.5-7B column as the pilot's signal;
the pooled column averages one real arm with one broken arm.

### Formation rate by fault × policy (pooled across models, false / writes)

| Fault condition | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| no_fault | 0.000 (0/12) | 0.000 (0/12) | 0.000 (0/12) | 0.000 (0/12) |
| explicit_error | 0.250 (3/12) | 0.250 (3/12) | 0.000 (0/12) | 0.000 (0/12) |
| empty_partial | 0.500 (6/12) | 0.500 (6/12) | 0.083 (1/12) | 0.000 (0/12) |
| silent_wrong_value | 0.500 (6/12) | 0.500 (6/12) | 0.333 (4/12) | 0.000 (0/12) |
| stale_value | 0.500 (6/12) | 0.417 (5/12) | 0.417 (5/12) | 0.000 (0/12) |

By fault overall (M1–M4): silent_wrong_value 0.333, stale_value 0.333,
empty_partial 0.271, explicit_error 0.125, no_fault 0.000.

### Classifier gate — PASS

Pre-registration requires the automatic classifier to agree with blind
hand labels on ≥90% of a 50-write sample before any formation number is
quoted. The sample was labelled blind (auto-labels stripped first;
judged from the stored write + DB truth alone): **50/50 agreement =
100%**, zero disagreements (`handlabel_sample_handlabeled.csv` in
`results/pilot/`). Caveats, stated plainly: 23 of the 50 sampled rows
are empty/trivial stores, so agreement is partly agreement on empty
writes; one row (HL-006) is a definition edge case — a stored
`task_success=false` for a refund that did succeed, labelled 0 under
the formation definition, matching the classifier.

### Anomaly — the Qwen3-8B arm produced almost no writes

Qwen3-8B's formation rate of 0.000 is an artifact of **write
generation**, not evidence that the model avoids false memories:

- Its stored writes are empty in 29/30 M1, 30/30 M2, and 30/30 M3 cells
  (empty object / no claims / provenance only).
- Its extraction log totals 652 per-fact entries across its 180 cells
  (108 entries over its 30 distinct origins, vs 2 for Qwen2.5-7B);
  the direct diagnostic is in the raw text: an unclosed ``<think>``
  block in 26-30 of 30 writes per style and a FACTS block in 0/30.
- Its summary text trails off inside unclosed `<think>` reasoning at
  the 256-token generation cap and never emits a parseable FACTS block.
  The reasoning itself heads toward false values (e.g. a fabricated
  tracking id, a wrong charge amount) that simply never reach the
  store — so the classifier sees an empty store and scores formation 0.
- M4 is the exception: its Qwen3 writes are non-empty (built from the
  claim fallback, then passed through the state-verified policy) and
  also score 0 — that cell means what it says.

**Consequence.** Week-3 scale-up is *not* cleared as a 2-model run.
The Qwen3 write generation must be fixed first (raise the token cap
and/or strip `<think>` content before the FACTS block, then re-test
the parse rate) and that arm re-piloted — or the arm dropped/replaced.
Scaling as-is would spend GPU on a broken arm. The Qwen2.5-7B arm is
unaffected: its formation signal is meaningfully nonzero (M1 = 1.000
under each of empty_partial / silent_wrong_value / stale_value at the
per-model level), so the pre-registered formation≈0 falsification
branch does not apply to it.

### Origin context

Across the 60 distinct origins (2 models × 6 chains × 5 faults):
origin task success 38/60; the origin claim was truthful in only 10/60
(no_fault: 7/12). Formation is measured on what the *policies* stored
from those origins, per the tables above.

## Earlier evidence (machinery, not model results)

Weeks 1–2 harness and policy evidence from scripted agents — the Week 1
exit criterion and the Week 2 policy properties — is documented in the
README ("Week 1 exit criterion — evidence", "Week 2 policy evidence")
and enforced by the test suite (1,164 tests).
