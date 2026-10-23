# PersistBench

**Cross-session propagation of silent agent failures through long-term memory — and the cost of verified memory writes.**

A deterministic benchmark for one unmeasured chain: a tool fails silently in Session 1 → the agent writes its own account of the outcome into long-term memory → different tasks in Sessions 2–4 act on that memory → whether a write-time guard stops the propagation. There is no attacker and no prompt injection; the failure mode is benign self-poisoning.

Paper: Karmendra Pandey, *"Fail-Plausible, Then Persistent"* (working title). Outline approved 2026-10-23; filed in Google Drive EB1 §03 (`persistbench-outline.pdf`, `persistbench-outline-record-2026-10-23.md`).

## Provenance — read this first

**Weeks 1–2 = harness, design, and policy machinery only. There are no LLM results yet, and nothing in this repo is a finding.** What exists:

- `docs/design-freeze.md` — the frozen design: RQ1–RQ5 / H1–H5, environment, fault conditions, memory policies M0–M5, six metrics, paired statistics plan.
- `docs/preregistration-draft.md` — hypotheses, conditions, primary metrics, analysis plan, and the four pre-registered falsification checks.
- A working deterministic harness with scripted agents, validated by the test suite below.
- **Week 2 (code half, complete):** memory policies **M0–M5 all implemented** under the frozen interface — M2 (self-summary + disclosure/provenance annotation), M3 (structured evidence-contract write), M4 (state-verified write with correction/rejection accounting) join Week 1's M0/M1/M5. Every policy records a per-write log (entries + cost placeholders) sufficient to compute the frozen metrics, and `runner.run_chain` executes a full chain under any policy × any fault. The Week 2 **pilot** (live models, formation rates) is **pending GPU sequencing** after the latent paper's runs — no GPU has been used for PersistBench yet.
- **Week 2 pilot execution package (complete, launch-gated):** `persistbench/llm_agent.py` (LLM origin agent behind a backend interface — deterministic fake backend for tests, HF transformers backend for real models; conservative JSON action parsing and claim extraction), real per-policy memory-write generation (M1 free-form summary, M2 under the disclosure instruction, M3 claims parsed into the evidence-contract format — unparseable claims are logged as `unknown`, never verified; M4 = model write through the state-verified policy), `persistbench/pilot.py` + `configs/pilot.json` + `scripts/run_pilot.py` (2 models × 6 chains × 5 faults × M0–M5, per-cell md/csv/jsonl outputs, and the 50-write hand-labelling sample export for the ≥90%-agreement classifier gate), and `kaggle/week2-pilot.ipynb` + `kaggle/RUNBOOK.md`. **One launch prerequisite is flagged in the RUNBOOK: this repo is local-only — it must be published to GitHub (or uploaded to Kaggle as files) before the notebook can acquire the code.** Plumbing was smoke-tested with a real tiny model from the local cache (Qwen2.5-0.5B-Instruct, CPU): see `results/smoke-llm-2026-10-23.md` — labelled SMOKE, not a result.

Two implementation notes that matter for reading the machinery (both follow the design freeze; flagged here so they are not mistaken for findings):

- **M3's evidence is tool observations, never DB state** (checking state is M4's definition). Under explicit-error and empty/partial faults the contract therefore stores no false claim as fact. Under silent-wrong-value and stale faults, the corrupted response *is* the supporting observation, so M3 marks the false claim `verified` — the contract's boundary, asserted in the tests, and exactly why H3 orders M3 below M4.
- **M4 corrects rather than merely rejects** where the DB holds a value (rejection is reserved for facts the DB does not hold and for unverifiable keys). Its write log accounts for every falsified field as corrected-or-rejected, and on no-fault writes it corrects/rejects nothing — pre-registration falsification check 3 (a guard that destroys legitimate utility is not viable) is tested directly.

Live-model runs (Qwen2.5-7B, Qwen3-8B, Llama-3.1-8B, Mistral-7B; ~3,600 sessions) are Weeks 3–4 of the plan. Any number quoted from this repo before then is a property of the scripted harness, not of any model.

## Layout

```
persistbench/
  world.py     deterministic retail world; DB ground truth; agent tools
  chains.py    the 12 task chains (Session 1 + 3 dependents + correction fact)
  faults.py    seeded fault injection: no_fault / explicit_error /
               empty_partial / silent_wrong_value / stale_value
  agents.py    scripted oracle + naive origin agents; dependent agent
  memory.py    memory policies M0–M5 (all implemented, Week 2) + write logs
  scorer.py    programmatic grading vs DB end-state — no LLM judge
  runner.py    origin / dependent trajectory runners
  llm_agent.py LLM origin agent + per-policy write generation (Week 2
               pilot prep; fake + HF transformers backends)
  pilot.py     pilot config validation, grid runner, formation tables,
               hand-labelling sample export + agreement gate
configs/       pilot.json — the frozen pilot configuration
scripts/       run_pilot.py — pilot entry point (launch-gated)
kaggle/        week2-pilot.ipynb + RUNBOOK.md (launch procedure)
results/       smoke evidence (labelled); pilot outputs land here
tests/         pytest suite, incl. the Week 1 exit criterion
docs/          design freeze + pre-registration draft
```

## Week 1 exit criterion — evidence

> The harness grades hand-written/scripted trajectories correctly in all 5 fault conditions.

Verified by `tests/test_exit_criterion.py` (plus `test_world.py`, `test_dependents.py`):

- Oracle agent under no-fault: task success + truthful claim on **12/12 chains**.
- Naive (unverifying) agent: scorer detects a falsified claim under **each of the 4 injected faults × 12 chains (48/48)**, and names the falsified fields.
- Oracle agent: claim **never falsified under any fault (60/60 chain × fault cases)**.
- The dissociation the paper rests on is mechanically demonstrated: silent/stale faults leave the DB correct but the naive claim false (24/24); explicit/empty faults fail both the task and the claim (24/24).
- Propagation machinery: an M1 write of the naive claim is classified false on the corrupted fact (12/12); the dependent task that needs that fact then fails under M1 memory and succeeds under M5 (oracle) and M0 (query the world); the pre-Session-4 correction event repairs a revising agent and not a stubborn one (12/12).
- Hand-written false trajectory (claims success for work never executed) is detected, including the `task_success` falsification.

Run it:

```bash
python3 -m venv .venv && .venv/bin/pip install pytest
.venv/bin/python -m pytest -q   # 1158 passed, 10 skipped (skips: Session-2 task
                                # does not depend on that chain's corrupted fact)
```

Standard library only at run time; pytest is the sole dev dependency.

## Week 2 policy evidence (machinery, not results)

Verified by `tests/test_week2_policies.py` (811 cases) over all 12 chains × 5 fault conditions, with scripted agents:

- **M4:** the stored write is never falsified, under any fault, for either scripted agent; its content matches the M5 oracle store; its log's corrected+rejected entries are exactly the scorer's falsified fields; on no-fault writes it accepts everything and its store equals M1's.
- **M3:** under explicit-error / empty-partial faults the whole write classifies clean (success claims stored `failed` / `unknown`, never verified); under no fault every claim is verified and downstream behaviour matches M1. Under silent/stale faults the false claim is stored `verified` on the corrupted observation's evidence — the documented boundary above.
- **M2:** the claim is stored verbatim with provenance on every entry (`verified_against_state: False` throughout); observed tool errors appear in the disclosures, and a success claim past an observed error is annotated `contradicted_by_observation`. Downstream behaviour matches M1 exactly — the annotation, not filtering, is M2's guard.
- **Correction:** where a guard kept the false memory out (M3/M4 under evidence-visible faults; M4 under all faults), even a stubborn agent's Session 4 is correct. Where the lie reached the store (M2/M3 under silent/stale), the Week-1 pattern holds: stubborn resists, revising is repaired.

## Next (per the approved plan)

- **Week 2 pilot (package ready, pending GPU sequencing):** 2 models × 6 chains × 5 faults × M0–M5 measuring Session-1 formation rates; classifier validation gate (≥90% agreement with hand labels on the exported 50-write sample). Main risk retired here. Launch per `kaggle/RUNBOOK.md`, sequenced after the latent paper's GPU runs — and note the RUNBOOK's repo-publication prerequisite (this repo is local-only).
- **Weeks 3–4:** full runs; correction analysis; first results memo.
- **Month 2:** LongMemEval + Mem0 transfer arms; write-up. Zenodo preprint week 8–10 after author review; NeurIPS 2027 Datasets & Benchmarks is the primary venue target.
