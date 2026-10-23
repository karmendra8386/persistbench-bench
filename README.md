# PersistBench

**Cross-session propagation of silent agent failures through long-term memory — and the cost of verified memory writes.**

A deterministic benchmark for one unmeasured chain: a tool fails silently in Session 1 → the agent writes its own account of the outcome into long-term memory → different tasks in Sessions 2–4 act on that memory → whether a write-time guard stops the propagation. There is no attacker and no prompt injection; the failure mode is benign self-poisoning.

Paper: Karmendra Pandey, *"Fail-Plausible, Then Persistent"* (working title). Outline approved 2026-10-23; filed in Google Drive EB1 §03 (`persistbench-outline.pdf`, `persistbench-outline-record-2026-10-23.md`).

## Provenance — read this first

**Week 1 = harness and design only. There are no LLM results yet, and nothing in this repo is a finding.** What exists:

- `docs/design-freeze.md` — the frozen design: RQ1–RQ5 / H1–H5, environment, fault conditions, memory policies M0–M5 (interface frozen; M2–M4 build in Week 2), six metrics, paired statistics plan.
- `docs/preregistration-draft.md` — hypotheses, conditions, primary metrics, analysis plan, and the four pre-registered falsification checks.
- A working deterministic harness with scripted agents, validated by the test suite below.

Live-model runs (Qwen2.5-7B, Qwen3-8B, Llama-3.1-8B, Mistral-7B; ~3,600 sessions) are Weeks 3–4 of the plan. Any number quoted from this repo before then is a property of the scripted harness, not of any model.

## Layout

```
persistbench/
  world.py     deterministic retail world; DB ground truth; agent tools
  chains.py    the 12 task chains (Session 1 + 3 dependents + correction fact)
  faults.py    seeded fault injection: no_fault / explicit_error /
               empty_partial / silent_wrong_value / stale_value
  agents.py    scripted oracle + naive origin agents; dependent agent
  memory.py    memory policies M0–M5 (Week 1 interface freeze)
  scorer.py    programmatic grading vs DB end-state — no LLM judge
  runner.py    origin / dependent trajectory runners
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
.venv/bin/python -m pytest -q   # 329 passed, 10 skipped (skips: Session-2 task
                                # does not depend on that chain's corrupted fact)
```

Standard library only at run time; pytest is the sole dev dependency.

## Next (per the approved plan)

- **Week 2:** memory store + live policies M2–M4; pilot (2 models × 6 chains) measuring Session-1 formation rates only; classifier validation gate (≥90% agreement with hand labels on 50 writes). Main risk retired here.
- **Weeks 3–4:** full runs; correction analysis; first results memo.
- **Month 2:** LongMemEval + Mem0 transfer arms; write-up. Zenodo preprint week 8–10 after author review; NeurIPS 2027 Datasets & Benchmarks is the primary venue target.
