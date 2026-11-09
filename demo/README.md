# PersistAIBench demo

`demo.gif` (top of the repo README) renders a live harness run made on
Nov 9, 2026. The full record is here:

| File | What it is |
|---|---|
| `contrast-output.txt` / `contrast-results.json` | The contrast run: scripted NaiveAgent, all 12 chains, `silent_wrong_value` + `stale_value` faults, three memory policies, every downstream belief scored against DB truth. Under both faults, **M1 (unverified write) stores the false memory in 12/12 chains** and downstream beliefs drop to **20/36 correct (55.6%)** / **27/36 (75.0%)**; **M4 (state-verified write) and M5 (oracle ceiling) store it in 0/12** and downstream beliefs stay **36/36 (100%)**. |
| `cli-cell-output.txt` | One scored cell through the shipped CLI (`persistaibench run --chain chain01 --policy M1 --fault silent_wrong_value --backend fake`). The fake backend is a deterministic plumbing smoke — its output is stamped `fake/…` so it can never be mistaken for a model result. |
| `run_demo.py` | Re-runs the contrast and rewrites `contrast-results.json`. |
| `transcript.txt` | Every command run for the demo, with full output and exit codes. |

Scope, stated plainly: this is the **deterministic scripted harness** —
the same chains, faults, policies and scorer as the live-model runs, with
scripted agents. The live-model propagation smoke (Gate G2b: 72/72,
M0 floor 24/24, M5 ceiling 24/24, Kaggle T4×2, Nov 7 2026) is recorded in
[`docs/g2b-smoke-results.md`](../docs/g2b-smoke-results.md); this demo is
its local analogue, not a re-run of it.
