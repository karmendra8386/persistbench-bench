# PersistBench — Kaggle Runbook

## Week 2 formation pilot (`week2-pilot.ipynb`)

**Status: GO given Nov 2, 2026** (owner: "Yeah, go ahead"). The latent
paper's staged Kaggle runs completed Nov 1, clearing the GPU
sequencing. Verify the actual remaining quota before committing
(Settings → Quotas; 30 h/week free tier, T4 x2), exactly as for the
latent launches.

**What it runs:** `scripts/run_pilot.py` with `configs/pilot.json` —
2 models (Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-8B, 4-bit) × 6 chains
(chain01/02/03/04/05/08 — see `persistbench/pilot.py` for the selection
rationale) × 5 fault conditions × memory policies M0–M5. The origin
trajectory is generated once per (model, chain, fault) and shared by all
six policies; the primary output is the Session-1 **false-memory
formation rate** per fault × policy, scored programmatically
(`scorer.classify_write`) — no LLM judge.

### Prerequisite: repo publication — RESOLVED Nov 1, 2026

The PersistBench repo is published (public) at
https://github.com/karmendra8386/persistbench-bench. The notebook's
acquisition cell clones it by default (`PERSISTBENCH_REPO_URL` can
override). Fallback **(B)** still works: upload the repo folder as a
Kaggle dataset / direct upload so that
`/kaggle/working/persistbench/configs/pilot.json` exists when the
notebook starts.

The acquisition cell hard-fails with these instructions if neither is
in place — a failed-fast launch, by design, like the latent notebooks'
GPU guard.

### Launch procedure (mirrors the latent project's pattern)

1. Read the current GPU quota first; report it. Free tier only.
2. Import `kaggle/week2-pilot.ipynb`; satisfy the code prerequisite
   above (clone URL or upload).
3. Set **Accelerator = GPU T4 x2** and **Internet = ON** *before*
   committing — a version launched with Accelerator=None fails by
   design at the GPU-check cell.
4. **Save Version → Save & Run All (Commit)** as a new version, then
   confirm the version is actually **Running/Queued** (a click is not
   proof).
5. Cancel any duplicate version that auto-started on import, so quota
   is not double-spent.
6. Set a retrieval check for ~12 h after launch (Kaggle's session cap;
   outputs persist in the version Output even if the session is
   killed).

### Expected outputs (version Output / `/kaggle/working/pilot-results`)

`pilot_summary.md` (formation table) · `pilot_formation.csv` ·
`pilot_cells.csv` · `pilot_cells.jsonl` · `pilot_writes.jsonl` (raw
summaries + stored writes + DB truth) · `pilot_summary.json` ·
`handlabel_sample.csv` (50 writes for the validation gate) ·
`handlabel_instructions.md`.

### After retrieval

1. **Classifier gate (pre-registered):** hand-label the 50-write sample
   per the instructions file; automatic-vs-hand agreement must be
   **≥ 90%** before any formation number is quoted anywhere.
2. **Formation ≈ 0 branch:** if M1 formation is near zero even under
   silent/stale faults, pivot the fault mix toward silent/stale before
   scaling (pre-registration falsification check 1) — do not simply
   proceed to Weeks 3–4.
3. Update the repo README status and the Drive §03 record + EB1
   tracker with the pilot outcome, in the same pass.
