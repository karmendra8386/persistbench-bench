# PersistBench — Kaggle Runbook

## Qwen3 re-pilot — Gate G1 (CURRENT, Nov 4, 2026)

The Nov 2 pilot's Qwen3-8B arm was a measurement failure, not a model
result: its `<think>` reasoning filled the shared 256-token cap and was
cut off mid-reasoning, so 86/90 M1–M3 writes are unclosed-`<think>`
truncations with empty stores (FACTS block in 0/30 M1 writes at the
time; see `docs/pilot-audit-2026-11-02.md`). The fix, in
`persistbench/llm_agent.py`, is two layers, Qwen3-only:

1. Thinking disabled at the chat template for Qwen3
   (`enable_thinking=False`, already shipped Nov 2 — the primary fix).
2. **NEW Nov 4:** a 1024-token generation floor for Qwen3
   (`effective_max_new_tokens` / `QWEN3_MIN_NEW_TOKENS`): action calls
   and writes that would have used 256 / 512 now get 1024, so any
   reasoning that leaks past the template switch still leaves room for
   the FACTS block. Greedy decoding stops at EOS, so the floor costs
   nothing when thinking is off. Qwen2.5 behaviour is unchanged
   (256 actions / 512 writes) — verified by tests, and the full pilot
   config `configs/pilot.json` is untouched.

### Exact launch steps (parent agent, in a browser — do NOT fake this)

1. Read the current GPU quota first; report it. Free tier only
   (Settings → Quotas; 30 h/week free tier, T4 x2).
2. Import `kaggle/week2-pilot.ipynb` from the repo at the commit
   reported with this re-pilot (it must contain the Nov 4 fix —
   check the run cell uses `--config configs/g1_qwen3.json`).
   The notebook clones
   https://github.com/karmendra8386/persistbench-bench by default;
   fallback (B) in the original section below still applies.
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

### Expected grid and outputs

- **Grid: Qwen3 ONLY** — 1 model (Qwen/Qwen3-8B, 4-bit) × 6 chains
  (chain01/02/03/04/05/08) × 5 fault conditions (no_fault,
  explicit_error, empty_partial, silent_wrong_value, stale_value) ×
  6 policies (M0–M5) = **180 cells, seed 0**, config
  `configs/g1_qwen3.json` (`max_new_tokens: 1024`).
- Outputs (version Output / `/kaggle/working/pilot-results`): same
  file list as the original pilot below (`pilot_summary.md`,
  `pilot_formation.csv`, `pilot_cells.csv`, `pilot_cells.jsonl`,
  `pilot_writes.jsonl`, `pilot_summary.json`,
  `handlabel_sample.csv`, `handlabel_instructions.md`).

### Re-pilot success criterion (Gate G1)

- **Write yield first:** M1–M3 write coverage ≥ 80% AND a FACTS/CLAIM
  block present in ≥ 80% of writes (Nov 2 baseline: 0/30 M1 FACTS
  blocks, 89/90 empty M1–M3 stores).
- **Parse failures at Qwen2.5's order of magnitude:** Qwen3's
  parse-failure total drops from the pilot's 328 / 90 M1–M3 writes
  toward Qwen2.5's 55 / 90 (≈ 0.6 per write, not 3.6 per write) —
  tens, not hundreds.
- **M1–M3 writes non-empty in most cells** (the same ≥ 80% coverage
  bar, per policy, not just pooled).
- If G1 fails, the Qwen3 arm is **replaced** (candidate:
  Qwen2.5-14B-Instruct or Llama-3.1-8B with verified write coverage)
  — not silently dropped, and never pooled with the broken Nov 2
  Qwen3 numbers.
- On G1 pass, the re-pilot numbers **replace** the Nov 2 Qwen3 arm;
  the Qwen2.5 arm is not re-run.

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
