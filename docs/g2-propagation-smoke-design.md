# Gate G2 — Propagation smoke design

Date: 2026-11-05. Status: **harness built, smoke not run.** Gate text
(amd. §C): "Sessions 2–4 (dependents + correction event) run
end-to-end on ≥2 chains × 2 faults with live models, proving H2/H4
are measurable before the full grid spends GPU."

G1 was accepted Nov 5 with the coverage caveat (77.8%); the formation
arm of the two-model scale-up is cleared, but the paper's headline
(H2: corrupted M1 memory is worse than no memory downstream) still
depends on this gate.

## What was built for G2

`LLMDependentAgent` (`persistbench/llm_agent.py`), wired into
`runner.run_chain(..., dependent_agent="live", dependent_backend=...)`.
The scripted path stays the default; scorer semantics are untouched.

Semantics (deliberate, matching the scripted agent where it matters):

- **Memory-only reading.** The model sees the policy-written fact
  dict rendered into the prompt — never the world DB. Annotation
  keys (`_provenance`, `_claims`, …) are never rendered.
- **Read-tool fallback only when the memory lacks the fact.** M0
  (no memory) and partial stores may call `get_order` /
  `get_product` / `get_shipment` — exactly the fallback the scripted
  agent performs by reading ground truth directly. When the memory
  records the fact, no tools are offered and a tool call is refused
  with an error observation. Write tools are never offered; any
  write-tool call is rejected before it reaches the world.
- **One generation per task.** `run()` resolves the believed value
  and builds the outcome (including `"believed"`) in a single pass,
  so `run_dependent` does not generate twice. Transcripts,
  invalid-reply counts, parse logs, and token usage ride on the
  outcome for audit.
- **Correction (Session 4 only).** A `revising` agent is shown the
  verified observation as overriding the stored memory for the
  primary fact; a `stubborn` agent is never shown it, so it keeps
  the memory belief — matching the scripted stubborn semantics (a
  live model cannot be instructed to ignore text it has already
  seen). The correction is never delivered before Session 4.
- **Conservative parsing.** Replies must be one JSON object,
  `{"answer": <value>}` or a read-tool call; values are coerced
  against the fact's intended type. Unparseable replies are fed back
  as errors (bounded turns); an abstention (JSON null) or an
  exhausted turn budget yields `believed=None`, which the
  programmatic scorer grades incorrect. A missing answer is missing
  data, never a guess.
- **Generation budget.** Dependent answers use the action budget
  (256 tokens), lifted to the Qwen3 1024-token floor for Qwen3 only
  — the same rule as the origin agent.

## Exact smoke config

| Dimension | Value |
|---|---|
| Models | Qwen2.5-7B-Instruct, Qwen3-8B (the two G1-cleared arms) |
| Chains | chain01 (tracking; return-label ACTION dependent; 2-step plan), chain05 (stock; DECISION dependent; non-order chain, 1-step plan) |
| Faults | silent_wrong_value, stale_value (the two evidence-invisible faults — the ones M3 cannot catch; the propagation headline) |
| Policies | M0 (no-memory floor), M1 (treatment), M5 (oracle ceiling) |
| Dependent mode | revising (stubborn is a scripted counterfactual, not a live behaviour) |
| Seed | 0 only (matches the pilot; the main run uses ≥3 seeds per amd. §D.8) |
| Origin | LLMAgent, pilot generation settings (actions 256, writes 512, Qwen3 floor 1024, thinking off) |

Cells: 2 models × 2 chains × 2 faults × 3 policies = **24 chain
runs**, each with 3 dependent sessions = **72 dependent outcomes**.

**Reuse, don't regenerate.** Every (model, chain, fault, policy)
cell in this grid already exists as an origin trajectory + stored
write in the pilot outputs (`outputs/pilot_writes.jsonl` for
Qwen2.5-7B; `outputs-qwen3-repilot/` for Qwen3-8B, seed 0). The
smoke runner loads those stores and generates **only the 72
dependent answers live** — minutes on a T4, not a second pilot.
If reuse proves impractical (e.g. a store format mismatch), full
regeneration is 24 origin runs + writes, still a small fraction of
the pilot's 3h13m. Either way the smoke spends no GPU on formation.

## What PASS / FAIL means

G2 is a **measurability smoke, not an effect gate** — n=2 chains is
too small for inference, so effect sizes are reported, not gated.
PASS requires all of:

1. **Answer yield ≥80%.** Share of the 72 dependent tasks that
   produce a parseable believed value (abstentions and turn-budget
   exhaustions count as missing, beside every rate per amd. §F.12).
2. **M0 floor holds.** Live M0 dependents correct on all eligible
   cells — the read-tool path works and the model can extract a fact
   from a tool observation.
3. **M5 ceiling holds.** Live M5 dependents correct on all eligible
   cells — the memory-reading path works.
4. **No cell crashes or timeouts**, and the correction is verified
   present in the transcript of every Session-4 cell.
5. **A propagation contrast is computable** on the eligible subset
   (see below) — i.e. at least one eligible M1 cell per chain/fault
   with an M0 and M5 baseline to compare against.

FAIL if any of 1–5 fails. A clean null — M1 downstream ≈ M0
downstream with the floor and ceiling holding — is a **reportable
smoke outcome, not a FAIL**, provided the eligible n supports the
comparison; the paper then reports the null honestly (amd. §G).

Reported (not gated), with Wilson 95% CIs: propagation rate
(M1 incorrect downstream vs M0), persistence (Session 2 → 3),
correction resistance (Session 4 still incorrect after the
verified observation), all on the **G6-eligible subset** — origins
where the M0 and M5 baselines both succeed downstream — with the
raw (unfiltered) numbers alongside. Chain is the independent unit;
no per-fault claim is made below n=12 chains (amd. §F.16).

## H4 decision (amd. §F.18) — recorded here

**H4 is narrowed.** The live dependent does **not** condition on
provenance: M2's annotation is stored metadata only, and no reader
variant is built for the smoke. Rationale: no current policy
*consumes* provenance downstream, so a provenance-conditioned
reader would be a new treatment needing its own amendment and
design, not a smoke tweak — and the smoke must first establish
whether the plain reader propagates at all. M2 is not in the smoke
policy set. If smoke transcripts show models engaging with
annotations (or the full grid later needs the mechanism), a
provenance-conditioned reader variant is proposed as a post-smoke
amendment. H4's wording must not promise a mechanism the policy
set doesn't implement.

## What remains for the actual Kaggle run

1. **Smoke runner script** (new, `scripts/`): load pilot/re-pilot
   stores for the 24 cells, drive `LLMDependentAgent` per spec,
   write `results/g2-smoke/` (cells JSONL/CSV, dependent transcripts
   + parse logs, summary MD). Reuse first; fall back to full
   regeneration only on a documented format mismatch.
2. **Config file** `configs/g2_smoke.json` (models, chains, faults,
   policies, seed 0) mirroring the table above.
3. **Kaggle notebook + RUNBOOK entry** for the smoke run
   (backend factory, GPU sequencing after any queued runs).
4. **Analysis**: answer yield, M0/M5 floor-ceiling checks,
   propagation / persistence / correction-resistance on the eligible
   subset with Wilson CIs, per-model only (amd. §A.1).
5. **G2 verdict record** (`docs/g2-smoke-results.md`) + an
   amendment §I entry recording the verdict and the H4 narrowing
   above. No scale-up spend beyond the smoke until G2 passes; G3–G7
   are untouched by this gate.
