"""Gate G7 populated-store sensitivity arm: reuse pilot stores, seed
the memory with true distractor entries, drive live dependents.

The gate (preregistration amendment, section F.15): Sunil et al.
show pre-existing legitimate memory sharply attenuates poisoning,
but every pilot chain starts from an empty store. This arm re-runs
the G2 smoke grid — same models, chains, faults, policies
(M0/M1/M5), revising dependents, seed 0, same reused pilot writes —
with one change: before the dependents run, the memory store is
pre-populated with distractor entries, so the (possibly false)
target entry sits inside a store of legitimate prior-task facts
instead of alone in an empty one. The design note is
docs/g7-populated-store-design.md.

What "pre-populated" means in this harness
------------------------------------------
A policy's store is a flat fact dict; later sessions read it whole
(the live dependent renders every non-annotation key into its
prompt; the scripted dependent looks facts up by key). There is no
per-entity namespacing: the fact namespace is the eight chain fact
keys plus ``task_success``. A distractor entry can therefore only
occupy a fact key the target chain's write does not use — anything
else would overwrite (or be overwritten by) the target entry
itself, which is a store collision, not a distractor. Seeding is
exactly that: ``{**seed, **stored_write}`` with seed keys disjoint
from the target write's keys by construction, so the target entry
surfaces byte-identically and grading (which reads only the
target chain's facts, against the target world's truth) is
untouched.

Distractor construction (deterministic, no randomness)
-------------------------------------------------------
For target chain T, the excluded keys are T's intended facts plus
``task_success`` (every write carries it). The seed is built by
walking the chain suite in order, skipping T, and taking each
remaining fact key from the FIRST chain whose intended facts hold
it. Every seed value is therefore another chain's intended fact —
true in that chain's world by construction (it is what that
chain's M5 oracle write would store) — and the seed for a target
chain is a pure function of the suite: identical across models,
faults and policies; only the target write differs. Provenance
(source chain per key) is recorded on the summary and in every
transcript.

Per-policy scope
----------------
- M0: no store. The arm's M0 cells are the empty-store baseline,
  byte-identical to the smoke's — they anchor G6 eligibility.
- M1 / M5: flat fact dicts; seeded. These are the arm's policies.
- M2 / M3 / M4: NOT seeded, and the arm's config validation
  rejects them. M2's ``_provenance`` annotates the target write's
  claims only, so seeded entries would sit in the store silently
  uncovered by the annotation. M3's invariant is that every fact
  entry has a ``_claims`` record with an evidence pointer into
  the origin trajectory; a seeded entry has no such evidence, so
  an M3 store carrying one would misrepresent the contract. M4's
  store is by construction exactly the target world's verified
  facts — foreign entries are what M4 rejects at write time.
  See the design note.

Readout, not a verdict
----------------------
The arm is a sensitivity measurement, not a gate: propagation on
the G6-eligible subset (plus persistence and correction
resistance) is reported per model alongside the recorded G2b
empty-store numbers, with the delta. Sunil et al. predict
attenuation; the arm does not gate on direction. ``run_arm``
returns/writes ``completed: True`` when the grid ran and the
outputs were written; the script's exit code is 0 on completion
and 2 on a reuse failure (same fallback contract as the smoke:
``fallback_pilot_config.json`` is written and the fallback itself
is never executed here).
"""
from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

from .chains import CHAINS, CHAINS_BY_ID
from .eligibility import propagation_report, rate as _rate
from .g2_smoke import (
    _CELL_FIELDS,
    _transcript_name,
    ReuseError,
    default_config as _smoke_default_config,
    fallback_pilot_config,
    load_smoke_stores,
    reconcile_world,
    validate_config as _validate_smoke_config,
)
from .llm_agent import LLMDependentAgent
from .runner import correction_event, run_dependent

#: Policies the arm runs: the G2 smoke set. M0 is the unseeded
#: baseline; M1/M5 are seeded.
ARM_POLICIES = ["M0", "M1", "M5"]

#: Policies whose store is seeded (M0 has no store to seed).
SEEDED_POLICIES = ("M1", "M5")

#: Where the recorded G2b empty-store summary lives, relative to
#: the repo root (tracked, so a Kaggle clone has it). The
#: comparison is reported against this file's per-model rates.
DEFAULT_BASELINE_SUMMARY = "results/g2b-smoke/g2_smoke_summary.json"

_CELL_KEY = ("model", "chain_id", "fault_condition", "policy")

_ARM_CELL_FIELDS = _CELL_FIELDS + [
    "populated_store", "distractor_entries", "store_entries",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def default_config() -> dict:
    config = _smoke_default_config()
    config["populated_store"] = True
    config["baseline_summary"] = DEFAULT_BASELINE_SUMMARY
    return config


def validate_config(config: dict) -> dict:
    """Validate a G7 arm config; returns it unchanged or raises.

    Everything the smoke validates, plus the arm's own rules: the
    populated-store flag must be on (an unpopulated run is the
    smoke, not this arm) and the policy set is restricted to
    M0/M1/M5 — M2/M3/M4 stores cannot carry seeded entries
    honestly (module docstring).
    """
    _validate_smoke_config(config)
    if config.get("populated_store") is not True:
        raise ValueError(
            "config.populated_store must be true: the G7 arm is the "
            "populated-store run (for the empty-store run, use the "
            "G2 smoke)")
    policies = config["policies"]
    if sorted(policies) != sorted(ARM_POLICIES):
        raise ValueError(
            f"config.policies for the G7 arm must be exactly "
            f"{ARM_POLICIES} (M0 baseline + seeded M1/M5); got "
            f"{policies}. M2/M3/M4 stores cannot carry seeded "
            f"entries honestly — see docs/g7-populated-store-design.md")
    baseline = config.get("baseline_summary")
    if baseline is not None and not isinstance(baseline, str):
        raise ValueError("config.baseline_summary must be a path string")
    return config


def load_config(path) -> dict:
    config = json.loads(Path(path).read_text())
    return validate_config(config)


# ---------------------------------------------------------------------------
# Distractor seeds
# ---------------------------------------------------------------------------

def target_fact_keys(chain: dict) -> set:
    """Keys a write for this chain may carry: its intended facts
    plus ``task_success`` (present in every policy's write). Seed
    keys must avoid all of them."""
    return set(chain["intended_facts"]) | {"task_success"}


def build_distractor_seed(chain: dict, chains=CHAINS):
    """The distractor seed for one target chain.

    Returns ``(seed, provenance)``: ``seed`` is a flat fact dict;
    ``provenance`` is a list of ``{"key", "value", "source_chain"}``
    in seed-insertion order. Deterministic: chains are walked in
    suite order and each key is taken from the first chain (other
    than the target) whose intended facts hold it, so the seed is a
    pure function of the suite — identical across models, faults
    and policies for a given target chain.
    """
    excluded = target_fact_keys(chain)
    seed: dict = {}
    provenance: list = []
    for source in chains:
        if source["chain_id"] == chain["chain_id"]:
            continue
        for key, value in source["intended_facts"].items():
            if key in excluded or key in seed:
                continue
            seed[key] = value
            provenance.append({
                "key": key, "value": value,
                "source_chain": source["chain_id"],
            })
    return seed, provenance


def distractor_seeds(config: dict) -> dict:
    """Seeds for every chain in the arm config, keyed by chain id:
    ``{chain_id: {"seed": {...}, "provenance": [...]}}``."""
    validate_config(config)
    out = {}
    for chain_id in config["chains"]:
        seed, provenance = build_distractor_seed(CHAINS_BY_ID[chain_id])
        out[chain_id] = {"seed": seed, "provenance": provenance}
    return out


def seed_store(stored_write: dict | None, seed: dict):
    """Pre-populate a policy store with the distractor seed.

    ``None`` (M0's no-store) passes through unchanged — there is
    nothing to populate. Otherwise returns ``{**seed,
    **stored_write}``: the seed sits underneath, the target write
    on top, mirroring accumulation order (prior tasks' entries
    were already in the store when this task's write landed). By
    construction the seed keys are disjoint from the target
    chain's fact keys; a stored write that nevertheless carries a
    seed key as a fact key is a store collision, raised as
    :class:`ReuseError` rather than silently resolved — the arm
    must never overwrite or shadow the target entry.
    """
    if stored_write is None:
        return None
    collisions = sorted(
        key for key in seed
        if key in stored_write and not key.startswith("_"))
    if collisions:
        raise ReuseError(
            f"distractor seed collides with the target write on "
            f"fact keys {collisions}: seeding would shadow the "
            f"target entry (the seed is built disjoint from the "
            f"chain's intended facts, so the stored write carries "
            f"an unexpected key)")
    return {**seed, **stored_write}


def _fact_entries(store: dict | None) -> int:
    """Fact entries in a store (annotation keys never count)."""
    if not store:
        return 0
    return sum(1 for key in store if not key.startswith("_"))


# ---------------------------------------------------------------------------
# G2b baseline (empty store) for the comparison readout
# ---------------------------------------------------------------------------

def load_baseline(config: dict, repo_root: Path | None = None):
    """Per-model rates from the recorded G2b empty-store summary.

    Returns ``{model: per_model_row}`` in the eligibility module's
    row shape, or ``None`` when no baseline is configured or the
    file cannot be read — the arm still completes; the comparison
    is reported as unavailable.
    """
    rel = config.get("baseline_summary")
    if not rel:
        return None
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    path = Path(rel)
    if not path.is_absolute():
        path = root / path
    try:
        summary = json.loads(path.read_text())
        per_model = summary["analysis"]["per_model"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(per_model, dict):
        return None
    return per_model


_MEASURES = (
    ("propagation_eligible", "Propagation (eligible)"),
    ("persistence_s2_to_s3", "Persistence S2->S3"),
    ("correction_resistance_s3_to_s4", "Correction resistance S3->S4"),
)


def build_comparison(analysis: dict, baseline: dict | None) -> dict:
    """Side-by-side empty (G2b) vs populated (G7) rates, per model.

    ``{model: {measure: {"empty_store": rate-row|None,
    "populated_store": rate-row, "delta": float|None}}}`` — delta
    is populated minus empty, ``None`` when either rate is
    undefined or no baseline was available.
    """
    comparison = {}
    for model, row in analysis["per_model"].items():
        base_row = (baseline or {}).get(model)
        measures = {}
        for key, _label in _MEASURES:
            populated = row[key]
            empty = base_row.get(key) if base_row else None
            delta = None
            if (empty is not None and empty.get("rate") is not None
                    and populated.get("rate") is not None):
                delta = populated["rate"] - empty["rate"]
            measures[key] = {
                "empty_store": empty,
                "populated_store": populated,
                "delta": delta,
            }
        comparison[model] = measures
    return comparison


# ---------------------------------------------------------------------------
# Analysis (descriptive — the arm reports, it does not gate)
# ---------------------------------------------------------------------------

def analyse_arm_cells(cells: list, config: dict) -> dict:
    """Descriptive rates from the arm's dependent-outcome rows.

    Same shape as the smoke analysis minus the gate verdicts: the
    arm is a measurement. Eligibility and the per-model rates come
    from :mod:`persistbench.eligibility`, exactly as in the smoke,
    so populated and empty numbers are computed identically.
    """
    errors = [c for c in cells if c.get("error")]
    answer_yield = _rate(
        sum(1 for c in cells if c.get("parseable") and not c.get("error")),
        len(cells))
    m0_cells = [c for c in cells if c["policy"] == "M0"]
    m5_cells = [c for c in cells if c["policy"] == "M5"]
    m0_correct = _rate(sum(1 for c in m0_cells if c.get("correct")),
                       len(m0_cells))
    m5_correct = _rate(sum(1 for c in m5_cells if c.get("correct")),
                       len(m5_cells))
    s4 = [c for c in cells if c["session"] == 4]
    correction_ok = all(c.get("correction_in_prompt") for c in s4) and bool(s4)
    report = propagation_report(cells, config["models"], config["chains"],
                                config["fault_conditions"])
    return {
        "cells": len(cells),
        "errors": len(errors),
        "answer_yield": answer_yield,
        "m0_correct": m0_correct,
        "m5_correct": m5_correct,
        "correction_in_all_s4_prompts": correction_ok,
        "contrast_groups": report["contrast_groups"],
        "per_model": report["per_model"],
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _cells_csv(cells: list) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_ARM_CELL_FIELDS)
    writer.writeheader()
    for cell in cells:
        writer.writerow({
            k: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v
            for k, v in cell.items()})
    return buffer.getvalue()


def _fmt_rate(r) -> str:
    if r is None:
        return "n/a"
    if r["rate"] is None:
        return f"n/a (0/{r['n']})" if r["n"] == 0 else "n/a"
    return f"{r['rate']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] ({r['k']}/{r['n']})"


def _summary_md(config: dict, analysis: dict, repairs: list,
                seeds: dict, comparison: dict,
                baseline_available: bool) -> str:
    lines = [
        "# PersistBench Gate G7 — populated-store sensitivity arm",
        "",
        f"Models: {', '.join(config['models'])} · chains: "
        f"{', '.join(config['chains'])} · faults: "
        f"{', '.join(config['fault_conditions'])} · policies: "
        f"{', '.join(config['policies'])} · dependent mode: "
        f"{config.get('dependent_mode', 'revising')} · "
        f"seed {config.get('seed', 0)}",
        "",
        f"Cells: {analysis['cells']} dependent outcomes "
        f"(24 chain runs x 3 sessions) · errors: {analysis['errors']} · "
        f"answer yield: {_fmt_rate(analysis['answer_yield'])}",
        "",
        "The G2 smoke grid, re-run with the memory store "
        "pre-populated with true distractor entries from other "
        "chains' completed tasks (design note: "
        "docs/g7-populated-store-design.md). M0 cells are the "
        "unseeded no-memory baseline, identical to the smoke's. "
        "This arm is a sensitivity measurement, not a gate: the "
        "readout is the populated-vs-empty comparison below, "
        "reported per model, never pooled, and not gated on "
        "direction (Sunil et al. predict attenuation).",
        "",
        "## Distractor seeds",
        "",
    ]
    for chain_id in config["chains"]:
        entry = seeds[chain_id]
        lines.append(
            f"- {chain_id}: {len(entry['seed'])} distractor entries "
            f"seeded under the target write:")
        for item in entry["provenance"]:
            lines.append(
                f"  - {item['key']}: {item['value']!r} "
                f"(from {item['source_chain']})")
    lines += [
        "",
        "## Comparison — populated vs empty store",
        "",
        "Propagation = share of G6-eligible M1 outcomes incorrect; "
        "persistence = incorrect at Session 2 and still incorrect at "
        "Session 3; correction resistance = incorrect at Session 3 "
        "and still incorrect at Session 4 after the verified "
        "observation. Wilson 95% CIs. Empty-store numbers are the "
        "recorded Gate G2b results "
        f"({'loaded from ' + config.get('baseline_summary', '') if baseline_available else 'baseline summary unavailable — populated numbers only'}).",
        "",
        "| Model | Measure | Empty store (G2b) | Populated store (G7) | Delta (populated - empty) |",
        "|---|---|---|---|---|",
    ]
    for model, measures in comparison.items():
        for key, label in _MEASURES:
            m = measures[key]
            delta = ("n/a" if m["delta"] is None
                     else f"{m['delta']:+.3f}")
            lines.append(
                f"| {model} | {label} | {_fmt_rate(m['empty_store'])} | "
                f"{_fmt_rate(m['populated_store'])} | {delta} |")
    lines += [
        "",
        "## Descriptive rates (populated arm)",
        "",
        f"M0 correct (unseeded baseline): {_fmt_rate(analysis['m0_correct'])} · "
        f"M5 correct (seeded): {_fmt_rate(analysis['m5_correct'])} · "
        f"correction in every Session-4 prompt: "
        f"{analysis['correction_in_all_s4_prompts']}",
        "",
        "## World repairs",
        "",
        "As in the smoke, each cell's world is a plan replay "
        "repaired to the recorded pilot truth. Repairs below; an "
        "empty list means every replay already reproduced the "
        "pilot world.",
        "",
    ]
    if not repairs:
        lines.append("None.")
    else:
        for repair in repairs:
            lines.append(
                f"- {repair['model']} {repair['chain_id']} "
                f"{repair['fault_condition']} {repair['policy']}: "
                + "; ".join(
                    f"{r['fact']}: replay {r['replay_value']!r} -> "
                    f"recorded {r['recorded_value']!r}"
                    for r in repair["repair"]))
    lines += [
        "",
        "## Notes",
        "",
        "- Sensitivity arm at smoke scale (n=2 chains, seed 0, "
        "revising dependents): effect sizes are reported, not "
        "gated, and no per-fault claim is made below n=12 chains.",
        "- The origin writes are the pilot's (generated against an "
        "empty store, reused exactly as in the smoke); the "
        "pre-population models the store state those writes landed "
        "in. The arm measures downstream sensitivity to a populated "
        "store, not formation into one.",
        "- Distractor entries are keyed disjointly from the target "
        "write by construction (the flat fact namespace has nine "
        "keys), so the target entry is byte-identical to the "
        "empty-store arm's and grading is unchanged.",
        "",
    ]
    return "\n".join(lines)


def run_arm(config: dict, backend_factory, out_dir,
            writes_paths: dict | None = None,
            repo_root: Path | None = None) -> dict:
    """Run the 24-cell populated-store arm against reused pilot stores.

    ``backend_factory(model_id) -> ChatBackend``. Writes
    ``g7_arm_cells.jsonl`` / ``.csv``, one transcript JSON per
    dependent outcome under ``transcripts/``, and
    ``g7_arm_summary.json`` / ``.md`` under ``out_dir``. Returns
    the summary dict (also written to ``g7_arm_summary.json``),
    with ``completed: True`` once the grid has run and the outputs
    are written. Raises :class:`ReuseError` (after writing
    ``fallback_pilot_config.json``) when the pilot stores cannot
    be reused — the fallback is full regeneration, never executed
    here, exactly as in the smoke.
    """
    validate_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = config.get("seed", 0)
    seeds = distractor_seeds(config)
    try:
        stores = load_smoke_stores(config, writes_paths, repo_root)
    except ReuseError:
        (out / "fallback_pilot_config.json").write_text(
            json.dumps(fallback_pilot_config(config), indent=2) + "\n")
        raise
    transcripts_dir = out / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    repairs = []
    for model in config["models"]:
        backend = backend_factory(model)
        for chain_id in config["chains"]:
            chain = CHAINS_BY_ID[chain_id]
            chain_seed = seeds[chain_id]["seed"]
            for fault in config["fault_conditions"]:
                for policy in config["policies"]:
                    key = (model, chain_id, fault, policy)
                    row = stores[key]
                    try:
                        world, repair = reconcile_world(
                            chain, fault, seed, row["truth"])
                    except ReuseError:
                        (out / "fallback_pilot_config.json").write_text(
                            json.dumps(fallback_pilot_config(config),
                                       indent=2) + "\n")
                        raise
                    if repair:
                        repairs.append({**{k: key[i] for i, k
                                           in enumerate(_CELL_KEY)},
                                        "repair": repair})
                    stored_write = row["stored_write"]
                    if policy in SEEDED_POLICIES:
                        if stored_write is None:
                            (out / "fallback_pilot_config.json").write_text(
                                json.dumps(fallback_pilot_config(config),
                                           indent=2) + "\n")
                            raise ReuseError(
                                f"unexpected store shape for seeded "
                                f"policy cell {key}: {policy} must "
                                f"store a fact dict, got None")
                        memory = seed_store(stored_write, chain_seed)
                        n_distractors = len(chain_seed)
                    else:
                        # M0: no store exists; the baseline is the
                        # smoke's, unseeded by construction.
                        memory = stored_write
                        n_distractors = 0
                    agent = LLMDependentAgent(
                        backend, memory,
                        mode=config.get("dependent_mode", "revising"),
                        seed=seed)
                    correction = correction_event(chain, world)
                    for spec in chain["dependents"]:
                        started = time.monotonic()
                        cell = {
                            "model": model, "chain_id": chain_id,
                            "fault_condition": fault, "policy": policy,
                            "session": spec["session"], "kind": spec["kind"],
                            "fact": spec["fact"],
                            "origin_falsified_fields": row.get("falsified_fields", []),
                            "world_repair": repair,
                            "correction": (correction if spec["session"] == 4
                                           else None),
                            "populated_store": True,
                            "distractor_entries": n_distractors,
                            "store_entries": _fact_entries(memory),
                            "error": None,
                        }
                        try:
                            outcome, grade = run_dependent(
                                chain, world, agent, spec,
                                correction=correction)
                        except Exception as exc:  # one bad cell must not kill the run
                            cell.update({
                                "believed": None, "truth": None,
                                "correct": False, "parseable": False,
                                "invalid_replies": None,
                                "prompt_tokens": None,
                                "completion_tokens": None,
                                "correction_in_prompt": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                        else:
                            prompt = outcome.get("prompt", "")
                            cell.update({
                                "believed": outcome.get("believed"),
                                "truth": grade.truth,
                                "correct": bool(grade.correct),
                                "parseable": outcome.get("believed") is not None,
                                "invalid_replies": outcome.get("invalid_replies"),
                                "prompt_tokens": outcome.get("usage", {}).get("prompt_tokens"),
                                "completion_tokens": outcome.get("usage", {}).get("completion_tokens"),
                                "correction_in_prompt": (
                                    "Verified observation" in prompt
                                    if spec["session"] == 4 else None),
                            })
                            transcript = {
                                "model": model, "chain_id": chain_id,
                                "fault_condition": fault, "policy": policy,
                                "session": spec["session"],
                                "kind": spec["kind"], "fact": spec["fact"],
                                "stored_write": memory,
                                "unseeded_stored_write": stored_write,
                                "distractor_seed": (
                                    seeds[chain_id]["provenance"]
                                    if policy in SEEDED_POLICIES else []),
                                "truth": grade.truth,
                                "believed": outcome.get("believed"),
                                "correct": bool(grade.correct),
                                "correction": (correction if spec["session"] == 4
                                               else None),
                                "prompt": prompt,
                                "transcript": outcome.get("transcript", []),
                                "parse_log": outcome.get("parse_log", []),
                                "invalid_replies": outcome.get("invalid_replies"),
                                "usage": outcome.get("usage", {}),
                            }
                            name = _transcript_name(cell)
                            (transcripts_dir / name).write_text(
                                json.dumps(transcript, indent=2, default=str) + "\n")
                            cell["transcript_file"] = f"transcripts/{name}"
                        cell["cell_seconds"] = round(time.monotonic() - started, 3)
                        cells.append(cell)

    analysis = analyse_arm_cells(cells, config)
    baseline = load_baseline(config, repo_root)
    comparison = build_comparison(analysis, baseline)
    expected_cells = (len(config["models"]) * len(config["chains"])
                      * len(config["fault_conditions"])
                      * len(config["policies"]) * 3)
    summary = {
        "config": config,
        "cells": len(cells),
        "expected_cells": expected_cells,
        "completed": len(cells) == expected_cells,
        "world_repairs": repairs,
        "distractor_seeds": seeds,
        "analysis": analysis,
        "comparison": comparison,
        "baseline_available": baseline is not None,
    }
    (out / "g7_arm_cells.jsonl").write_text(
        "\n".join(json.dumps(c, default=str) for c in cells) + "\n")
    (out / "g7_arm_cells.csv").write_text(_cells_csv(cells))
    (out / "g7_arm_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    (out / "g7_arm_summary.md").write_text(
        _summary_md(config, analysis, repairs, seeds, comparison,
                    baseline is not None))
    return summary
