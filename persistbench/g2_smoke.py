"""Gate G2 propagation smoke: reuse pilot stores, drive live dependents.

The smoke proves H2/H4 are measurable before the full grid spends GPU
(docs/g2-propagation-smoke-design.md). It runs NO formation: the 24
origin trajectories + stored writes already exist in the pilot outputs,
so the smoke generates only the 72 dependent answers live (24 chain
runs x 3 dependent sessions).

Reuse, don't regenerate
------------------------
``load_smoke_stores`` reads the stored write for every
(model, chain, fault, policy) cell from the pilot writes files:
``outputs/pilot_writes.jsonl`` for Qwen2.5-7B-Instruct (Nov 2 pilot)
and ``results/qwen3-repilot/pilot_writes.jsonl`` for Qwen3-8B (Nov 4
re-pilot; the tracked copy is byte-identical to the retrieved
``outputs-qwen3-repilot/`` copy). Each cell must appear exactly once;
a missing or duplicated cell is a reuse failure (see below).

World reconstruction
--------------------
The pilot writes files record the stored write and the DB truth, but
not the origin trajectory's tool steps, so the post-origin world the
dependents read from is rebuilt by replaying the chain's plan through
the world with the fault injected at the critical step
(``replay_origin_world``). For silent/stale faults the DB effect is
applied correctly and only the response is corrupted, so a replay
reproduces the pilot world whenever the origin executed the plan
exactly once, in order.

It does not always do so. The Nov 2 Qwen2.5-7B origin, shown a
corrupted restock response (stock 4 / 0 instead of 40), issued a
second restock — 36 more units under silent_wrong_value, 40 more
under stale_value — genuinely changing the DB: recorded truth is
stock 76 / 80, replay gives 40. ``reconcile_world`` therefore
compares the replay truth against the recorded truth and patches the
DB fields underlying any mismatched intended fact to the recorded
value (``set_fact``, the write mirror of ``chains.resolve_fact``);
every patch is recorded on the cell as ``world_repair``. The recorded
truth is authoritative — grading, the read-tool fallback, and the
Session-4 correction all read the reconciled world, never the raw
replay. If a mismatch cannot be patched (e.g. a ``charged_cents``
truth with no payment in the DB), reuse of that cell fails.

Reuse failure and the fallback
------------------------------
A ``ReuseError`` means the pilot stores cannot be reused for the
smoke as they stand (missing/duplicated cell, unexpected store
shape, unrepairable world). ``run_smoke`` writes a fallback pilot
config (``fallback_pilot_config.json``) into the output directory and
re-raises; the fallback is full regeneration: run the 24 origin runs
+ writes with ``scripts/run_pilot.py`` on that config (2 models x
2 chains x 2 faults x 6 policies — the origin is generated once per
model/chain/fault, 24 generations), then re-run the smoke against
the fresh ``pilot_writes.jsonl``. The fallback is implemented and
documented here; it is never executed by this module.

Gates (design note; measurability smoke, not an effect gate)
------------------------------------------------------------
PASS requires all of: (1) answer yield >= 80% over the 72 dependent
outcomes; (2) M0 floor holds — every live M0 dependent correct;
(3) M5 ceiling holds — every live M5 dependent correct; (4) no cell
errors, and the correction is verified present in the prompt of
every Session-4 cell; (5) a propagation contrast is computable —
at least one G6-eligible session per (model, chain, fault) group.
A clean null (M1 downstream ~= M0 downstream with floor and ceiling
holding) is a reportable smoke outcome, not a FAIL.

G6 eligibility is per (model, chain, fault, session): the session is
eligible iff the M0 and M5 dependent outcomes for that same session
are both correct. Propagation / persistence / correction-resistance
are reported on the eligible subset with Wilson 95% CIs, per model
only (never pooled), with the raw (unfiltered) numbers alongside.
Chain is the independent unit; no per-fault claim below n=12 chains.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import time
from pathlib import Path

from .chains import CHAINS_BY_ID, build_world, resolve_fact
from .faults import FaultInjector
from .llm_agent import LLMDependentAgent
from .runner import correction_event, run_dependent

# The exact smoke grid (design note, "Exact smoke config" table).
G2_MODELS = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B"]
G2_CHAINS = ["chain01_pay_ship_laptop", "chain05_restock_lamp"]
G2_FAULTS = ["silent_wrong_value", "stale_value"]
G2_POLICIES = ["M0", "M1", "M5"]
G2_DEPENDENT_MODE = "revising"

# Short labels for file names (model ids contain "/" and ".").
_MODEL_SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen25",
    "Qwen/Qwen3-8B": "qwen3",
}

# Default pilot writes files, relative to the repo root. Both are
# tracked, so a Kaggle clone has them; the Qwen3 path is the tracked
# results/ copy (byte-identical to outputs-qwen3-repilot/).
DEFAULT_WRITES = {
    "Qwen/Qwen2.5-7B-Instruct": "outputs/pilot_writes.jsonl",
    "Qwen/Qwen3-8B": "results/qwen3-repilot/pilot_writes.jsonl",
}

_CELL_KEY = ("model", "chain_id", "fault_condition", "policy")


class ReuseError(Exception):
    """The pilot stores cannot be reused for the smoke as they stand.

    See the module docstring ("Reuse failure and the fallback"): the
    caller should regenerate the 24 origin runs + writes with
    ``scripts/run_pilot.py`` on the fallback config, then re-run.
    """


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def default_config() -> dict:
    return {
        "models": list(G2_MODELS),
        "chains": list(G2_CHAINS),
        "fault_conditions": list(G2_FAULTS),
        "policies": list(G2_POLICIES),
        "dependent_mode": G2_DEPENDENT_MODE,
        "seed": 0,
        "max_new_tokens": 256,
        "load_in_4bit": True,
        "writes": dict(DEFAULT_WRITES),
    }


def validate_config(config: dict) -> dict:
    """Validate a G2 smoke config; returns it unchanged or raises."""
    models = config.get("models")
    if not isinstance(models, list) or not models or not all(
        isinstance(m, str) and m for m in models
    ):
        raise ValueError("config.models must be a non-empty list of model ids")
    chains = config.get("chains")
    if not isinstance(chains, list) or not chains:
        raise ValueError("config.chains must be a non-empty list of chain ids")
    unknown = [c for c in chains if c not in CHAINS_BY_ID]
    if unknown:
        raise ValueError(f"unknown chain ids: {unknown}")
    faults = config.get("fault_conditions")
    if not isinstance(faults, list) or not faults or not all(
        f in ("no_fault", "explicit_error", "empty_partial",
              "silent_wrong_value", "stale_value") for f in faults
    ):
        raise ValueError("config.fault_conditions must be a non-empty list "
                         "of known fault conditions")
    policies = config.get("policies")
    if not isinstance(policies, list) or not policies or not all(
        p in ("M0", "M1", "M2", "M3", "M4", "M5") for p in policies
    ):
        raise ValueError("config.policies must be a non-empty list of "
                         "policies M0-M5")
    if config.get("dependent_mode", G2_DEPENDENT_MODE) not in (
            "revising", "stubborn"):
        raise ValueError("config.dependent_mode must be 'revising' or "
                         "'stubborn'")
    if not isinstance(config.get("seed", 0), int):
        raise ValueError("config.seed must be an int")
    writes = config.get("writes", {})
    if not isinstance(writes, dict):
        raise ValueError("config.writes must map model id -> writes file")
    for model in models:
        if model not in writes:
            raise ValueError(f"config.writes has no pilot writes file for "
                             f"model {model!r}")
    return config


def load_config(path) -> dict:
    config = json.loads(Path(path).read_text())
    return validate_config(config)


# ---------------------------------------------------------------------------
# Store loading (reuse)
# ---------------------------------------------------------------------------

def _cell_key(row: dict) -> tuple:
    return tuple(row[k] for k in _CELL_KEY)


def load_smoke_stores(config: dict, writes_paths: dict | None = None,
                      repo_root: Path | None = None) -> dict:
    """Load the stored write for every smoke cell from the pilot outputs.

    Returns ``{(model, chain, fault, policy): row}`` where ``row`` is
    the pilot writes record (``stored_write``, ``truth``,
    ``falsified_fields``, ...). Raises :class:`ReuseError` on a
    missing cell, a duplicated cell, an unexpected store shape (M0
    must store nothing), or an unreadable writes file. The Qwen2.5
    file holds both Nov 2 models, so rows are selected by model.
    """
    validate_config(config)
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    paths = dict(config.get("writes", {}))
    if writes_paths:
        paths.update(writes_paths)
    wanted = {
        (model, chain, fault, policy)
        for model in config["models"]
        for chain in config["chains"]
        for fault in config["fault_conditions"]
        for policy in config["policies"]
    }
    stores = {}
    for model in config["models"]:
        path = Path(paths[model])
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise ReuseError(
                f"pilot writes file for {model!r} not found: {path} "
                f"(fallback: regenerate with scripts/run_pilot.py; "
                f"see module docstring)")
        try:
            rows = [json.loads(line) for line in path.read_text().splitlines()
                    if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            raise ReuseError(f"cannot read pilot writes file {path}: {exc}") from exc
        for row in rows:
            # Each writes file is the source for its own model only:
            # the Nov 2 file holds both models' rows, and the broken
            # Nov 2 Qwen3 arm must never be picked up for the Qwen3
            # smoke cells (the re-pilot file is their source).
            if row.get("model") != model:
                continue
            key = _cell_key(row) if all(k in row for k in _CELL_KEY) else None
            if key is None or key not in wanted:
                continue
            if key in stores:
                raise ReuseError(
                    f"duplicated smoke cell in {path}: {key} "
                    f"(fallback: regenerate; see module docstring)")
            stored = row.get("stored_write")
            if key[3] == "M0" and stored is not None:
                raise ReuseError(
                    f"unexpected store shape for M0 cell {key}: M0 must "
                    f"store nothing, got {stored!r}")
            if stored is not None and not isinstance(stored, dict):
                raise ReuseError(
                    f"unexpected store shape for cell {key}: stored_write "
                    f"must be a dict or null, got {stored!r}")
            if "truth" not in row or not isinstance(row["truth"], dict):
                raise ReuseError(
                    f"cell {key} carries no recorded DB truth to reconcile "
                    f"the world against")
            stores[key] = row
    missing = sorted(wanted - set(stores))
    if missing:
        raise ReuseError(
            f"{len(missing)} smoke cell(s) missing from the pilot writes "
            f"files: {missing} (fallback: regenerate; see module docstring)")
    return stores


# ---------------------------------------------------------------------------
# World reconstruction: replay + repair to the recorded truth
# ---------------------------------------------------------------------------

def replay_origin_world(chain: dict, fault_condition: str, seed: int = 0):
    """Rebuild the post-origin world by replaying the chain's plan.

    The plan's write tools run in order against a fresh world, with
    the fault injected at the critical step — exactly the
    fault-injection semantics of ``runner.run_origin``, minus the
    agent. For silent/stale faults the DB effect is applied and only
    the response is corrupted, so this reproduces the pilot world
    whenever the origin executed the plan exactly once, in order.
    """
    world = build_world(chain)
    injector = FaultInjector(fault_condition, seed=seed)
    for index, (tool, kwargs) in enumerate(chain["plan"]):
        fn = getattr(world, tool)
        if index == chain["critical_step"]:
            injector.apply(fn, kwargs, chain["fault"])
        else:
            fn(**kwargs)
    return world


def set_fact(world, chain: dict, key: str, value) -> None:
    """Write one ground-truth fact into the world DB.

    The write mirror of ``chains.resolve_fact``: the same DB slot the
    scorer reads. Raises :class:`ReuseError` when the slot cannot be
    set — a ``charged_cents``/``refund_cents`` truth with no payment
    in the DB has nowhere to go.
    """
    state = world.state
    if key == "stock":
        state["products"][chain["product"]]["stock"] = value
        return
    order = state["orders"][chain["order"]]
    if key == "order_status":
        order["status"] = value
    elif key == "payment_status":
        order["payment_status"] = value
    elif key == "address":
        order["address"] = value
    elif key == "tracking":
        order["tracking"] = value
    elif key == "cancellation_fee_cents":
        order["cancellation_fee_cents"] = value
    elif key in ("charged_cents", "refund_cents"):
        payment_id = order["payment_id"]
        if payment_id is None or payment_id not in state["payments"]:
            raise ReuseError(
                f"cannot repair fact {key!r}: no payment in the DB to "
                f"hold the recorded value {value!r}")
        state["payments"][payment_id]["amount_cents"] = value
    else:
        raise ReuseError(f"cannot repair unknown fact key {key!r}")


def reconcile_world(chain: dict, fault_condition: str, seed: int,
                    recorded_truth: dict):
    """Replay the origin plan, then repair the world to recorded truth.

    Returns ``(world, repair)`` where ``repair`` is a list of
    ``{"fact", "replay_value", "recorded_value"}`` — empty when the
    replay already reproduces the pilot world. Raises
    :class:`ReuseError` if a mismatch cannot be patched, or if the
    patched world still disagrees with the recorded truth.
    """
    world = replay_origin_world(chain, fault_condition, seed)
    repair = []
    try:
        replayed_truth = {key: resolve_fact(world, chain, key)
                          for key in recorded_truth}
    except KeyError as exc:
        raise ReuseError(
            f"recorded truth for {chain['chain_id']} / {fault_condition} "
            f"names a fact the world cannot resolve: {exc}") from exc
    for key, recorded in recorded_truth.items():
        replayed = replayed_truth[key]
        if replayed != recorded:
            set_fact(world, chain, key, recorded)
            repair.append({"fact": key, "replay_value": replayed,
                           "recorded_value": recorded})
    for key, recorded in recorded_truth.items():
        if resolve_fact(world, chain, key) != recorded:
            raise ReuseError(
                f"world still disagrees with recorded truth for "
                f"{chain['chain_id']} / {fault_condition} after repair: "
                f"{key}={resolve_fact(world, chain, key)!r} vs "
                f"recorded {recorded!r}")
    return world, repair


# ---------------------------------------------------------------------------
# Fallback: full regeneration config
# ---------------------------------------------------------------------------

def fallback_pilot_config(config: dict) -> dict:
    """A pilot config that regenerates the 24 smoke origins + writes.

    The full pilot grid machinery (``scripts/run_pilot.py``) requires
    exactly the 5 frozen fault conditions and all 6 policies, so the
    fallback config keeps those and narrows only models and chains to
    the smoke grid. ``max_new_tokens`` stays 256 — the Qwen3
    1024-token floor in ``llm_agent.effective_max_new_tokens`` lifts
    Qwen3 actions/writes automatically, exactly as in the re-pilot,
    while Qwen2.5 keeps the pilot's 256/512 budgets.
    """
    return {
        "models": list(config["models"]),
        "chains": list(config["chains"]),
        "fault_conditions": ["no_fault", "explicit_error", "empty_partial",
                             "silent_wrong_value", "stale_value"],
        "policies": ["M0", "M1", "M2", "M3", "M4", "M5"],
        "seed": config.get("seed", 0),
        "max_steps": 8,
        "load_in_4bit": True,
        "max_new_tokens": 256,
        "handlabel_sample_size": 50,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.959964):
    """Wilson 95% CI for k/n. Same formula as scripts/pilot_stats.py."""
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def _rate(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "ci_lo": lo, "ci_hi": hi}


def analyse_cells(cells: list, config: dict) -> dict:
    """Gate checks + reported rates from the dependent-outcome rows.

    ``cells`` are the per-outcome rows written by :func:`run_smoke`.
    Eligibility (G6) is per (model, chain, fault, session): the session
    is eligible iff the M0 and M5 outcomes for that same session are
    both correct. Rates are per model only, never pooled.
    """
    errors = [c for c in cells if c.get("error")]
    by_key = {}
    for cell in cells:
        by_key.setdefault(
            (cell["model"], cell["chain_id"], cell["fault_condition"],
             cell["policy"], cell["session"]), cell)

    def outcome(model, chain, fault, policy, session):
        return by_key.get((model, chain, fault, policy, session))

    answer_yield = _rate(
        sum(1 for c in cells if c.get("parseable") and not c.get("error")),
        len(cells))
    m0_cells = [c for c in cells if c["policy"] == "M0"]
    m5_cells = [c for c in cells if c["policy"] == "M5"]
    m0_correct = _rate(sum(1 for c in m0_cells if c.get("correct")), len(m0_cells))
    m5_correct = _rate(sum(1 for c in m5_cells if c.get("correct")), len(m5_cells))
    s4 = [c for c in cells if c["session"] == 4]
    correction_ok = all(c.get("correction_in_prompt") for c in s4) and bool(s4)

    # Eligibility per (model, chain, fault, session).
    eligible = {}
    for model in config["models"]:
        for chain in config["chains"]:
            for fault in config["fault_conditions"]:
                for session in (2, 3, 4):
                    m0 = outcome(model, chain, fault, "M0", session)
                    m5 = outcome(model, chain, fault, "M5", session)
                    eligible[(model, chain, fault, session)] = bool(
                        m0 and m5 and m0.get("correct") and m5.get("correct"))

    contrast_groups = {}
    for model in config["models"]:
        for chain in config["chains"]:
            for fault in config["fault_conditions"]:
                sessions = [s for s in (2, 3, 4)
                            if eligible[(model, chain, fault, s)]]
                contrast_groups[f"{model}|{chain}|{fault}"] = sessions
    contrast_ok = all(bool(v) for v in contrast_groups.values())

    per_model = {}
    for model in config["models"]:
        def elig_m1(session=None):
            rows = []
            for chain in config["chains"]:
                for fault in config["fault_conditions"]:
                    for s in (2, 3, 4):
                        if session is not None and s != session:
                            continue
                        if not eligible[(model, chain, fault, s)]:
                            continue
                        cell = outcome(model, chain, fault, "M1", s)
                        if cell is not None:
                            rows.append(cell)
            return rows

        elig_all = elig_m1()
        raw_all = [c for c in cells if c["model"] == model and c["policy"] == "M1"]
        # Persistence S2 -> S3: incorrect at S2, still incorrect at S3.
        pers_num, pers_den = 0, 0
        corr_num, corr_den = 0, 0
        for chain in config["chains"]:
            for fault in config["fault_conditions"]:
                s2 = outcome(model, chain, fault, "M1", 2)
                s3 = outcome(model, chain, fault, "M1", 3)
                s4c = outcome(model, chain, fault, "M1", 4)
                if (eligible[(model, chain, fault, 2)]
                        and eligible[(model, chain, fault, 3)]
                        and s2 is not None and not s2.get("correct")):
                    pers_den += 1
                    if s3 is not None and not s3.get("correct"):
                        pers_num += 1
                if (eligible[(model, chain, fault, 3)]
                        and eligible[(model, chain, fault, 4)]
                        and s3 is not None and not s3.get("correct")):
                    corr_den += 1
                    if s4c is not None and not s4c.get("correct"):
                        corr_num += 1
        per_model[model] = {
            "eligible_sessions": sum(1 for (m, _, _, _), v in eligible.items()
                                     if m == model and v),
            "eligible_groups": sum(
                1 for k, v in contrast_groups.items()
                if k.startswith(model + "|") and v),
            "propagation_eligible": _rate(
                sum(1 for c in elig_all if not c.get("correct")), len(elig_all)),
            "propagation_raw": _rate(
                sum(1 for c in raw_all if not c.get("correct")), len(raw_all)),
            "persistence_s2_to_s3": _rate(pers_num, pers_den),
            "correction_resistance_s3_to_s4": _rate(corr_num, corr_den),
        }

    gates = {
        "answer_yield_ge_80pct": answer_yield["rate"] is not None
        and answer_yield["rate"] >= 0.80,
        "m0_floor_holds": m0_correct["rate"] == 1.0 and len(m0_cells) > 0,
        "m5_ceiling_holds": m5_correct["rate"] == 1.0 and len(m5_cells) > 0,
        "no_errors_and_correction_present": not errors and correction_ok,
        "contrast_computable": contrast_ok,
    }
    gates["pass"] = all(gates.values())
    return {
        "cells": len(cells),
        "errors": len(errors),
        "answer_yield": answer_yield,
        "m0_correct": m0_correct,
        "m5_correct": m5_correct,
        "correction_in_all_s4_prompts": correction_ok,
        "contrast_groups": contrast_groups,
        "per_model": per_model,
        "gates": gates,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_CELL_FIELDS = [
    "model", "chain_id", "fault_condition", "policy", "session", "kind",
    "fact", "believed", "truth", "correct", "parseable", "invalid_replies",
    "prompt_tokens", "completion_tokens", "cell_seconds", "transcript_file",
    "origin_falsified_fields", "world_repair", "correction",
    "correction_in_prompt", "error",
]


def _model_short(model: str) -> str:
    if model in _MODEL_SHORT:
        return _MODEL_SHORT[model]
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()


def _transcript_name(cell: dict) -> str:
    chain_short = cell["chain_id"].split("_")[0]
    return (f"{_model_short(cell['model'])}_{chain_short}_"
            f"{cell['fault_condition']}_{cell['policy']}_"
            f"s{cell['session']}.json")


def _cells_csv(cells: list) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_CELL_FIELDS)
    writer.writeheader()
    for cell in cells:
        writer.writerow({
            k: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v
            for k, v in cell.items()})
    return buffer.getvalue()


def _summary_md(config: dict, analysis: dict, repairs: list) -> str:
    gates = analysis["gates"]
    lines = [
        "# PersistBench Gate G2 — propagation smoke",
        "",
        f"Models: {', '.join(config['models'])} · chains: "
        f"{', '.join(config['chains'])} · faults: "
        f"{', '.join(config['fault_conditions'])} · policies: "
        f"{', '.join(config['policies'])} · dependent mode: "
        f"{config.get('dependent_mode', G2_DEPENDENT_MODE)} · "
        f"seed {config.get('seed', 0)}",
        "",
        f"Cells: {analysis['cells']} dependent outcomes "
        f"(24 chain runs x 3 sessions) · errors: {analysis['errors']}",
        "",
        "## Gate verdict",
        "",
        f"**{'PASS' if gates['pass'] else 'FAIL'}** — a clean null (M1 "
        "downstream ~= M0 downstream with floor and ceiling holding) is "
        "a reportable smoke outcome, not a FAIL.",
        "",
        "| Gate | Result |",
        "|---|---|",
        f"| 1. Answer yield >= 80% | "
        f"{analysis['answer_yield']['rate']:.3f} "
        f"({analysis['answer_yield']['k']}/{analysis['answer_yield']['n']}) "
        f"— {'PASS' if gates['answer_yield_ge_80pct'] else 'FAIL'} |",
        f"| 2. M0 floor holds (all M0 correct) | "
        f"{analysis['m0_correct']['rate']:.3f} "
        f"({analysis['m0_correct']['k']}/{analysis['m0_correct']['n']}) "
        f"— {'PASS' if gates['m0_floor_holds'] else 'FAIL'} |",
        f"| 3. M5 ceiling holds (all M5 correct) | "
        f"{analysis['m5_correct']['rate']:.3f} "
        f"({analysis['m5_correct']['k']}/{analysis['m5_correct']['n']}) "
        f"— {'PASS' if gates['m5_ceiling_holds'] else 'FAIL'} |",
        f"| 4. No errors; correction in every Session-4 prompt | "
        f"{'PASS' if gates['no_errors_and_correction_present'] else 'FAIL'} |",
        f"| 5. Propagation contrast computable (>= 1 eligible session "
        f"per model x chain x fault) | "
        f"{'PASS' if gates['contrast_computable'] else 'FAIL'} |",
        "",
        "## Reported rates (per model only, never pooled)",
        "",
        "Propagation = share of G6-eligible M1 outcomes incorrect; "
        "persistence = incorrect at Session 2 and still incorrect at "
        "Session 3; correction resistance = incorrect at Session 3 and "
        "still incorrect at Session 4 after the verified observation. "
        "Wilson 95% CIs; raw (unfiltered) M1 numbers alongside.",
        "",
        "| Model | Eligible sessions | Propagation (eligible) | "
        "Propagation (raw) | Persistence S2->S3 | Correction resistance S3->S4 |",
        "|---|---|---|---|---|---|",
    ]
    for model, row in analysis["per_model"].items():
        def fmt(r):
            if r["rate"] is None:
                return "n/a"
            return f"{r['rate']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] ({r['k']}/{r['n']})"
        lines.append(
            f"| {model} | {row['eligible_sessions']} | "
            f"{fmt(row['propagation_eligible'])} | "
            f"{fmt(row['propagation_raw'])} | "
            f"{fmt(row['persistence_s2_to_s3'])} | "
            f"{fmt(row['correction_resistance_s3_to_s4'])} |")
    lines += [
        "",
        "## World repairs",
        "",
        "The pilot writes record DB truth but not the origin tool steps, "
        "so each cell's world is a plan replay repaired to the recorded "
        "truth (see module docstring). Repairs below; an empty list "
        "means every replay already reproduced the pilot world.",
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
        "- G2 is a measurability smoke (n=2 chains): effect sizes are "
        "reported, not gated. No per-fault claim is made below n=12 "
        "chains.",
        "- H4 is narrowed for this gate: the live dependent does not "
        "condition on provenance (design note); M2 is not in the smoke "
        "policy set.",
        "",
    ]
    return "\n".join(lines)


def run_smoke(config: dict, backend_factory, out_dir,
              writes_paths: dict | None = None,
              repo_root: Path | None = None) -> dict:
    """Run the 24-cell smoke against reused pilot stores.

    ``backend_factory(model_id) -> ChatBackend``. Writes
    ``g2_smoke_cells.jsonl`` / ``.csv``, one transcript JSON per
    dependent outcome under ``transcripts/``, and
    ``g2_smoke_summary.json`` / ``.md`` under ``out_dir``. Returns
    the summary dict (also written to ``g2_smoke_summary.json``).
    Raises :class:`ReuseError` (after writing
    ``fallback_pilot_config.json``) when the pilot stores cannot be
    reused — the fallback is full regeneration, never executed here.
    """
    validate_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = config.get("seed", 0)
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
            for fault in config["fault_conditions"]:
                for policy in config["policies"]:
                    key = (model, chain_id, fault, policy)
                    row = stores[key]
                    try:
                        world, repair = reconcile_world(
                            chain, fault, seed, row["truth"])
                    except ReuseError:
                        (out / "fallback_pilot_config.json").write_text(
                            json.dumps(fallback_pilot_config(config), indent=2)
                            + "\n")
                        raise
                    if repair:
                        repairs.append({**{k: key[i] for i, k in
                                           enumerate(("model", "chain_id",
                                                      "fault_condition",
                                                      "policy"))},
                                        "repair": repair})
                    memory = row["stored_write"]
                    agent = LLMDependentAgent(
                        backend, memory,
                        mode=config.get("dependent_mode", G2_DEPENDENT_MODE),
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

    analysis = analyse_cells(cells, config)
    summary = {
        "config": config,
        "cells": len(cells),
        "world_repairs": repairs,
        "analysis": analysis,
    }
    (out / "g2_smoke_cells.jsonl").write_text(
        "\n".join(json.dumps(c, default=str) for c in cells) + "\n")
    (out / "g2_smoke_cells.csv").write_text(_cells_csv(cells))
    (out / "g2_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    (out / "g2_smoke_summary.md").write_text(
        _summary_md(config, analysis, repairs))
    return summary
