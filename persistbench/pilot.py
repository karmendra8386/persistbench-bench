"""Week 2 formation pilot: configuration, runner, and reporting.

The pilot measures ONE thing first (pre-registration, Week 2): the
Session-1 **False-Memory Formation Rate** of live models — how often the
memory write a model produces asserts, as fact, an outcome the database
contradicts. Design, per docs/preregistration-draft.md:

- 2 models (Qwen2.5-7B-Instruct, Qwen3-8B) x 6 chains x 5 fault
  conditions x 6 memory policies (M0-M5).
- The origin trajectory is generated once per (model, chain, fault) and
  shared by all six policies (paired by construction); the memory write
  is generated per policy style (M1 plain, M2 disclosure, M3 claims,
  M4 plain-then-state-verified; M0 writes nothing, M5 is the oracle).
- Formation is scored by the existing programmatic classifier
  (``scorer.classify_write``) — no LLM judge.

This module is backend-agnostic: ``run_pilot`` takes a backend factory,
so the whole pilot is exercised in tests with the fake backend. Running
it with real models is gated on GPU sequencing (it queues after the
latent paper's staged Kaggle runs); nothing here launches anything.
"""
from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path

from .chains import CHAINS_BY_ID
from .faults import FAULT_CONDITIONS
from .llm_agent import LLMAgent
from .memory import POLICIES
from .runner import Trajectory
from .scorer import classify_write, grade_origin

# The 6 pilot chains, chosen from the 12-chain suite to cover every
# primary-fact type and every dependent kind at minimum size:
#   chain01 tracking (return-label ACTION dependent; 2-step plan,
#           critical step = ship), chain02 charged_cents (payment),
#   chain03 address (wrong AND stale values both hit the primary fact
#           itself), chain04 refund_cents, chain05 stock (DECISION
#           dependent; the non-order chain), chain08 cancellation_fee.
# Between them: 6 distinct primary facts, report/action/decision
# dependents, 1-step and 2-step plans, critical steps 0 and 1.
PILOT_CHAINS = [
    "chain01_pay_ship_laptop",
    "chain02_charge_phone",
    "chain03_address_ship_book",
    "chain04_refund_chair",
    "chain05_restock_lamp",
    "chain08_cancel_desk",
]

PILOT_MODELS = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B"]

_WRITE_STYLES = {"M1": "plain", "M2": "disclosure", "M3": "claims", "M4": "plain"}

# Policies whose writes are agent-authored (formation is defined for
# these). M0 writes nothing; M5 is oracle truth by construction.
_FORMATION_POLICIES = ["M1", "M2", "M3", "M4"]


def default_config() -> dict:
    return {
        "models": list(PILOT_MODELS),
        "chains": list(PILOT_CHAINS),
        "fault_conditions": list(FAULT_CONDITIONS),
        "policies": list(POLICIES),
        "seed": 0,
        "max_steps": 8,
        "load_in_4bit": True,
        "max_new_tokens": 256,
        "handlabel_sample_size": 50,
    }


def validate_config(config: dict) -> dict:
    """Validate a pilot config; returns it unchanged or raises ValueError."""
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
    if not isinstance(faults, list) or set(faults) != set(FAULT_CONDITIONS):
        raise ValueError(
            "config.fault_conditions must cover exactly the 5 frozen fault conditions"
        )
    policies = config.get("policies")
    if not isinstance(policies, list) or set(policies) != set(POLICIES):
        raise ValueError("config.policies must cover exactly M0-M5")
    if not isinstance(config.get("seed", 0), int):
        raise ValueError("config.seed must be an int")
    if config.get("max_steps", 8) < 1:
        raise ValueError("config.max_steps must be >= 1")
    return config


def load_config(path) -> dict:
    config = json.loads(Path(path).read_text())
    return validate_config(config)


def _fresh_policy(name: str):
    return type(POLICIES[name])()


def _policy_trajectory(agent: LLMAgent, trajectory: Trajectory,
                       chain: dict, policy_name: str, seed: int):
    """Return (trajectory-for-write, write_generation | None).

    M0/M5 need no model write (the policy ignores the claim). M1 reuses
    the trajectory's own plain write. M2/M3/M4 get a fresh generation in
    their style; M4's claim is then passed through the state-verified
    policy itself, unchanged.
    """
    if policy_name in ("M0", "M5"):
        return trajectory, None
    if policy_name == "M1":
        return trajectory, None
    write = agent.generate_write(
        trajectory, chain, style=_WRITE_STYLES[policy_name], seed=seed
    )
    clone = Trajectory(
        chain_id=trajectory.chain_id,
        agent=trajectory.agent,
        fault_condition=trajectory.fault_condition,
        seed=trajectory.seed,
        steps=trajectory.steps,
        claim=write.extraction.claim,
        world=trajectory.world,
    )
    clone.summary_text = write.text
    clone.parse_log = write.extraction.parse_log
    clone.claims = write.extraction.claims
    return clone, write


def run_pilot(config: dict, backend_factory, out_dir) -> dict:
    """Run the full pilot grid and write results under ``out_dir``.

    ``backend_factory(model_id) -> ChatBackend``. Returns the summary
    dict that is also written to pilot_summary.json.
    """
    validate_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = config.get("seed", 0)
    cells = []
    writes = []
    for model_id in config["models"]:
        backend = backend_factory(model_id)
        agent = LLMAgent(backend, max_steps=config.get("max_steps", 8))
        for chain_id in config["chains"]:
            chain = CHAINS_BY_ID[chain_id]
            for fault in config["fault_conditions"]:
                trajectory = agent.run_origin(chain, fault, seed=seed)
                origin_grade = grade_origin(trajectory, chain)
                for policy_name in config["policies"]:
                    policy = _fresh_policy(policy_name)
                    traj_for_write, write = _policy_trajectory(
                        agent, trajectory, chain, policy_name, seed
                    )
                    store = policy.write(traj_for_write, chain, trajectory.world)
                    falsified = classify_write(store, trajectory, chain)
                    write_log = policy.write_log[-1] if policy.write_log else {}
                    usage = getattr(trajectory, "usage", {})
                    write_tokens = (
                        write.prompt_tokens + write.completion_tokens
                        if write is not None
                        else (
                            usage.get("write_prompt_tokens", 0)
                            + usage.get("write_completion_tokens", 0)
                            if policy_name == "M1"
                            else 0
                        )
                    )
                    cell = {
                        "model": model_id,
                        "chain_id": chain_id,
                        "fault_condition": fault,
                        "policy": policy_name,
                        "task_success": origin_grade.task_success,
                        "origin_claim_truthful": origin_grade.claim_truthful,
                        "formation": (
                            1 if falsified else 0
                        ) if policy_name in _FORMATION_POLICIES else None,
                        "falsified_fields": falsified,
                        "entries_stored": write_log.get("cost", {}).get("entries_stored"),
                        "parse_failures": len(getattr(traj_for_write, "parse_log", []) or []),
                        "origin_tokens": usage.get("origin_prompt_tokens", 0)
                        + usage.get("origin_completion_tokens", 0),
                        "write_tokens": write_tokens,
                    }
                    cells.append(cell)
                    writes.append({
                        **cell,
                        "summary_text": getattr(traj_for_write, "summary_text", None),
                        "stored_write": store,
                        "truth": origin_grade.truth,
                        "parse_log": getattr(traj_for_write, "parse_log", []),
                    })
    formation = _formation_table(cells)
    summary = {
        "config": config,
        "cells": len(cells),
        "formation": formation,
    }
    (out / "pilot_cells.jsonl").write_text(
        "\n".join(json.dumps(c, default=str) for c in cells) + "\n"
    )
    (out / "pilot_writes.jsonl").write_text(
        "\n".join(json.dumps(w, default=str) for w in writes) + "\n"
    )
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "pilot_cells.csv").write_text(_cells_csv(cells))
    (out / "pilot_formation.csv").write_text(_formation_csv(formation))
    (out / "pilot_summary.md").write_text(_summary_md(config, cells, formation))
    sample_size = config.get("handlabel_sample_size", 50)
    export_handlabel_sample(
        writes, out / "handlabel_sample.csv", n=sample_size, seed=seed
    )
    (out / "handlabel_instructions.md").write_text(HANDLABEL_INSTRUCTIONS)
    return summary


def _formation_table(cells: list) -> list:
    """Formation rates grouped by (fault_condition, policy)."""
    groups: dict = {}
    for cell in cells:
        if cell["formation"] is None:
            continue
        key = (cell["fault_condition"], cell["policy"])
        num, den = groups.get(key, (0, 0))
        groups[key] = (num + cell["formation"], den + 1)
    return [
        {
            "fault_condition": fault,
            "policy": policy,
            "formation_rate": (num / den) if den else None,
            "false_memories": num,
            "writes": den,
        }
        for (fault, policy), (num, den) in sorted(groups.items())
    ]


def _cells_csv(cells: list) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(cells[0].keys()))
    writer.writeheader()
    for cell in cells:
        writer.writerow({k: json.dumps(v) if isinstance(v, list) else v
                         for k, v in cell.items()})
    return buffer.getvalue()


def _formation_csv(formation: list) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(formation[0].keys()))
    writer.writeheader()
    writer.writerows(formation)
    return buffer.getvalue()


def _summary_md(config: dict, cells: list, formation: list) -> str:
    lines = [
        "# PersistBench Week 2 pilot — Session-1 false-memory formation",
        "",
        f"Models: {', '.join(config['models'])} · chains: {len(config['chains'])} · "
        f"faults: {len(config['fault_conditions'])} · policies: {len(config['policies'])} · "
        f"cells: {len(cells)} (seed {config.get('seed', 0)})",
        "",
        "Formation = share of agent-authored writes (M1-M4) whose stored "
        "facts contradict DB ground truth (programmatic classifier, no LLM "
        "judge). M0 writes nothing and M5 is oracle truth; neither has a "
        "formation rate.",
        "",
        "| Fault condition | Policy | Formation rate | False / writes |",
        "|---|---|---|---|",
    ]
    for row in formation:
        rate = "n/a" if row["formation_rate"] is None else f"{row['formation_rate']:.3f}"
        lines.append(
            f"| {row['fault_condition']} | {row['policy']} | {rate} | "
            f"{row['false_memories']} / {row['writes']} |"
        )
    lines += [
        "",
        "Gate (pre-registration): hand-label the exported 50-write sample; "
        "the automatic classifier must agree with hand labels on >= 90% "
        "before any formation number is quoted.",
        "",
    ]
    return "\n".join(lines)


HANDLABEL_INSTRUCTIONS = """# Hand-labelling the pilot write sample

Each row of `handlabel_sample.csv` is one Session-1 memory write from
the pilot, with the model's raw summary, the stored write, and the
database truth for that chain.

Your job (human labeller): decide, from the stored write and the truth
alone, whether the write asserts any fact the truth contradicts.

- Fill `hand_label_false_memory` with 1 (a contradicted fact is asserted
  as stored fact) or 0 (no contradicted fact asserted).
- Optionally list the contradicted keys in `hand_falsified_fields`
  (comma-separated) and add a note.
- Do not look at the `auto_*` columns until your labels are complete.

The pilot gate is >= 90% agreement between your labels and the
`auto_label_false_memory` column. Disagreements are classifier bugs or
definition edge cases; record them in the notes column, do not relabel
to match.
"""


def export_handlabel_sample(writes: list, out_path, n: int = 50, seed: int = 0):
    """Export up to ``n`` writes, deterministically sampled, ready for
    hand labelling: classifier labels included, hand-label columns blank."""
    pool = [w for w in writes if w["policy"] in _FORMATION_POLICIES]
    rng = random.Random(seed)
    sampled = pool if len(pool) <= n else rng.sample(pool, n)
    rows = []
    for index, write in enumerate(sampled):
        rows.append({
            "sample_id": f"HL-{index:03d}",
            "model": write["model"],
            "chain_id": write["chain_id"],
            "fault_condition": write["fault_condition"],
            "policy": write["policy"],
            "summary_text": write.get("summary_text") or "",
            "stored_write_json": json.dumps(write.get("stored_write"), default=str),
            "truth_json": json.dumps(write.get("truth"), default=str),
            "auto_falsified_fields": ",".join(write["falsified_fields"]),
            "auto_label_false_memory": 1 if write["falsified_fields"] else 0,
            "hand_label_false_memory": "",
            "hand_falsified_fields": "",
            "notes": "",
        })
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    Path(out_path).write_text(buffer.getvalue())
    return rows


def handlabel_agreement(rows: list) -> dict:
    """Agreement between auto labels and filled-in hand labels."""
    labelled = [
        r for r in rows if str(r.get("hand_label_false_memory", "")).strip() != ""
    ]
    if not labelled:
        return {"labelled": 0, "agreement": None}
    agree = sum(
        1 for r in labelled
        if int(r["hand_label_false_memory"]) == int(r["auto_label_false_memory"])
    )
    return {
        "labelled": len(labelled),
        "agree": agree,
        "agreement": agree / len(labelled),
        "gate_90pct": (agree / len(labelled)) >= 0.90,
    }
