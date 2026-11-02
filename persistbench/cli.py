"""Command-line interface for PersistAIBench.

Installed as the ``persistaibench`` entry point (see pyproject.toml).

Subcommands:

``run``   Run ONE pilot cell (model x chain x fault x policy) and print
          the scored cell as JSON — the same per-cell computation the
          full pilot grid performs in ``pilot.run_pilot``.

``pilot`` Run the full pilot grid from a config file; equivalent to
          ``scripts/run_pilot.py``.

Backends:

``--backend fake``  Deterministic scripted backend. Plumbing smoke only —
                    numbers from a fake run are NOT results; the model
                    id is stamped ``fake/...`` so output cannot be
                    mistaken for a real model run.
``--backend hf``    Hugging Face transformers backend (requires torch +
                    transformers; models run at 4-bit by default, per
                    the pilot configuration).
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .chains import CHAINS_BY_ID
from .faults import FAULT_CONDITIONS
from .llm_agent import FakeLLMBackend, HFTransformersBackend, LLMAgent
from .memory import POLICIES
from .pilot import (
    _FORMATION_POLICIES,
    _fresh_policy,
    _policy_trajectory,
    default_config,
    load_config,
    run_pilot,
)
from .scorer import classify_write, grade_origin


def resolve_chain(chain_arg: str) -> str:
    """Resolve a chain id or unique id prefix (e.g. ``chain01``)."""
    if chain_arg in CHAINS_BY_ID:
        return chain_arg
    matches = [cid for cid in CHAINS_BY_ID if cid.startswith(chain_arg)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(
            f"error: unknown chain {chain_arg!r}; known chains: "
            + ", ".join(sorted(CHAINS_BY_ID))
        )
    raise SystemExit(
        f"error: chain prefix {chain_arg!r} is ambiguous: "
        + ", ".join(sorted(matches))
    )


def _backend_factory(backend: str, *, load_in_4bit: bool = True,
                     max_new_tokens: int = 256, seed: int = 0):
    if backend == "fake":
        def factory(model_id):
            return FakeLLMBackend(
                lambda messages: '{"done": true}',
                model_label=f"fake/{model_id}",
            )
    else:
        def factory(model_id):
            return HFTransformersBackend(
                model_id,
                load_in_4bit=load_in_4bit,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
    return factory


def run_cell(config: dict, chain_id: str, fault: str, policy_name: str,
             backend_factory) -> dict:
    """Run a single (model, chain, fault, policy) cell.

    Mirrors the per-cell computation in ``pilot.run_pilot``: the origin
    trajectory is generated once for the cell's model, the policy write
    is generated in the policy's style, and formation is scored by the
    programmatic classifier (no LLM judge).
    """
    model_id = config["models"][0]
    chain = CHAINS_BY_ID[chain_id]
    seed = config.get("seed", 0)
    backend = backend_factory(model_id)
    agent = LLMAgent(backend, max_steps=config.get("max_steps", 8))
    trajectory = agent.run_origin(chain, fault, seed=seed)
    origin_grade = grade_origin(trajectory, chain)
    policy = _fresh_policy(policy_name)
    traj_for_write, _write = _policy_trajectory(
        agent, trajectory, chain, policy_name, seed
    )
    store = policy.write(traj_for_write, chain, trajectory.world)
    falsified = classify_write(store, trajectory, chain)
    return {
        "model": backend.model_label,
        "chain_id": chain_id,
        "fault_condition": fault,
        "policy": policy_name,
        "task_success": origin_grade.task_success,
        "origin_claim_truthful": origin_grade.claim_truthful,
        "formation": (
            1 if falsified else 0
        ) if policy_name in _FORMATION_POLICIES else None,
        "falsified_fields": falsified,
        "stored_write": store,
        "truth": origin_grade.truth,
    }


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["fake", "hf"], default="fake",
                        help="fake = deterministic scripted backend "
                             "(plumbing smoke only, default); hf = real "
                             "model via Hugging Face transformers")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="persistaibench",
        description="PersistAIBench — an evaluation benchmark for AI "
                    "agents' memory truthfulness over long multi-step "
                    "tasks.",
    )
    parser.add_argument("--version", action="version",
                        version=f"persistaibench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run", help="run one chain/policy cell and print the scored cell as JSON"
    )
    p_run.add_argument("--chain", required=True,
                       help="chain id or unique prefix, e.g. chain01")
    p_run.add_argument("--policy", default="M1", choices=sorted(POLICIES),
                       help="memory policy M0-M5 (default M1)")
    p_run.add_argument("--fault", default="silent_wrong_value",
                       choices=list(FAULT_CONDITIONS),
                       help="fault condition (default silent_wrong_value)")
    p_run.add_argument("--model", default=None,
                       help="model id (default: first pilot model)")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--max-steps", type=int, default=8)
    p_run.add_argument("--max-new-tokens", type=int, default=256)
    p_run.add_argument("--no-4bit", action="store_true",
                       help="hf backend: do not load the model in 4-bit")
    _add_backend_args(p_run)

    p_pilot = sub.add_parser(
        "pilot", help="run the full pilot grid from a config file"
    )
    p_pilot.add_argument("--config", default="configs/pilot.json")
    p_pilot.add_argument("--out", default="results/pilot")
    _add_backend_args(p_pilot)
    p_pilot.set_defaults(backend="hf")

    args = parser.parse_args(argv)

    if args.command == "run":
        config = default_config()
        if args.model:
            config["models"] = [args.model]
        config["seed"] = args.seed
        config["max_steps"] = args.max_steps
        factory = _backend_factory(
            args.backend,
            load_in_4bit=not args.no_4bit,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        cell = run_cell(
            config, resolve_chain(args.chain), args.fault, args.policy, factory
        )
        print(json.dumps(cell, indent=2, default=str))
        return 0

    if args.command == "pilot":
        config = load_config(args.config)
        factory = _backend_factory(
            args.backend,
            load_in_4bit=config.get("load_in_4bit", True),
            max_new_tokens=config.get("max_new_tokens", 256),
            seed=config.get("seed", 0),
        )
        summary = run_pilot(config, factory, args.out)
        print(f"pilot cells: {summary['cells']} -> {args.out}")
        for row in summary["formation"]:
            rate = row["formation_rate"]
            rate_text = "n/a" if rate is None else f"{rate:.3f}"
            print(f"  {row['fault_condition']:<18} {row['policy']}: "
                  f"formation {rate_text} "
                  f"({row['false_memories']}/{row['writes']})")
        return 0

    return 2  # pragma: no cover - argparse enforces a subcommand


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
