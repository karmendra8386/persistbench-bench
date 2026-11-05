#!/usr/bin/env python3
"""Run the Gate G2 propagation smoke.

Reuses the pilot stores (see persistbench/g2_smoke.py): the 24 origin
trajectories + stored writes already exist, so this generates only the
72 dependent answers live — minutes on a T4, not a second pilot.

Usage:
    python scripts/g2_smoke.py --config configs/g2_smoke.json \
        --out results/g2-smoke [--backend hf|fake]

--backend hf   Hugging Face transformers (default; the two G1-cleared
               models at 4-bit on a Kaggle T4, per the config).
--backend fake Deterministic scripted backend — plumbing smoke only.
               It always abstains ({"answer": null}), so a fake run
               exercises loading, driving, writing, and gate
               arithmetic and is EXPECTED to FAIL the answer-yield
               gate with exit 1. Numbers from a fake run are NOT
               smoke results. The meaningful fake-backend check (a
               correct-reader stand-in) lives in tests/test_g2_smoke.py.

--qwen25-writes / --qwen3-writes override the pilot writes files
(defaults are the config's `writes` map, relative to the repo root);
use them on Kaggle if the pilot outputs were uploaded elsewhere.

On a reuse failure (missing/duplicated cell, unexpected store shape,
unrepairable world) the run writes `fallback_pilot_config.json` into
the output directory and exits non-zero: regenerate the 24 origins +
writes with `scripts/run_pilot.py` on that config, then re-run the
smoke against the fresh `pilot_writes.jsonl`. The fallback is never
executed here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persistbench.g2_smoke import ReuseError, load_config, run_smoke  # noqa: E402
from persistbench.llm_agent import FakeLLMBackend, HFTransformersBackend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/g2_smoke.json")
    parser.add_argument("--out", default="results/g2-smoke")
    parser.add_argument("--backend", choices=["hf", "fake"], default="hf")
    parser.add_argument("--qwen25-writes", default=None)
    parser.add_argument("--qwen3-writes", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    writes_paths = {}
    if args.qwen25_writes:
        writes_paths["Qwen/Qwen2.5-7B-Instruct"] = args.qwen25_writes
    if args.qwen3_writes:
        writes_paths["Qwen/Qwen3-8B"] = args.qwen3_writes

    if args.backend == "fake":
        def factory(model_id):
            return FakeLLMBackend(
                lambda messages: '{"answer": null}',
                model_label=f"fake/{model_id}",
            )
    else:
        def factory(model_id):
            return HFTransformersBackend(
                model_id,
                load_in_4bit=config.get("load_in_4bit", True),
                max_new_tokens=config.get("max_new_tokens", 256),
                seed=config.get("seed", 0),
            )

    try:
        summary = run_smoke(config, factory, args.out,
                            writes_paths=writes_paths or None)
    except ReuseError as exc:
        print(f"REUSE FAILED: {exc}")
        print(f"Fallback config written to {args.out}/fallback_pilot_config.json")
        print("Regenerate with: python scripts/run_pilot.py "
              f"--config {args.out}/fallback_pilot_config.json "
              "--out <fresh-pilot-dir>, then re-run this smoke.")
        return 2
    analysis = summary["analysis"]
    gates = analysis["gates"]
    print(f"smoke outcomes: {analysis['cells']} -> {args.out}")
    print(f"  answer yield: {analysis['answer_yield']['rate']:.3f} "
          f"({analysis['answer_yield']['k']}/{analysis['answer_yield']['n']})")
    print(f"  M0 correct: {analysis['m0_correct']['k']}/{analysis['m0_correct']['n']}  "
          f"M5 correct: {analysis['m5_correct']['k']}/{analysis['m5_correct']['n']}")
    print(f"  world repairs: {len(summary['world_repairs'])}")
    print(f"  GATE G2: {'PASS' if gates['pass'] else 'FAIL'}")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
