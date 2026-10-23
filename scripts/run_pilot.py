#!/usr/bin/env python3
"""Run the PersistBench Week 2 formation pilot.

PREP STATUS: this script is ready to run, but the pilot itself is gated
on GPU sequencing — it queues after the latent paper's staged Kaggle
runs (see kaggle/RUNBOOK.md). Do not launch it until that sequencing
clears.

Usage:
    python scripts/run_pilot.py --config configs/pilot.json \
        --out results/pilot [--backend hf|fake]

--backend hf   Hugging Face transformers (default; pilot models at
               4-bit on a Kaggle T4, per the config).
--backend fake Deterministic scripted backend — plumbing smoke only.
               Numbers from a fake run are NOT pilot results and the
               output directory is stamped accordingly by the config
               echo in pilot_summary.json (model ids read "fake/...").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persistbench.llm_agent import FakeLLMBackend, HFTransformersBackend  # noqa: E402
from persistbench.pilot import load_config, run_pilot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--out", default="results/pilot")
    parser.add_argument("--backend", choices=["hf", "fake"], default="hf")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.backend == "fake":
        def factory(model_id):
            return FakeLLMBackend(
                lambda messages: '{"done": true}',
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

    summary = run_pilot(config, factory, args.out)
    print(f"pilot cells: {summary['cells']} -> {args.out}")
    for row in summary["formation"]:
        rate = row["formation_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.3f}"
        print(f"  {row['fault_condition']:<18} {row['policy']}: "
              f"formation {rate_text} ({row['false_memories']}/{row['writes']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
