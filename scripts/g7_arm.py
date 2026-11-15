#!/usr/bin/env python3
"""Run the Gate G7 populated-store sensitivity arm.

Re-runs the Gate G2 smoke grid (see persistbench/g7_arm.py) with the
memory store pre-populated with true distractor entries from other
chains' completed tasks, reusing the same pilot stores — the arm
generates only the 72 dependent answers live.

Usage:
    python scripts/g7_arm.py --config configs/g7_arm.json \
        --out results/g7-arm [--backend hf|fake]

--backend hf   Hugging Face transformers (default; the two
               G1-cleared models at 4-bit on a Kaggle T4, per the
               config).
--backend fake Deterministic scripted backend — plumbing only.
               It always abstains ({"answer": null}), so a fake run
               exercises loading, seeding, driving and writing, but
               its numbers are NOT arm results. The meaningful
               fake-backend check (a correct-reader stand-in) lives
               in tests/test_g7_arm.py.

--qwen25-writes / --qwen3-writes override the pilot writes files
(defaults are the config's `writes` map, relative to the repo
root); use them on Kaggle if the pilot outputs were uploaded
elsewhere.

Exit codes — the arm is a measurement, not a gate:

- 0: the arm COMPLETED (grid ran, outputs written). Read the
  populated-vs-empty comparison in g7_arm_summary.md; there is no
  PASS/FAIL verdict and no direction is gated on.
- 2: reuse failure (missing/duplicated cell, unexpected store
  shape, unrepairable world, seed collision). The run writes
  `fallback_pilot_config.json` into the output directory:
  regenerate the 24 origins + writes with `scripts/run_pilot.py`
  on that config, then re-run the arm against the fresh
  `pilot_writes.jsonl`. The fallback is never executed here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persistbench.g2_smoke import ReuseError  # noqa: E402
from persistbench.g7_arm import load_config, run_arm  # noqa: E402
from persistbench.llm_agent import FakeLLMBackend, HFTransformersBackend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/g7_arm.json")
    parser.add_argument("--out", default="results/g7-arm")
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
        summary = run_arm(config, factory, args.out,
                          writes_paths=writes_paths or None)
    except ReuseError as exc:
        print(f"REUSE FAILED: {exc}")
        print(f"Fallback config written to {args.out}/fallback_pilot_config.json")
        print("Regenerate with: python scripts/run_pilot.py "
              f"--config {args.out}/fallback_pilot_config.json "
              "--out <fresh-pilot-dir>, then re-run this arm.")
        return 2
    analysis = summary["analysis"]
    print(f"arm outcomes: {analysis['cells']} -> {args.out} "
          f"(completed: {summary['completed']})")
    print(f"  answer yield: {analysis['answer_yield']['rate']:.3f} "
          f"({analysis['answer_yield']['k']}/{analysis['answer_yield']['n']})")
    print(f"  M0 correct (unseeded baseline): "
          f"{analysis['m0_correct']['k']}/{analysis['m0_correct']['n']}  "
          f"M5 correct (seeded): "
          f"{analysis['m5_correct']['k']}/{analysis['m5_correct']['n']}")
    for model, measures in summary["comparison"].items():
        prop = measures["propagation_eligible"]
        empty = prop["empty_store"]
        empty_txt = ("n/a" if empty is None or empty["rate"] is None
                     else f"{empty['rate']:.3f} ({empty['k']}/{empty['n']})")
        pop = prop["populated_store"]
        pop_txt = (f"{pop['rate']:.3f} ({pop['k']}/{pop['n']})"
                   if pop["rate"] is not None else "n/a")
        delta_txt = ("n/a" if prop["delta"] is None
                     else f"{prop['delta']:+.3f}")
        print(f"  {model}: propagation (eligible) empty {empty_txt} "
              f"-> populated {pop_txt} (delta {delta_txt})")
    print("  ARM COMPLETED — a measurement, not a verdict; see "
          "g7_arm_summary.md for the full comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
