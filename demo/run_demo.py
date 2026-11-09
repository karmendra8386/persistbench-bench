"""Reproduce the PersistAIBench demo contrast.

Question: when a tool fails silently in Session 1 and the agent writes its
own (false) account of the outcome into memory, do Sessions 2-4 act on the
falsehood — and does the memory-write policy decide whether they do?

Runs the deterministic scripted harness (same chains, faults, policies and
programmatic scorer as the live-model runs): the scripted NaiveAgent works
all 12 chains under two silent faults, its account is written under three
memory policies, and every downstream belief is graded against DB truth.

  M1  unverified self-summary write (the propagation risk)
  M4  state-verified write (falsified fields corrected/rejected at write)
  M5  oracle-truth write (the ceiling: memory is always true)

Writes contrast-results.json next to this script and prints the table.

    python demo/run_demo.py        # from the repo root

This is a scripted-harness run. The live-model propagation smoke (G2b,
72/72, Kaggle T4x2, Nov 7 2026) is recorded in docs/g2b-smoke-results.md —
this demo exercises the same machinery locally; it is not that run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persistbench.agents import NaiveAgent  # noqa: E402
from persistbench.chains import CHAINS  # noqa: E402
from persistbench.faults import SILENT_WRONG_VALUE, STALE_VALUE  # noqa: E402
from persistbench.memory import M1SelfSummary, M4StateVerified, M5OracleTruth  # noqa: E402
from persistbench.runner import run_chain  # noqa: E402

POLICIES = {"M1": M1SelfSummary, "M4": M4StateVerified, "M5": M5OracleTruth}
FAULTS = [SILENT_WRONG_VALUE, STALE_VALUE]
HERE = Path(__file__).resolve().parent

rows = []
for fault in FAULTS:
    for pname, pcls in POLICIES.items():
        false_writes = dep_total = dep_correct = 0
        for chain in CHAINS:
            res = run_chain(chain, NaiveAgent(), fault, pcls(), seed=7)
            if res["write_falsified"]:
                false_writes += 1
            for dep in res["dependents"]:
                dep_total += 1
                dep_correct += 1 if dep["grade"].correct else 0
        rows.append({"fault": fault, "policy": pname, "chains": len(CHAINS),
                     "false_memory_writes": false_writes,
                     "downstream_correct": dep_correct,
                     "downstream_total": dep_total,
                     "downstream_accuracy": round(dep_correct / dep_total, 3)})
        r = rows[-1]
        print(f"{fault:<20} {pname}: false memory written in "
              f"{r['false_memory_writes']}/{r['chains']} chains; "
              f"downstream beliefs correct {dep_correct}/{dep_total} "
              f"({r['downstream_accuracy']:.1%})")

(HERE / "contrast-results.json").write_text(json.dumps(rows, indent=1))
print("\nwrote contrast-results.json")
