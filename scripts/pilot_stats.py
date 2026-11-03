"""Deepened statistics for the Week-2 formation pilot.

The shipped pilot tables (docs/results.md) report pooled rates only.
This script recomputes, from outputs/pilot_cells.jsonl +
outputs/pilot_writes.jsonl + the hand-labelled sample:

- per-model x policy formation with Wilson 95% CIs
- non-empty-only formation (empty stores excluded from denominator)
- write coverage (non-empty writes / cells) per model x policy
- field-level formation (falsified field-instances / stored entries)
- paired McNemar (exact binomial) M1 vs M2/M3/M4 on shared origins,
  Qwen2.5-7B only (the Qwen3-8B arm is a measurement failure)
- Clopper-Pearson bound for M4's zero; rule-of-three bound for the
  classifier gate, overall and on the non-empty hand-label subset
- token totals (RQ5/H5 inputs)

Writes outputs/pilot_stats.json and outputs/pilot_stats.md.
Standard library only.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

QWEN25 = "Qwen/Qwen2.5-7B-Instruct"
QWEN3 = "Qwen/Qwen3-8B"
POLICIES = ["M1", "M2", "M3", "M4"]
FAULTS = ["no_fault", "explicit_error", "empty_partial",
          "silent_wrong_value", "stale_value"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def binom_sf(k, n, p=0.5):
    """P(X >= k) for X ~ Bin(n, p), exact."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) * 0.5 ** n for i in range(0, k + 1))
    return min(1.0, 2 * tail)


def fact_keys(store):
    return [k for k in (store or {}) if not k.startswith("_")]


def main():
    cells = [json.loads(l) for l in open(OUT / "pilot_cells.jsonl")]
    writes = [json.loads(l) for l in open(OUT / "pilot_writes.jsonl")]
    wmap = {(w["model"], w["chain_id"], w["fault_condition"], w["policy"]): w
            for w in writes}
    stats = {}

    # --- per model x policy: headline, coverage, non-empty-only --------
    table = {}
    for model in (QWEN25, QWEN3):
        for pol in POLICIES:
            cs = [c for c in cells if c["model"] == model
                  and c["policy"] == pol and c["formation"] is not None]
            k = sum(c["formation"] for c in cs)
            ne = [c for c in cs
                  if fact_keys(wmap[(c["model"], c["chain_id"],
                                      c["fault_condition"], pol)]
                                ["stored_write"])]
            kne = sum(c["formation"] for c in ne)
            table[f"{model}|{pol}"] = {
                "formation": wilson(k, len(cs)),
                "coverage": (len(ne), len(cs)),
                "formation_nonempty": wilson(kne, len(ne)),
            }
    stats["per_model_policy"] = table

    # --- per fault (Qwen2.5, M1) ---------------------------------------
    fault_rows = {}
    for fault in FAULTS:
        cs = [c for c in cells if c["model"] == QWEN25
              and c["policy"] == "M1" and c["fault_condition"] == fault]
        fault_rows[fault] = wilson(sum(c["formation"] for c in cs), len(cs))
    stats["qwen25_m1_by_fault"] = fault_rows

    # --- field-level formation (Qwen2.5, M1-M3) -------------------------
    fld = {}
    for pol in ["M1", "M2", "M3"]:
        ws = [w for w in writes if w["model"] == QWEN25 and w["policy"] == pol]
        stored = sum(len(fact_keys(w["stored_write"])) for w in ws)
        false_inst = sum(len(w["falsified_fields"]) for w in ws)
        fld[pol] = {"false_instances": false_inst, "stored_entries": stored,
                    "rate": false_inst / stored if stored else None}
    stats["field_level_qwen25"] = fld

    # --- paired McNemar on shared origins (Qwen2.5) ---------------------
    by_origin = defaultdict(dict)
    for c in cells:
        if c["model"] == QWEN25 and c["formation"] is not None:
            by_origin[(c["chain_id"], c["fault_condition"])][c["policy"]] = \
                c["formation"]
    pairs = {}
    for other in ["M2", "M3", "M4"]:
        b = sum(1 for v in by_origin.values()
                if v.get("M1") == 1 and v.get(other) == 0)
        cc = sum(1 for v in by_origin.values()
                 if v.get("M1") == 0 and v.get(other) == 1)
        pairs[f"M1_vs_{other}"] = {"b_M1only": b, "c_otheronly": cc,
                                   "p_exact": mcnemar_exact(b, cc)}
    stats["mcnemar_qwen25"] = pairs

    # --- M4 zero bound ---------------------------------------------------
    m4 = [c for c in cells if c["model"] == QWEN25 and c["policy"] == "M4"]
    n4 = len(m4)
    stats["m4_zero"] = {"n": n4, "clopper_pearson_upper95":
                        1 - 0.05 ** (1 / n4) if n4 else None}

    # --- tokens ----------------------------------------------------------
    toks = {}
    for model in (QWEN25, QWEN3):
        cs = [c for c in cells if c["model"] == model]
        toks[model] = {
            "origin_tokens_total": sum(c["origin_tokens"] for c in cs),
            "write_tokens_total": sum(c["write_tokens"] for c in cs),
            "origin_per_unique_origin":
                sum(c["origin_tokens"] for c in cs
                    if c["policy"] == "M0") / 30,
        }
    stats["tokens"] = toks
    # RQ5 pilot slice: M4 vs M1 write-token totals, Qwen2.5
    for pol in POLICIES:
        toks[f"{QWEN25}|{pol}_write"] = sum(
            c["write_tokens"] for c in cells
            if c["model"] == QWEN25 and c["policy"] == pol)

    # --- hand-label re-audit ---------------------------------------------
    rows = list(csv.DictReader(open(OUT / "handlabel_sample_handlabeled.csv")))
    def nonempty(r):
        s = json.loads(r["stored_write_json"] or "{}")
        return len(fact_keys(s)) > 0
    ne_rows = [r for r in rows if nonempty(r)]
    def agree(rs):
        a = sum(1 for r in rs
                if int(r["hand_label_false_memory"])
                == int(r["auto_label_false_memory"]))
        return a, len(rs)
    stats["handlabel"] = {
        "overall": agree(rows),
        "nonempty_only": agree(ne_rows),
        "empty_rows": len(rows) - len(ne_rows),
        "rule_of_three_upper_nonempty": 3 / len(ne_rows) if ne_rows else None,
    }

    (OUT / "pilot_stats.json").write_text(json.dumps(stats, indent=1,
                                                     default=str))
    # --- markdown ---------------------------------------------------------
    L = ["# Pilot deepened statistics (scripts/pilot_stats.py)", ""]
    L.append("## Per model x policy")
    L.append("| Model | Policy | Formation (Wilson 95%) | Coverage | "
             "Formation, non-empty only |")
    L.append("|---|---|---|---|---|")
    for model in (QWEN25, QWEN3):
        for pol in POLICIES:
            t = table[f"{model}|{pol}"]
            p, lo, hi = t["formation"]
            pne, lone, hine = t["formation_nonempty"]
            cov_n, cov_d = t["coverage"]
            ne_str = ("n/a (no non-empty writes)" if pne is None else
                      f"{pne:.3f} [{lone:.3f}, {hine:.3f}]")
            L.append(f"| {model.split('/')[-1]} | {pol} | "
                     f"{p:.3f} [{lo:.3f}, {hi:.3f}] | {cov_n}/{cov_d} | "
                     f"{ne_str} |")
    L += ["", "## Qwen2.5-7B M1 by fault",
          "| Fault | Formation (Wilson 95%) |", "|---|---|"]
    for f, (p, lo, hi) in fault_rows.items():
        L.append(f"| {f} | {p:.3f} [{lo:.3f}, {hi:.3f}] |")
    L += ["", "## Paired McNemar (Qwen2.5-7B, shared origins)"]
    for name, v in pairs.items():
        L.append(f"- {name}: b={v['b_M1only']}, c={v['c_otheronly']}, "
                 f"exact p={v['p_exact']:.3g}")
    L += ["", f"M4 zero: 0/{n4}, Clopper-Pearson 95% upper "
          f"{stats['m4_zero']['clopper_pearson_upper95']:.3f} "
          "(and zero by construction — see audit).",
          "", "## Field-level (Qwen2.5-7B)"]
    for pol, v in fld.items():
        L.append(f"- {pol}: {v['false_instances']}/{v['stored_entries']} "
                 f"stored entries false = {v['rate']:.3f}")
    hl = stats["handlabel"]
    L += ["", "## Hand-label gate re-audit",
          f"- Overall: {hl['overall'][0]}/{hl['overall'][1]}; "
          f"empty/trivial rows: {hl['empty_rows']}",
          f"- Non-empty only: {hl['nonempty_only'][0]}/"
          f"{hl['nonempty_only'][1]}; rule-of-three 95% upper on "
          f"disagreement {hl['rule_of_three_upper_nonempty']:.3f}",
          "", "## Tokens"]
    for model in (QWEN25, QWEN3):
        t = toks[model]
        L.append(f"- {model.split('/')[-1]}: origin total "
                 f"{t['origin_tokens_total']:,}, write total "
                 f"{t['write_tokens_total']:,}, origin per unique origin "
                 f"{t['origin_per_unique_origin']:,.0f}")
    (OUT / "pilot_stats.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
