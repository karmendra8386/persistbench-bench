"""Gate G6 eligibility filter (persistbench/eligibility.py).

Unit tests pin the F.14 semantics on synthetic cells: a session is
eligible iff the M0 and M5 outcomes for that same session are both
correct; eligible rates exclude ineligible sessions while raw rates
keep them; persistence / correction-resistance pairs require
eligibility at both sessions of the pair plus an incorrect M1
outcome at the first.

Reproduction tests run the frozen G2 and G2b smoke cells through
the formal module and require the recorded summaries back exactly —
the filter was extracted from the smoke analysis, and these tests
prove the extraction changed nothing.
"""
import json
from pathlib import Path

import pytest

from persistbench.eligibility import (
    contrast_groups,
    eligibility_map,
    model_rates,
    propagation_report,
    rate,
    wilson,
)
from persistbench.g2_smoke import analyse_cells

REPO = Path(__file__).resolve().parent.parent
MODEL = "model-a"
OTHER = "model-b"
CHAINS = ["chain-a", "chain-b"]
FAULTS = ["fault-a"]


def _cell(model, chain, fault, policy, session, correct):
    return {"model": model, "chain_id": chain,
            "fault_condition": fault, "policy": policy,
            "session": session, "correct": correct,
            "parseable": True, "error": None,
            "correction_in_prompt": True}


def _group(model, chain, fault, m0, m5, m1):
    """Cells for one (model, chain, fault) group.

    ``m0`` / ``m5`` / ``m1`` map session -> correct; a session absent
    from a map produces no cell for that policy.
    """
    cells = []
    for policy, table in (("M0", m0), ("M5", m5), ("M1", m1)):
        for session, correct in table.items():
            cells.append(_cell(model, chain, fault, policy, session,
                               correct))
    return cells


_ALL = {2: True, 3: True, 4: True}


# ---------------------------------------------------------------------------
# Wilson / rate anchors
# ---------------------------------------------------------------------------

def test_wilson_matches_recorded_smoke_values():
    # The Qwen3 G2/G2b propagation row of the recorded summaries.
    assert wilson(4, 12) == (0.3333333333333333,
                             0.13812009012485987,
                             0.6093779130595061)
    assert wilson(0, 0) == (None, None, None)
    assert rate(0, 0) == {"k": 0, "n": 0, "rate": None,
                          "ci_lo": None, "ci_hi": None}


# ---------------------------------------------------------------------------
# Eligibility semantics
# ---------------------------------------------------------------------------

def test_eligibility_requires_both_baselines_correct():
    cells = (
        _group(MODEL, "chain-a", "fault-a", _ALL, _ALL, _ALL)
        # chain-b: M5 wrong at S2, M0 wrong at S3, both right at S4.
        + _group(MODEL, "chain-b", "fault-a",
                 {2: True, 3: False, 4: True},
                 {2: False, 3: True, 4: True},
                 _ALL))
    eligible = eligibility_map(cells, [MODEL], CHAINS, FAULTS)
    assert eligible[(MODEL, "chain-a", "fault-a", 2)] is True
    assert eligible[(MODEL, "chain-a", "fault-a", 3)] is True
    assert eligible[(MODEL, "chain-a", "fault-a", 4)] is True
    assert eligible[(MODEL, "chain-b", "fault-a", 2)] is False
    assert eligible[(MODEL, "chain-b", "fault-a", 3)] is False
    assert eligible[(MODEL, "chain-b", "fault-a", 4)] is True
    groups = contrast_groups(eligible, [MODEL], CHAINS, FAULTS)
    assert groups == {"model-a|chain-a|fault-a": [2, 3, 4],
                      "model-a|chain-b|fault-a": [4]}


def test_eligibility_missing_baseline_row_is_ineligible():
    # No M5 row at Session 2: not eligible, however good M0 looks.
    cells = _group(MODEL, "chain-a", "fault-a", _ALL,
                   {3: True, 4: True}, _ALL)
    eligible = eligibility_map(cells, [MODEL], ["chain-a"], FAULTS)
    assert eligible[(MODEL, "chain-a", "fault-a", 2)] is False
    assert eligible[(MODEL, "chain-a", "fault-a", 3)] is True


def test_ineligible_sessions_leave_eligible_rates_but_stay_in_raw():
    cells = (
        # chain-a: fully eligible; M1 wrong at S2 only.
        _group(MODEL, "chain-a", "fault-a", _ALL, _ALL,
               {2: False, 3: True, 4: True})
        # chain-b: S2 ineligible (M5 wrong); M1 wrong at S2 and S3.
        + _group(MODEL, "chain-b", "fault-a", _ALL,
                 {2: False, 3: True, 4: True},
                 {2: False, 3: False, 4: True}))
    report = propagation_report(cells, [MODEL], CHAINS, FAULTS)
    row = report["per_model"][MODEL]
    # Eligible M1 rows: chain-a x3 + chain-b S3,S4 = 5; wrong: a-S2, b-S3.
    assert row["propagation_eligible"]["k"] == 2
    assert row["propagation_eligible"]["n"] == 5
    # Raw M1 rows: all 6; wrong: a-S2, b-S2, b-S3.
    assert row["propagation_raw"]["k"] == 3
    assert row["propagation_raw"]["n"] == 6
    assert row["eligible_sessions"] == 5


def test_persistence_and_correction_pairing_rules():
    chains = ["c1", "c2"]
    faults = ["f1", "f2"]
    cells = []
    # (c1, f1): eligible throughout; M1 wrong at 2, 3, 4
    #   -> persistence 1/1, correction 1/1.
    cells += _group(MODEL, "c1", "f1", _ALL, _ALL,
                    {2: False, 3: False, 4: False})
    # (c1, f2): eligible throughout; M1 wrong at S2, right after
    #   -> persistence 0/1; correction excluded (S3 correct).
    cells += _group(MODEL, "c1", "f2", _ALL, _ALL,
                    {2: False, 3: True, 4: True})
    # (c2, f1): S3 ineligible (M5 wrong); M1 wrong at S2 and S3
    #   -> both pairs excluded.
    cells += _group(MODEL, "c2", "f1", _ALL,
                    {2: True, 3: False, 4: True},
                    {2: False, 3: False, 4: True})
    # (c2, f2): eligible throughout; M1 right at S2, wrong at S3,
    #   right at S4 -> persistence excluded (S2 correct),
    #   correction 0/1.
    cells += _group(MODEL, "c2", "f2", _ALL, _ALL,
                    {2: True, 3: False, 4: True})
    report = propagation_report(cells, [MODEL], chains, faults)
    row = report["per_model"][MODEL]
    assert row["persistence_s2_to_s3"]["k"] == 1
    assert row["persistence_s2_to_s3"]["n"] == 2
    assert row["correction_resistance_s3_to_s4"]["k"] == 1
    assert row["correction_resistance_s3_to_s4"]["n"] == 2
    # 12 (group, session) pairs, one ineligible: (c2, f1) at S3.
    assert row["eligible_sessions"] == 11
    assert row["eligible_groups"] == 4


def test_rates_are_per_model_never_pooled():
    cells = (
        _group(MODEL, "chain-a", "fault-a", _ALL, _ALL,
               {2: False, 3: False, 4: False})
        + _group(OTHER, "chain-a", "fault-a", _ALL, _ALL,
                 {2: True, 3: True, 4: True}))
    report = propagation_report(cells, [MODEL, OTHER], ["chain-a"],
                                FAULTS)
    assert list(report["per_model"]) == [MODEL, OTHER]
    assert report["per_model"][MODEL]["propagation_raw"]["k"] == 3
    assert report["per_model"][OTHER]["propagation_raw"]["k"] == 0
    # model_rates on its own sees only its own model's rows.
    eligible = eligibility_map(cells, [MODEL, OTHER], ["chain-a"],
                               FAULTS)
    groups = contrast_groups(eligible, [MODEL, OTHER], ["chain-a"],
                             FAULTS)
    row = model_rates(cells, eligible, groups, MODEL, ["chain-a"],
                      FAULTS)
    assert row["propagation_raw"]["n"] == 3


# ---------------------------------------------------------------------------
# Reproduction of the frozen smoke analyses
# ---------------------------------------------------------------------------

# Headline numbers from the recorded summaries (the full per-model
# dicts and contrast groups are also compared verbatim below).
SMOKES = {
    "g2-smoke": {
        "Qwen/Qwen2.5-7B-Instruct": (5, 0, 5),
        "Qwen/Qwen3-8B": (12, 4, 12),
    },
    "g2b-smoke": {
        "Qwen/Qwen2.5-7B-Instruct": (12, 2, 12),
        "Qwen/Qwen3-8B": (12, 4, 12),
    },
}


def _load_smoke(dirname):
    out = REPO / "results" / dirname
    cells = [json.loads(line)
             for line in (out / "g2_smoke_cells.jsonl")
             .read_text().splitlines() if line.strip()]
    summary = json.loads((out / "g2_smoke_summary.json").read_text())
    return cells, summary


@pytest.mark.parametrize("dirname", sorted(SMOKES))
def test_frozen_smoke_reproduced_through_formal_module(dirname):
    cells, summary = _load_smoke(dirname)
    config = summary["config"]
    recorded = summary["analysis"]
    report = propagation_report(cells, config["models"],
                                config["chains"],
                                config["fault_conditions"])
    assert report["contrast_groups"] == recorded["contrast_groups"]
    assert report["per_model"] == recorded["per_model"]
    for model, (sessions, k, n) in SMOKES[dirname].items():
        row = report["per_model"][model]
        assert row["eligible_sessions"] == sessions
        assert row["propagation_eligible"]["k"] == k
        assert row["propagation_eligible"]["n"] == n


@pytest.mark.parametrize("dirname", sorted(SMOKES))
def test_frozen_smoke_full_analysis_unchanged_after_refactor(dirname):
    # The whole analyse_cells output — gates included — still equals
    # the recorded analysis on the frozen cells.
    cells, summary = _load_smoke(dirname)
    assert analyse_cells(cells, summary["config"]) == summary["analysis"]
