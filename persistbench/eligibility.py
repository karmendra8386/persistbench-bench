"""Gate G6 — the eligibility filter for propagation contrasts.

Definition (preregistration amendment, section F.14 — "Eligibility
filter for propagation (MemForm-style)"):

    H2 contrasts are computed on the eligible subset: origins where
    the no-memory (M0) and clean/oracle (M5) baselines both succeed
    downstream, so origin-task difficulty cannot masquerade as
    propagation.

Operationally, eligibility is per (model, chain, fault, session):
the session is eligible iff the M0 and M5 dependent outcomes for
that same session are both correct. A session whose no-memory
baseline already fails downstream cannot distinguish "the false
memory propagated" from "this dependent task is hard", and a
session the verified-write baseline fails is not a clean contrast
either — both are excluded from the eligible rates (they remain in
the raw, unfiltered rates reported alongside).

Reported rates (per model only, never pooled), all with Wilson 95%
CIs:

- propagation (eligible / raw): share of M1 dependent outcomes
  incorrect — on the eligible subset, and on all M1 outcomes.
- persistence S2 -> S3: of the (chain, fault) pairs eligible at
  both sessions with the M1 outcome incorrect at Session 2, the
  share still incorrect at Session 3.
- correction resistance S3 -> S4: of the pairs eligible at both
  sessions with the M1 outcome incorrect at Session 3, the share
  still incorrect at Session 4, after the verified-observation
  correction event.

This module is the formal home of the filter. It was extracted,
semantics unchanged, from the Gate G2 smoke analysis
(``persistbench.g2_smoke.analyse_cells``), which now calls
:func:`propagation_report`; the frozen G2 / G2b smoke outputs are
reproduced exactly through it (``tests/test_eligibility.py``).
"""
from __future__ import annotations

import math

#: Dependent sessions scored in a chain run (Session 1 is the origin).
SESSIONS = (2, 3, 4)

#: The two baselines that must both succeed for a session to count.
BASELINE_POLICIES = ("M0", "M5")

#: The policy whose downstream outcomes the contrast is about.
TARGET_POLICY = "M1"


def wilson(k: int, n: int, z: float = 1.959964):
    """Wilson 95% CI for k/n. Same formula as scripts/pilot_stats.py."""
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def rate(k: int, n: int) -> dict:
    """A ``{"k", "n", "rate", "ci_lo", "ci_hi"}`` row from :func:`wilson`."""
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "ci_lo": lo, "ci_hi": hi}


def index_cells(cells: list) -> dict:
    """Index outcome rows by (model, chain, fault, policy, session).

    The first row wins on a duplicate key, matching the smoke
    analysis. Rows are the per-outcome dicts the runners write
    (``model``, ``chain_id``, ``fault_condition``, ``policy``,
    ``session``, ``correct``, ...).
    """
    by_key = {}
    for cell in cells:
        by_key.setdefault(
            (cell["model"], cell["chain_id"], cell["fault_condition"],
             cell["policy"], cell["session"]), cell)
    return by_key


def eligibility_map(cells: list, models, chains, fault_conditions,
                    sessions=SESSIONS) -> dict:
    """The G6 filter: {(model, chain, fault, session): bool}.

    A session is eligible iff the M0 and M5 outcome rows for that
    same (model, chain, fault, session) both exist and are both
    correct.
    """
    by_key = index_cells(cells)

    def outcome(model, chain, fault, policy, session):
        return by_key.get((model, chain, fault, policy, session))

    eligible = {}
    for model in models:
        for chain in chains:
            for fault in fault_conditions:
                for session in sessions:
                    m0 = outcome(model, chain, fault, "M0", session)
                    m5 = outcome(model, chain, fault, "M5", session)
                    eligible[(model, chain, fault, session)] = bool(
                        m0 and m5 and m0.get("correct") and m5.get("correct"))
    return eligible


def contrast_groups(eligible: dict, models, chains, fault_conditions,
                    sessions=SESSIONS) -> dict:
    """Eligible sessions per group: {"model|chain|fault": [sessions]}."""
    groups = {}
    for model in models:
        for chain in chains:
            for fault in fault_conditions:
                groups[f"{model}|{chain}|{fault}"] = [
                    s for s in sessions if eligible[(model, chain, fault, s)]]
    return groups


def model_rates(cells: list, eligible: dict, groups: dict, model: str,
                chains, fault_conditions, sessions=SESSIONS) -> dict:
    """The per-model reported rates (eligible + raw) for one model.

    ``groups`` is the output of :func:`contrast_groups`. Persistence
    pairs require eligibility at both Sessions 2 and 3 and an
    incorrect M1 outcome at Session 2; correction resistance pairs
    require eligibility at both Sessions 3 and 4 and an incorrect
    M1 outcome at Session 3.
    """
    by_key = index_cells(cells)

    def outcome(chain, fault, policy, session):
        return by_key.get((model, chain, fault, policy, session))

    def elig_m1(session=None):
        rows = []
        for chain in chains:
            for fault in fault_conditions:
                for s in sessions:
                    if session is not None and s != session:
                        continue
                    if not eligible[(model, chain, fault, s)]:
                        continue
                    cell = outcome(chain, fault, "M1", s)
                    if cell is not None:
                        rows.append(cell)
        return rows

    elig_all = elig_m1()
    raw_all = [c for c in cells
               if c["model"] == model and c["policy"] == "M1"]
    # Persistence S2 -> S3: incorrect at S2, still incorrect at S3.
    pers_num, pers_den = 0, 0
    corr_num, corr_den = 0, 0
    for chain in chains:
        for fault in fault_conditions:
            s2 = outcome(chain, fault, "M1", 2)
            s3 = outcome(chain, fault, "M1", 3)
            s4c = outcome(chain, fault, "M1", 4)
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
    return {
        "eligible_sessions": sum(1 for (m, _, _, _), v in eligible.items()
                                 if m == model and v),
        "eligible_groups": sum(
            1 for k, v in groups.items()
            if k.startswith(model + "|") and v),
        "propagation_eligible": rate(
            sum(1 for c in elig_all if not c.get("correct")), len(elig_all)),
        "propagation_raw": rate(
            sum(1 for c in raw_all if not c.get("correct")), len(raw_all)),
        "persistence_s2_to_s3": rate(pers_num, pers_den),
        "correction_resistance_s3_to_s4": rate(corr_num, corr_den),
    }


def propagation_report(cells: list, models, chains,
                       fault_conditions, sessions=SESSIONS) -> dict:
    """Eligibility map, contrast groups and per-model rates in one call.

    Returns ``{"eligible": ..., "contrast_groups": ..., "per_model":
    ...}`` with per-model rows keyed by model id, in ``models``
    order. This is the whole G6-filtered half of the smoke analysis.
    """
    eligible = eligibility_map(cells, models, chains, fault_conditions,
                               sessions)
    groups = contrast_groups(eligible, models, chains, fault_conditions,
                             sessions)
    per_model = {
        model: model_rates(cells, eligible, groups, model, chains,
                           fault_conditions, sessions)
        for model in models
    }
    return {"eligible": eligible, "contrast_groups": groups,
            "per_model": per_model}
