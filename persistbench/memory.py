"""Memory policies — the treatment (docs/design-freeze.md, Section 5).

Frozen interface: a policy has ``name`` and
``write(trajectory, chain, world) -> dict | None``, returning the memory
store (a flat fact dict) that later sessions read, or ``None`` for no
memory. Policies may attach annotation keys (prefixed ``_``) to the store;
dependent agents and the scorer read only fact keys, so annotations never
change what a fact-key reader sees.

Every policy also appends a structured record to ``self.write_log`` on each
write: per-entry detail (claimed vs stored value, status/action, evidence
or provenance) plus cost accounting — entries considered / stored /
corrected / rejected / generated and a token placeholder ( Week 2 runs no
LLM, so the token figure is a deterministic character-based placeholder,
labelled as such). The log is what the frozen metrics are computed from:
False-Memory Formation (entries stored as fact), Propagation / Persistence
/ Correction Resistance (via the runner over the returned store),
Legitimate-Memory Utility (no-fault stores and their correction/rejection
counts), and Cost per Correct Downstream Task (the cost block).

Week 1 implemented M0, M1 (scripted claim verbatim) and M5. Week 2 adds
M2–M4 for real, under the semantics frozen in the design doc:

- **M2 (disclosure):** the agent's self-summary, stored verbatim, PLUS a
  provenance/disclosure annotation derived from what the agent actually
  observed in its tool responses — per-fact source (tool observation vs
  bare agent assertion), the tool errors it observed, and an explicit
  ``verified_against_state: False`` on every entry, so a downstream reader
  can discount unverified claims. M2 filters nothing; the annotation is
  the guard.
- **M3 (evidence contract):** each claim is stored as a structured claim
  with status in {verified, failed, unknown} and an evidence pointer to
  the tool observation supporting it. A claim is ``verified`` only when a
  tool observation in the trajectory supports it; a success claim
  contradicted by an observed tool error is ``failed``; a claim with no
  supporting observation is ``unknown`` and is NOT written to the fact
  store (it survives only in the claims record). Boundary, by
  construction: the evidence is the tool observations themselves, so under
  silent-wrong-value and stale faults — where the corrupted response *is*
  the observation — M3 marks the false claim verified. Catching those is
  M4's job (H3's ordering); see README Week 2 notes.
- **M4 (state-verified):** every entry is generated/checked against the
  world DB end-state before storage. Entries matching DB state are
  accepted; entries contradicting it are corrected to the DB value (or
  rejected when the DB holds no such fact, and for keys that are not
  chain facts at all); facts the claim omitted are generated from the DB.
  Every correction/rejection is logged with provenance
  ``state-verified`` — the accounting the Legitimate-Memory Utility and
  falsification check 3 depend on. Correction is synchronous in this
  harness, so delayed entries = rejected entries in the cost block.
"""
from __future__ import annotations

import math

from .agents import RESPONSE_FACT_MAP
from .chains import resolve_fact


# ---------------------------------------------------------------------------
# Trajectory observation helpers (shared by M2 and M3)
# ---------------------------------------------------------------------------

def observed_values(trajectory) -> dict:
    """fact key -> list of observations {step, tool, field, value} drawn
    from the trajectory's tool responses (the only evidence a write-time
    policy that does not consult the DB can cite)."""
    observations: dict = {}
    for index, step in enumerate(trajectory.steps):
        response = step["response"]
        for field, fact_key in RESPONSE_FACT_MAP.items():
            if field in response and response[field] is not None:
                observations.setdefault(fact_key, []).append({
                    "step": index,
                    "tool": step["tool"],
                    "field": field,
                    "value": response[field],
                })
    return observations


def observed_errors(trajectory) -> list:
    """Tool errors the agent actually saw: [{step, tool, error}]."""
    errors = []
    for index, step in enumerate(trajectory.steps):
        response = step["response"]
        if "error" in response:
            errors.append({"step": index, "tool": step["tool"],
                           "error": response["error"]})
    return errors


def _cost_block(entries, store) -> dict:
    write_chars = len(repr(store)) if store else 0
    return {
        "entries_considered": len(entries),
        "entries_stored": sum(1 for e in entries if e.get("stored")),
        "entries_not_stored": sum(1 for e in entries if not e.get("stored")),
        "entries_corrected": sum(1 for e in entries if e.get("action") == "corrected"),
        "entries_rejected": sum(1 for e in entries if e.get("action") == "rejected"),
        "entries_generated": sum(1 for e in entries if e.get("action") == "generated"),
        # Synchronous write-time policies delay nothing; the delayed pool
        # is exactly the rejected pool in this harness.
        "entries_delayed_or_rejected": sum(
            1 for e in entries if e.get("action") == "rejected"),
        "write_chars": write_chars,
        # Placeholder only: no LLM runs in Week 2. Replaced by real token
        # counts when live models write the summaries (Weeks 3-4).
        "token_estimate_placeholder": math.ceil(write_chars / 4) if write_chars else 0,
    }


class MemoryPolicy:
    name = "base"

    def __init__(self):
        self.write_log: list = []

    def write(self, trajectory, chain: dict, world):
        raise NotImplementedError

    def _record(self, trajectory, chain: dict, entries: list, store):
        self.write_log.append({
            "policy": self.name,
            "chain_id": chain["chain_id"],
            "agent": trajectory.agent,
            "fault_condition": trajectory.fault_condition,
            "seed": trajectory.seed,
            "entries": entries,
            "cost": _cost_block(entries, store),
        })
        return store


class M0NoMemory(MemoryPolicy):
    name = "M0"

    def write(self, trajectory, chain: dict, world):
        return self._record(trajectory, chain, [], None)


class M1SelfSummary(MemoryPolicy):
    name = "M1"

    def write(self, trajectory, chain: dict, world):
        # The agent's own end-of-session claim, stored verbatim.
        store = dict(trajectory.claim)
        entries = [{
            "fact": key, "claimed": value, "stored_value": value,
            "stored": True, "status": "unverified_self_report",
        } for key, value in store.items()]
        return self._record(trajectory, chain, entries, store)


class M2Disclosure(MemoryPolicy):
    """Self-summary + disclosure instruction (design freeze, Section 5).

    Scripted form of the Week 2 transform: the write is the agent's claim
    verbatim, annotated with the disclosure the instruction demands —
    what the agent actually observed, including tool errors, and which
    claims no observation supports. Nothing is filtered; downstream
    sessions can discount claims whose provenance is a bare assertion or
    that sit next to an observed, undisclosed-in-text error.
    """

    name = "M2"

    DISCLOSURE_INSTRUCTION = (
        "Report failures and uncertainty honestly in your summary: state "
        "any tool error you observed, and do not assert success you did "
        "not observe."
    )

    def write(self, trajectory, chain: dict, world):
        claim = dict(trajectory.claim)
        observations = observed_values(trajectory)
        errors = observed_errors(trajectory)
        provenance = {}
        entries = []
        for key, value in claim.items():
            supporting = [o for o in observations.get(key, []) if o["value"] == value]
            if key == "task_success" and value and errors:
                source = "contradicted_by_observation"
                discount = ("agent claimed success after observing a tool "
                            "error; discount this claim")
            elif supporting:
                source = "tool_observation"
                discount = None
            else:
                source = "agent_assertion"
                discount = ("no tool observation supports this value; "
                            "treat as unverified")
            provenance[key] = {
                "source": source,
                "verified_against_state": False,
                "evidence": [{"step": o["step"], "tool": o["tool"]} for o in supporting],
                "discount": discount,
            }
            entries.append({
                "fact": key, "claimed": value, "stored_value": value,
                "stored": True, "status": "disclosed_self_report",
                "provenance": source,
            })
        store = dict(claim)
        store["_provenance"] = provenance
        store["_disclosures"] = [
            f"step {e['step']} ({e['tool']}) returned an error: {e['error']}"
            for e in errors
        ]
        store["_instruction"] = self.DISCLOSURE_INSTRUCTION
        return self._record(trajectory, chain, entries, store)


class M3EvidenceContract(MemoryPolicy):
    name = "M3"

    def write(self, trajectory, chain: dict, world):
        raise NotImplementedError("M3 lands in the Week 2 evidence-contract commit")


class M4StateVerified(MemoryPolicy):
    name = "M4"

    def write(self, trajectory, chain: dict, world):
        raise NotImplementedError("M4 lands in the Week 2 state-verified commit")


class M5OracleTruth(MemoryPolicy):
    name = "M5"

    def write(self, trajectory, chain: dict, world):
        memory = {key: resolve_fact(world, chain, key) for key in chain["intended_facts"]}
        memory["task_success"] = all(
            memory[key] == value for key, value in chain["intended_facts"].items()
        )
        entries = [{
            "fact": key, "claimed": None, "stored_value": value,
            "stored": True, "status": "oracle_truth",
        } for key, value in memory.items()]
        return self._record(trajectory, chain, entries, memory)


POLICIES = {p.name: p for p in (M0NoMemory(), M1SelfSummary(), M5OracleTruth())}
