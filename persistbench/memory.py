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
    """Structured evidence-contract write (design freeze, Section 5).

    Every claim in the agent's summary becomes a structured claim with a
    status and an evidence pointer into the trajectory's tool responses:

    - ``verified`` — a tool observation supports the claimed value. The
      value is written to the fact store.
    - ``failed`` — the claim is contradicted by its evidence: a fact claim
      whose observations all disagree, or a success claim made after an
      observed tool error. For ``task_success`` the store records False
      (the outcome the evidence supports); a contradicted fact value is
      not written as fact.
    - ``unknown`` — no tool observation supports the claim. It is stored
      only in the claims record, never in the fact store and never as
      verified — the contract's core rule.

    Evidence means tool observations, never DB state (that is M4). Under
    silent/stale faults the corrupted response is itself the supporting
    observation, so the false claim is verified here by construction —
    the documented boundary of the contract, and the reason H3 orders
    M3 below M4.
    """

    name = "M3"

    def write(self, trajectory, chain: dict, world):
        claim = dict(trajectory.claim)
        observations = observed_values(trajectory)
        errors = observed_errors(trajectory)
        claims = []
        store: dict = {}

        fact_keys = [k for k in claim if k != "task_success"]
        for key in fact_keys:
            value = claim[key]
            seen = observations.get(key, [])
            supporting = [o for o in seen if o["value"] == value]
            if supporting:
                status = "verified"
                evidence = {"step": supporting[0]["step"],
                            "tool": supporting[0]["tool"],
                            "field": supporting[0]["field"]}
                store[key] = value
                stored = True
            elif seen:
                status = "failed"
                evidence = {"step": seen[0]["step"], "tool": seen[0]["tool"],
                            "field": seen[0]["field"]}
                stored = False
            else:
                status = "unknown"
                evidence = None
                stored = False
            claims.append({
                "fact": key, "claimed": value, "status": status,
                "evidence": evidence, "stored": stored,
                "stored_value": value if stored else None,
            })

        if "task_success" in claim:
            claimed_success = claim["task_success"]
            facts_verified = all(c["status"] == "verified" for c in claims)
            if not claimed_success:
                status, stored_value = "failed", False
                evidence = None
            elif errors:
                status, stored_value = "failed", False
                evidence = {"step": errors[0]["step"], "tool": errors[0]["tool"],
                            "error": errors[0]["error"]}
            elif facts_verified:
                status, stored_value = "verified", True
                evidence = {"basis": "all fact claims verified"}
            else:
                status, stored_value = "unknown", None
                evidence = None
            if stored_value is not None:
                store["task_success"] = stored_value
            claims.append({
                "fact": "task_success", "claimed": claimed_success,
                "status": status, "evidence": evidence,
                "stored": stored_value is not None,
                "stored_value": stored_value,
            })

        store["_claims"] = claims
        entries = [{
            "fact": c["fact"], "claimed": c["claimed"],
            "stored_value": c["stored_value"], "stored": c["stored"],
            "status": c["status"], "evidence": c["evidence"],
        } for c in claims]
        return self._record(trajectory, chain, entries, store)


class M4StateVerified(MemoryPolicy):
    """State-verified write (design freeze, Section 5): the system-side
    guard. Every claimed entry is checked against the world DB end-state
    (``resolve_fact``) before storage:

    - claimed == DB state -> ``accepted``, stored as claimed;
    - claimed != DB state, DB holds a value -> ``corrected`` to the DB
      value, stored with provenance ``state-verified``;
    - claimed != DB state, DB holds no such fact -> ``rejected`` (nothing
      stored);
    - a chain fact the claim omitted -> ``generated`` from the DB;
    - a claimed key that is neither a chain fact nor ``task_success`` ->
      ``rejected`` (not a fact this world can verify).

    Nothing contradicting the DB is ever stored, so fault-derived false
    memories are eliminated by construction; the price is exactly the
    correction/rejection/generation accounting in the write log.
    """

    name = "M4"

    def write(self, trajectory, chain: dict, world):
        claim = dict(trajectory.claim)
        truth = {key: resolve_fact(world, chain, key)
                 for key in chain["intended_facts"]}
        truth["task_success"] = all(
            truth[key] == value for key, value in chain["intended_facts"].items()
        )
        store: dict = {}
        entries = []
        for key, value in claim.items():
            if key not in truth:
                entries.append({
                    "fact": key, "claimed": value, "stored_value": None,
                    "stored": False, "action": "rejected",
                    "provenance": "state-verified",
                })
                continue
            db_value = truth[key]
            if value == db_value:
                action = "accepted"
            elif db_value is None:
                action = "rejected"
            else:
                action = "corrected"
            stored = db_value is not None and action != "rejected"
            if stored:
                store[key] = db_value
            entries.append({
                "fact": key, "claimed": value,
                "stored_value": db_value if stored else None,
                "stored": stored, "action": action,
                "provenance": "state-verified",
            })
        for key, db_value in truth.items():
            if key in claim or db_value is None:
                continue
            store[key] = db_value
            entries.append({
                "fact": key, "claimed": None, "stored_value": db_value,
                "stored": True, "action": "generated",
                "provenance": "state-verified",
            })
        return self._record(trajectory, chain, entries, store)


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


POLICIES = {p.name: p for p in (
    M0NoMemory(), M1SelfSummary(), M2Disclosure(),
    M3EvidenceContract(), M4StateVerified(), M5OracleTruth(),
)}
