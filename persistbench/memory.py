"""Memory policies — Week 1 interface freeze.

Frozen interface (docs/design-freeze.md, Section 5): a policy has ``name``
and ``write(trajectory, chain, world) -> dict | None``, returning the memory
store (a flat fact dict) that later sessions read, or ``None`` for no memory.

Week 1 implements M0, M1 (for scripted trajectories: the agent's own claim,
verbatim — the scripted stand-in for a free-form self-summary) and M5.
M2–M4 semantics are frozen in the design doc; their live-model builds land
in Week 2 and raise ``NotImplementedError`` until then.
"""
from __future__ import annotations

from .chains import resolve_fact


class MemoryPolicy:
    name = "base"

    def write(self, trajectory, chain: dict, world):
        raise NotImplementedError


class M0NoMemory(MemoryPolicy):
    name = "M0"

    def write(self, trajectory, chain: dict, world):
        return None


class M1SelfSummary(MemoryPolicy):
    name = "M1"

    def write(self, trajectory, chain: dict, world):
        # The agent's own end-of-session claim, stored verbatim.
        return dict(trajectory.claim)


class M2Disclosure(MemoryPolicy):
    name = "M2"

    def write(self, trajectory, chain: dict, world):
        raise NotImplementedError("M2 build lands in Week 2 (live-model self-summary + disclosure instruction)")


class M3EvidenceContract(MemoryPolicy):
    name = "M3"

    def write(self, trajectory, chain: dict, world):
        raise NotImplementedError("M3 build lands in Week 2 (structured evidence-contract write)")


class M4StateVerified(MemoryPolicy):
    name = "M4"

    def write(self, trajectory, chain: dict, world):
        raise NotImplementedError("M4 build lands in Week 2 (state-verified write)")


class M5OracleTruth(MemoryPolicy):
    name = "M5"

    def write(self, trajectory, chain: dict, world):
        memory = {key: resolve_fact(world, chain, key) for key in chain["intended_facts"]}
        memory["task_success"] = all(
            memory[key] == value for key, value in chain["intended_facts"].items()
        )
        return memory


POLICIES = {p.name: p for p in (M0NoMemory(), M1SelfSummary(), M5OracleTruth())}
