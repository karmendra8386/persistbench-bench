"""Scripted agents for Week 1 harness validation.

Week 1 has no LLM runs. Two scripted origin agents exercise the harness:

- ``OracleAgent`` builds its end-of-session claim by re-reading database
  ground truth, so its claim is truthful by construction, under every fault.
- ``NaiveAgent`` is the optimistic summariser: it records what it set out to
  do (the plan's intended facts), overwritten by whatever the tool responses
  reported, and never verifies against the DB. It models the false-success
  behaviour FTA measured at response time, in scripted form.

A scripted ``DependentAgent`` resolves later-session tasks from a memory
dict, from a correction event, or — with no memory — by querying the world.
"""
from __future__ import annotations

from .chains import resolve_fact

# Tool-response field -> fact key, used by the naive agent's claim builder.
RESPONSE_FACT_MAP = {
    "tracking": "tracking",
    "order_status": "order_status",
    "payment_status": "payment_status",
    "amount_cents": "charged_cents",
    "refund_amount_cents": "refund_cents",
    "address": "address",
    "stock": "stock",
    "cancellation_fee_cents": "cancellation_fee_cents",
}


class OracleAgent:
    name = "oracle"

    def build_claim(self, chain: dict, steps: list, world) -> dict:
        claim = {key: resolve_fact(world, chain, key) for key in chain["intended_facts"]}
        claim["task_success"] = all(
            claim[key] == value for key, value in chain["intended_facts"].items()
        )
        return claim


class NaiveAgent:
    name = "naive"

    def build_claim(self, chain: dict, steps: list, world) -> dict:
        claim = dict(chain["intended_facts"])
        claim["task_success"] = True
        for step in steps:
            response = step["response"]
            for response_field, fact_key in RESPONSE_FACT_MAP.items():
                if response_field in response and response[response_field] is not None:
                    claim[fact_key] = response[response_field]
        return claim


class DependentAgent:
    """Resolves one dependent task.

    ``memory`` is the fact dict carried from Session 1 (``None`` = M0, the
    agent queries the world directly). ``correction`` is the verified
    observation delivered before Session 4; a ``revising`` agent adopts it
    for the chain's primary fact, a ``stubborn`` agent ignores it.
    """

    def __init__(self, memory: dict | None, mode: str = "revising"):
        if mode not in ("revising", "stubborn"):
            raise ValueError(f"unknown dependent-agent mode {mode!r}")
        self.memory = memory
        self.mode = mode

    def belief(self, chain: dict, world, fact: str, correction: dict | None):
        if (
            correction
            and self.mode == "revising"
            and fact == chain["primary_fact"]
            and fact in correction
        ):
            return correction[fact]
        if self.memory is not None and fact in self.memory:
            return self.memory[fact]
        return resolve_fact(world, chain, fact)

    def run(self, chain: dict, world, spec: dict, correction: dict | None = None) -> dict:
        fact = spec["fact"]
        believed = self.belief(chain, world, fact, correction if spec["session"] == 4 else None)
        if spec["kind"] == "report":
            return {"answer": believed}
        if spec["kind"] == "decision":
            return {"answer": believed >= spec["threshold"]}
        if spec["kind"] == "action":
            action_world = world.clone()
            result = getattr(action_world, spec["tool"])(**{spec["arg"]: believed})
            return {"answer": result, "acted_value": believed}
        raise ValueError(f"unknown dependent kind {spec['kind']!r}")
