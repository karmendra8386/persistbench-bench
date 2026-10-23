"""Runs scripted trajectories through the world, faults, and scorer."""
from __future__ import annotations

from dataclasses import dataclass, field

from .chains import build_world, resolve_fact
from .faults import FaultInjector
from .scorer import grade_dependent


@dataclass
class Trajectory:
    chain_id: str
    agent: str
    fault_condition: str
    seed: int
    steps: list = field(default_factory=list)
    claim: dict = field(default_factory=dict)
    world: object = None


def run_origin(chain: dict, agent, fault_condition: str, seed: int = 0) -> Trajectory:
    """Execute a chain's Session-1 plan under one fault condition."""
    world = build_world(chain)
    injector = FaultInjector(fault_condition, seed=seed)
    steps = []
    for index, (tool, kwargs) in enumerate(chain["plan"]):
        fn = getattr(world, tool)
        if index == chain["critical_step"]:
            response = injector.apply(fn, kwargs, chain["fault"])
            injected = fault_condition != "no_fault"
        else:
            response = fn(**kwargs)
            injected = False
        steps.append({
            "tool": tool,
            "kwargs": dict(kwargs),
            "response": response,
            "fault_injected": injected,
        })
    claim = agent.build_claim(chain, steps, world)
    return Trajectory(
        chain_id=chain["chain_id"],
        agent=agent.name,
        fault_condition=fault_condition,
        seed=seed,
        steps=steps,
        claim=claim,
        world=world,
    )


def correction_event(chain: dict, world) -> dict:
    """The verified observation delivered before Session 4: the primary
    fact, read from ground truth."""
    return {chain["primary_fact"]: resolve_fact(world, chain, chain["primary_fact"])}


def run_dependent(chain: dict, world, agent, spec: dict, correction: dict | None = None):
    """Run and grade one dependent task. Returns (outcome, DependentGrade).

    The outcome is annotated with the agent's believed fact value so the
    scorer can grade decisions as well as reports and actions.
    """
    from .agents import DependentAgent  # local import to avoid a cycle

    assert isinstance(agent, DependentAgent)
    believed = agent.belief(chain, world, spec["fact"],
                            correction if spec["session"] == 4 else None)
    outcome = agent.run(chain, world, spec, correction=correction)
    outcome["believed"] = believed
    return outcome, grade_dependent(chain, world, spec, outcome)
