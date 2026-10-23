"""Runs scripted trajectories through the world, faults, and scorer."""
from __future__ import annotations

from dataclasses import dataclass, field

from .chains import build_world, resolve_fact
from .faults import FaultInjector
from .scorer import classify_write, grade_dependent, grade_origin


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


def run_chain(chain: dict, origin_agent, fault_condition: str, policy,
              seed: int = 0, dependent_mode: str = "revising") -> dict:
    """Execute one full chain under one memory policy x one fault condition.

    Session 1 runs under the fault; the policy writes memory from the
    resulting trajectory; Sessions 2-4 then run against that memory, with
    the correction event delivered before Session 4. Works for any of
    M0-M5 and any fault condition. Returns everything the frozen metrics
    need: the origin grade, the policy's write log record, the stored
    memory and its false-memory classification, and the graded dependent
    outcomes (from which Propagation, Persistence and Correction
    Resistance are computed).
    """
    from .agents import DependentAgent  # local import to avoid a cycle

    trajectory = run_origin(chain, origin_agent, fault_condition, seed=seed)
    memory = policy.write(trajectory, chain, trajectory.world)
    correction = correction_event(chain, trajectory.world)
    dependents = []
    for spec in chain["dependents"]:
        outcome, grade = run_dependent(
            chain, trajectory.world,
            DependentAgent(memory, mode=dependent_mode), spec,
            correction=correction,
        )
        dependents.append({"spec": spec, "outcome": outcome, "grade": grade})
    return {
        "chain_id": chain["chain_id"],
        "policy": policy.name,
        "fault_condition": fault_condition,
        "trajectory": trajectory,
        "origin_grade": grade_origin(trajectory, chain),
        "memory": memory,
        "write_falsified": classify_write(memory, trajectory, chain),
        "write_log": policy.write_log[-1] if getattr(policy, "write_log", None) else None,
        "correction": correction,
        "dependents": dependents,
    }


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
