"""WEEK 1 EXIT CRITERION (docs/design-freeze.md, Section 8):

The harness grades hand-written/scripted trajectories correctly in ALL 5
fault conditions:
- the scripted oracle agent scores correct (task success + truthful claim)
  under no-fault, on every chain;
- under each injected fault, the scorer detects the outcome mismatch in the
  scripted naive agent's claim, on every chain;
- the oracle's claim is never falsified, under any fault, on any chain.
"""
import pytest

from persistbench.agents import NaiveAgent, OracleAgent
from persistbench.chains import CHAINS
from persistbench.faults import (
    EMPTY_PARTIAL,
    EXPLICIT_ERROR,
    FAULT_CONDITIONS,
    NO_FAULT,
    SILENT_WRONG_VALUE,
    STALE_VALUE,
)
from persistbench.runner import Trajectory, run_origin
from persistbench.scorer import grade_origin

ORACLE = OracleAgent()
NAIVE = NaiveAgent()
CHAIN_IDS = [c["chain_id"] for c in CHAINS]
FAULTY = [EXPLICIT_ERROR, EMPTY_PARTIAL, SILENT_WRONG_VALUE, STALE_VALUE]


def _grade(chain, agent, fault):
    return grade_origin(run_origin(chain, agent, fault, seed=7), chain)


# ---- suite shape ----------------------------------------------------------

def test_suite_has_12_chains():
    assert len(CHAINS) == 12
    assert len(set(CHAIN_IDS)) == 12


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_chain_is_wellformed(chain):
    assert [d["session"] for d in chain["dependents"]] == [2, 3, 4]
    assert 0 <= chain["critical_step"] < len(chain["plan"])
    for key in ("wrong_field", "wrong_value", "stale_field", "stale_value", "error_message"):
        assert key in chain["fault"]
    assert chain["primary_fact"] in chain["intended_facts"]


# ---- exit criterion: oracle under no fault --------------------------------

@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_oracle_no_fault_scores_correct(chain):
    grade = _grade(chain, ORACLE, NO_FAULT)
    assert grade.task_success, f"{chain['chain_id']}: oracle should complete the task"
    assert grade.claim_truthful, f"{chain['chain_id']}: oracle claim falsified {grade.falsified_fields}"


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_naive_no_fault_is_also_truthful(chain):
    # Control sanity: with no fault, even the unverifying agent's claim
    # happens to match the DB — faults, not the agent, create the falsity.
    grade = _grade(chain, NAIVE, NO_FAULT)
    assert grade.task_success
    assert grade.claim_truthful


# ---- exit criterion: every injected fault is detected ----------------------

@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", FAULTY)
def test_scorer_detects_naive_mismatch_under_every_fault(chain, fault):
    grade = _grade(chain, NAIVE, fault)
    assert not grade.claim_truthful, (
        f"{chain['chain_id']} / {fault}: naive claim should be falsified"
    )
    assert grade.falsified_fields, "scorer must name the falsified fields"


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", FAULT_CONDITIONS)
def test_oracle_claim_never_falsified(chain, fault):
    grade = _grade(chain, ORACLE, fault)
    assert grade.claim_truthful, (
        f"{chain['chain_id']} / {fault}: oracle claim falsified {grade.falsified_fields}"
    )


# ---- the dissociation the paper is built on --------------------------------

@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", [SILENT_WRONG_VALUE, STALE_VALUE])
def test_response_lies_leave_db_correct_but_claim_false(chain, fault):
    # Silent/stale faults corrupt the *report*, not the world: the task
    # genuinely succeeded, yet the naive claim is false. Formation material.
    grade = _grade(chain, NAIVE, fault)
    assert grade.task_success
    assert not grade.claim_truthful


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", [EXPLICIT_ERROR, EMPTY_PARTIAL])
def test_suppressed_effects_fail_the_task_and_the_claim(chain, fault):
    grade = _grade(chain, NAIVE, fault)
    assert not grade.task_success
    assert not grade.claim_truthful
    assert "task_success" in grade.falsified_fields


# ---- hand-written trajectory ------------------------------------------------

def test_handwritten_false_trajectory_is_detected():
    from persistbench.chains import CHAINS_BY_ID, build_world

    chain = CHAINS_BY_ID["chain01_pay_ship_laptop"]
    world = build_world(chain)  # nothing executed: order is still 'placed'
    hand_written = Trajectory(
        chain_id=chain["chain_id"], agent="hand-written", fault_condition=NO_FAULT,
        seed=0, steps=[],
        claim={"order_status": "shipped", "payment_status": "captured",
               "charged_cents": 150000, "tracking": "TRK-000001", "task_success": True},
        world=world,
    )
    grade = grade_origin(hand_written, chain)
    assert not grade.task_success
    assert not grade.claim_truthful
    assert "task_success" in grade.falsified_fields
    assert "order_status" in grade.falsified_fields


# ---- determinism & injection bookkeeping ------------------------------------

@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", FAULT_CONDITIONS)
def test_runs_are_deterministic(chain, fault):
    first = run_origin(chain, NAIVE, fault, seed=3)
    second = run_origin(chain, NAIVE, fault, seed=3)
    assert [s["response"] for s in first.steps] == [s["response"] for s in second.steps]
    assert first.claim == second.claim


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_fault_flag_marks_only_the_critical_step(chain):
    traj = run_origin(chain, NAIVE, SILENT_WRONG_VALUE, seed=0)
    flagged = [i for i, s in enumerate(traj.steps) if s["fault_injected"]]
    assert flagged == [chain["critical_step"]]
    clean = run_origin(chain, NAIVE, NO_FAULT, seed=0)
    assert not any(s["fault_injected"] for s in clean.steps)
