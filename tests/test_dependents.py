"""Propagation machinery: memory writes, dependent tasks, correction event.

These tests prove the Week 1 harness can *measure* the paper's chain —
fault -> falsified memory write -> downstream failure -> correction — with
scripted agents. They are harness validation, not results.
"""
import pytest

from persistbench.agents import DependentAgent, NaiveAgent, OracleAgent
from persistbench.chains import CHAINS
from persistbench.faults import SILENT_WRONG_VALUE, STALE_VALUE
from persistbench.memory import M0NoMemory, M1SelfSummary, M5OracleTruth
from persistbench.runner import correction_event, run_dependent, run_origin
from persistbench.scorer import classify_write

NAIVE = NaiveAgent()
CHAIN_IDS = [c["chain_id"] for c in CHAINS]

# Tool-response field -> fact key (mirrors agents.RESPONSE_FACT_MAP).
FIELD_TO_FACT = {
    "tracking": "tracking",
    "amount_cents": "charged_cents",
    "refund_amount_cents": "refund_cents",
    "address": "address",
    "stock": "stock",
    "cancellation_fee_cents": "cancellation_fee_cents",
    "order_status": "order_status",
    "payment_status": "payment_status",
}


def _corrupted_fact(chain, fault):
    spec = chain["fault"]
    if fault == SILENT_WRONG_VALUE:
        return FIELD_TO_FACT[spec["wrong_field"]]
    return FIELD_TO_FACT[spec["stale_field"]]


def _spec(chain, session):
    return next(d for d in chain["dependents"] if d["session"] == session)


# ---- memory writes -----------------------------------------------------------

@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_m1_write_of_naive_claim_is_classified_false(chain):
    traj = run_origin(chain, NAIVE, SILENT_WRONG_VALUE, seed=1)
    memory = M1SelfSummary().write(traj, chain, traj.world)
    falsified = classify_write(memory, traj, chain)
    assert falsified, f"{chain['chain_id']}: M1 write should be falsified"
    assert _corrupted_fact(chain, SILENT_WRONG_VALUE) in falsified


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_m5_write_is_never_falsified(chain):
    traj = run_origin(chain, NAIVE, SILENT_WRONG_VALUE, seed=1)
    memory = M5OracleTruth().write(traj, chain, traj.world)
    assert classify_write(memory, traj, chain) == []


def test_m0_writes_nothing():
    # Week 1 also asserted here that M2 still raised NotImplementedError.
    # Week 2 implements M2-M4, superseding that stub assertion; M2-M4
    # behaviour is covered in tests/test_week2_policies.py.
    chain = CHAINS[0]
    traj = run_origin(chain, NAIVE, SILENT_WRONG_VALUE, seed=1)
    assert M0NoMemory().write(traj, chain, traj.world) is None


# ---- propagation into Session 2 ----------------------------------------------

@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", [SILENT_WRONG_VALUE, STALE_VALUE])
def test_corrupted_memory_fails_the_dependent_that_needs_the_fact(chain, fault):
    spec = _spec(chain, 2)
    if spec["fact"] != _corrupted_fact(chain, fault):
        pytest.skip("Session-2 task does not depend on the corrupted fact")
    traj = run_origin(chain, NAIVE, fault, seed=1)
    memory = M1SelfSummary().write(traj, chain, traj.world)
    _, bad = run_dependent(chain, traj.world, DependentAgent(memory), spec)
    assert not bad.correct, f"{chain['chain_id']}/{fault}: M1 memory should mislead Session 2"
    _, good_oracle = run_dependent(
        chain, traj.world, DependentAgent(M5OracleTruth().write(traj, chain, traj.world)), spec)
    assert good_oracle.correct
    _, good_none = run_dependent(chain, traj.world, DependentAgent(None), spec)
    assert good_none.correct, "M0 (query the world) must stay correct — the H2 floor"


# ---- action dependents (return labels) ----------------------------------------

@pytest.mark.parametrize("chain", [c for c in CHAINS if any(
    d["kind"] == "action" for d in c["dependents"])],
    ids=[c["chain_id"] for c in CHAINS if any(d["kind"] == "action" for d in c["dependents"])])
def test_action_dependent_uses_the_believed_tracking(chain):
    spec = next(d for d in chain["dependents"] if d["kind"] == "action")
    traj = run_origin(chain, NAIVE, SILENT_WRONG_VALUE, seed=1)
    memory = M1SelfSummary().write(traj, chain, traj.world)
    _, bad = run_dependent(chain, traj.world, DependentAgent(memory), spec)
    assert not bad.correct  # label attempted against TRK-999999, which does not exist
    _, good = run_dependent(
        chain, traj.world, DependentAgent(M5OracleTruth().write(traj, chain, traj.world)), spec)
    assert good.correct


# ---- correction event before Session 4 -----------------------------------------

@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_correction_event_repairs_a_revising_agent_but_not_a_stubborn_one(chain):
    spec = _spec(chain, 4)
    assert spec["fact"] == chain["primary_fact"]
    traj = run_origin(chain, NAIVE, SILENT_WRONG_VALUE, seed=1)
    memory = M1SelfSummary().write(traj, chain, traj.world)
    correction = correction_event(chain, traj.world)
    _, stubborn = run_dependent(chain, traj.world, DependentAgent(memory, mode="stubborn"),
                                spec, correction=correction)
    assert not stubborn.correct, f"{chain['chain_id']}: stubborn agent keeps the false memory"
    _, revising = run_dependent(chain, traj.world, DependentAgent(memory, mode="revising"),
                                spec, correction=correction)
    assert revising.correct, f"{chain['chain_id']}: correction should repair a revising agent"
