"""Week 2: the live memory policies M2, M3, M4 (design freeze, Section 5).

Scripted trajectories only — no LLM runs. These tests prove the policy
machinery across all 12 chains x the 5 fault conditions:

(a) fault-derived false claims are never stored as verified fact by M4
    (any fault) or by M3 (where the evidence contract can see the fault:
    explicit error / empty-partial). M2 stores the claim verbatim but
    always with discounting provenance, never state-verified.
(b) no-fault legitimate memories pass through every guard intact
    (Legitimate-Memory Utility preserved; M4 accepts every true entry).
(c) correction-event behaviour under each policy, revising and stubborn.

One boundary is asserted, not hidden: under silent-wrong-value and stale
faults, M3 marks the false claim ``verified`` because the corrupted tool
response *is* the supporting observation. The evidence contract checks
claims against observations; only M4 checks them against DB state. That
is H3's ordering, mechanised.
"""
import pytest

from persistbench.agents import DependentAgent, NaiveAgent, OracleAgent
from persistbench.chains import CHAINS
from persistbench.faults import (
    EMPTY_PARTIAL,
    EXPLICIT_ERROR,
    FAULT_CONDITIONS,
    NO_FAULT,
    SILENT_WRONG_VALUE,
    STALE_VALUE,
)
from persistbench.memory import (
    POLICIES,
    M1SelfSummary,
    M2Disclosure,
    M3EvidenceContract,
    M4StateVerified,
    M5OracleTruth,
)
from persistbench.runner import Trajectory, correction_event, run_chain, run_dependent, run_origin
from persistbench.scorer import classify_write, grade_origin

NAIVE = NaiveAgent()
CHAIN_IDS = [c["chain_id"] for c in CHAINS]
FAULTY = [EXPLICIT_ERROR, EMPTY_PARTIAL, SILENT_WRONG_VALUE, STALE_VALUE]
EVIDENCE_VISIBLE = [EXPLICIT_ERROR, EMPTY_PARTIAL]   # M3 can see these
EVIDENCE_HIDDEN = [SILENT_WRONG_VALUE, STALE_VALUE]  # the observation itself lies

# Tool-response field -> fact key (mirrors agents.RESPONSE_FACT_MAP,
# plus the two status fields responses also carry).
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


def _dependent_correct(chain, world, memory, session, mode="revising"):
    spec = _spec(chain, session)
    correction = correction_event(chain, world) if session == 4 else None
    _, grade = run_dependent(chain, world, DependentAgent(memory, mode=mode),
                             spec, correction=correction)
    return grade.correct


def _claim(store, fact):
    return next(c for c in store["_claims"] if c["fact"] == fact)


# ---- shared: write logs exist and carry the frozen accounting ---------------

@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", FAULT_CONDITIONS)
@pytest.mark.parametrize("policy_cls", [M2Disclosure, M3EvidenceContract, M4StateVerified])
def test_write_log_records_entries_and_cost_placeholders(chain, fault, policy_cls):
    policy = policy_cls()
    traj = run_origin(chain, NAIVE, fault, seed=1)
    policy.write(traj, chain, traj.world)
    assert len(policy.write_log) == 1
    record = policy.write_log[0]
    assert record["policy"] == policy.name
    assert record["chain_id"] == chain["chain_id"]
    assert record["fault_condition"] == fault
    cost = record["cost"]
    assert cost["entries_considered"] == len(record["entries"])
    assert cost["entries_stored"] + cost["entries_not_stored"] == cost["entries_considered"]
    assert "token_estimate_placeholder" in cost


# ---- M2: disclosure --------------------------------------------------------

@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", FAULT_CONDITIONS)
def test_m2_stores_claim_verbatim_with_provenance(chain, fault):
    traj = run_origin(chain, NAIVE, fault, seed=1)
    store = M2Disclosure().write(traj, chain, traj.world)
    # facts are the agent's claim, verbatim (M2 filters nothing)
    for key, value in traj.claim.items():
        assert store[key] == value
    # every entry carries provenance, and none claims state verification
    assert set(store["_provenance"]) == set(traj.claim)
    for prov in store["_provenance"].values():
        assert prov["verified_against_state"] is False
        assert prov["source"] in (
            "tool_observation", "agent_assertion", "contradicted_by_observation")


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_m2_discloses_the_observed_error_it_claimed_past(chain):
    traj = run_origin(chain, NAIVE, EXPLICIT_ERROR, seed=1)
    store = M2Disclosure().write(traj, chain, traj.world)
    assert store["_disclosures"], "the observed tool error must be disclosed"
    prov = store["_provenance"]["task_success"]
    assert prov["source"] == "contradicted_by_observation"
    assert prov["discount"]


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", EVIDENCE_HIDDEN)
def test_m2_marks_the_corrupted_fact_as_observation_only(chain, fault):
    # (a) for M2: the false value is stored, but its provenance says it
    # rests on a tool observation alone, never verified against state.
    traj = run_origin(chain, NAIVE, fault, seed=1)
    store = M2Disclosure().write(traj, chain, traj.world)
    fact = _corrupted_fact(chain, fault)
    prov = store["_provenance"][fact]
    assert prov["source"] == "tool_observation"
    assert prov["verified_against_state"] is False
    # the fault-injected step is among the evidence (an earlier step may
    # legitimately report the same value first — e.g. 'paid' before ship)
    assert any(e["step"] == chain["critical_step"] for e in prov["evidence"])


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", FAULT_CONDITIONS)
def test_m2_preserves_legitimate_utility(chain, fault):
    # (b) M2's store carries the same fact values as M1, so dependent
    # behaviour is unchanged in every session, fault or not.
    traj = run_origin(chain, NAIVE, fault, seed=1)
    m1 = M1SelfSummary().write(traj, chain, traj.world)
    m2 = M2Disclosure().write(traj, chain, traj.world)
    for session in (2, 3, 4):
        assert _dependent_correct(chain, traj.world, m2, session) == \
            _dependent_correct(chain, traj.world, m1, session)


# ---- M3: evidence contract ---------------------------------------------------

@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", EVIDENCE_VISIBLE)
def test_m3_never_stores_a_false_claim_as_verified_fact(chain, fault):
    # (a) for M3, evidence-visible faults: the whole write classifies
    # clean — no falsified field is stored, as fact or otherwise.
    traj = run_origin(chain, NAIVE, fault, seed=1)
    store = M3EvidenceContract().write(traj, chain, traj.world)
    assert classify_write(store, traj, chain) == []
    grade = grade_origin(traj, chain)
    for fact in grade.falsified_fields:
        if fact == "task_success":
            assert store.get("task_success") is not True
        else:
            assert fact not in store
        assert _claim(store, fact)["status"] in ("failed", "unknown")


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_m3_contract_statuses_for_explicit_error(chain):
    traj = run_origin(chain, NAIVE, EXPLICIT_ERROR, seed=1)
    store = M3EvidenceContract().write(traj, chain, traj.world)
    success = _claim(store, "task_success")
    assert success["status"] == "failed"          # an error was observed
    assert success["evidence"] and "error" in success["evidence"]
    assert store["task_success"] is False


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_m3_contract_statuses_for_empty_partial(chain):
    traj = run_origin(chain, NAIVE, EMPTY_PARTIAL, seed=1)
    store = M3EvidenceContract().write(traj, chain, traj.world)
    success = _claim(store, "task_success")
    # No observation supports the success claim and none contradicts it:
    # the contract stores it as unknown — never verified, never asserted.
    assert success["status"] == "unknown"
    assert store.get("task_success") is not True


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", EVIDENCE_HIDDEN)
def test_m3_boundary_corrupted_observation_verifies_the_false_claim(chain, fault):
    # Documented boundary (H3 ordering): the corrupted response is the
    # evidence, so the contract verifies the false claim — with the
    # evidence pointer naming the fault-injected step. Only M4 (state
    # check) catches this class; see the M4 tests below.
    traj = run_origin(chain, NAIVE, fault, seed=1)
    store = M3EvidenceContract().write(traj, chain, traj.world)
    fact = _corrupted_fact(chain, fault)
    claim = _claim(store, fact)
    assert claim["status"] == "verified"
    assert claim["claimed"] != grade_origin(traj, chain).truth[fact]  # it IS false
    assert fact in classify_write(store, traj, chain)


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_m3_no_fault_write_is_fully_verified_and_useful(chain):
    # (b) No fault: every claim verified, nothing lost vs M1 downstream.
    traj = run_origin(chain, NAIVE, NO_FAULT, seed=1)
    store = M3EvidenceContract().write(traj, chain, traj.world)
    assert classify_write(store, traj, chain) == []
    assert all(c["status"] == "verified" for c in store["_claims"])
    assert all(key in store for key in chain["intended_facts"])
    m1 = M1SelfSummary().write(traj, chain, traj.world)
    for session in (2, 3, 4):
        assert _dependent_correct(chain, traj.world, store, session) == \
            _dependent_correct(chain, traj.world, m1, session)


# ---- M4: state-verified write --------------------------------------------------

@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", FAULT_CONDITIONS)
@pytest.mark.parametrize("agent", [NAIVE, OracleAgent()], ids=["naive", "oracle"])
def test_m4_never_stores_a_false_memory(chain, fault, agent):
    # (a) for M4, all faults, both scripted agents: by construction.
    traj = run_origin(chain, agent, fault, seed=1)
    store = M4StateVerified().write(traj, chain, traj.world)
    assert classify_write(store, traj, chain) == []


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", FAULT_CONDITIONS)
def test_m4_store_matches_oracle_truth(chain, fault):
    traj = run_origin(chain, NAIVE, fault, seed=1)
    store = M4StateVerified().write(traj, chain, traj.world)
    oracle = M5OracleTruth().write(traj, chain, traj.world)
    for key in chain["intended_facts"]:
        assert store.get(key) == oracle[key]  # .get: M4 omits DB-absent facts
    assert store["task_success"] == oracle["task_success"]


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", FAULT_CONDITIONS)
def test_m4_log_accounts_for_every_falsified_field(chain, fault):
    # The correction/rejection accounting the utility metric reads:
    # exactly the scorer's falsified fields are corrected or rejected,
    # and every other claimed entry is accepted untouched.
    policy = M4StateVerified()
    traj = run_origin(chain, NAIVE, fault, seed=1)
    policy.write(traj, chain, traj.world)
    grade = grade_origin(traj, chain)
    record = policy.write_log[-1]
    acted = [e for e in record["entries"] if e["action"] in ("corrected", "rejected")]
    assert len(acted) == len(grade.falsified_fields)
    assert {e["fact"] for e in acted} == set(grade.falsified_fields)
    assert all(e["provenance"] == "state-verified" for e in record["entries"])
    cost = record["cost"]
    assert cost["entries_corrected"] + cost["entries_rejected"] == len(acted)
    assert cost["entries_delayed_or_rejected"] == cost["entries_rejected"]


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_m4_no_fault_accepts_everything_and_matches_m1(chain):
    # (b) + pre-registration falsification check 3: the guard must not
    # reject or rewrite true entries — utility is preserved exactly.
    policy = M4StateVerified()
    traj = run_origin(chain, NAIVE, NO_FAULT, seed=1)
    store = policy.write(traj, chain, traj.world)
    cost = policy.write_log[-1]["cost"]
    assert cost["entries_corrected"] == 0
    assert cost["entries_rejected"] == 0
    assert cost["entries_generated"] == 0
    assert store == M1SelfSummary().write(traj, chain, traj.world)


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", FAULT_CONDITIONS)
def test_m4_dependents_all_correct(chain, fault):
    # With a state-verified store there is nothing false to propagate:
    # the agent's believed value equals ground truth in every session.
    # Full task success follows — except action dependents (return
    # labels) under suppressed-effect faults, where Session 1's shipment
    # never happened and no memory policy can make the action possible;
    # there the correct belief is what the guard is responsible for.
    traj = run_origin(chain, NAIVE, fault, seed=1)
    store = M4StateVerified().write(traj, chain, traj.world)
    for spec in chain["dependents"]:
        correction = correction_event(chain, traj.world) if spec["session"] == 4 else None
        _, grade = run_dependent(chain, traj.world, DependentAgent(store),
                                 spec, correction=correction)
        assert grade.believed == grade.truth, f"{chain['chain_id']}/{fault}/{spec}"
        if spec["kind"] != "action" or fault not in EVIDENCE_VISIBLE:
            assert grade.correct, f"{chain['chain_id']}/{fault}/{spec}"


def test_m4_rejects_keys_it_cannot_verify():
    from persistbench.chains import CHAINS_BY_ID, build_world

    chain = CHAINS_BY_ID["chain01_pay_ship_laptop"]
    world = build_world(chain)
    hand_written = Trajectory(
        chain_id=chain["chain_id"], agent="hand-written",
        fault_condition=NO_FAULT, seed=0, steps=[],
        claim={"order_status": "shipped", "tracking": "TRK-000001",
               "task_success": True, "bogus_fact": 123},
        world=world,
    )
    policy = M4StateVerified()
    store = policy.write(hand_written, chain, world)
    assert "bogus_fact" not in store
    rejected = [e for e in policy.write_log[-1]["entries"] if e["action"] == "rejected"]
    assert any(e["fact"] == "bogus_fact" for e in rejected)
    # claimed-but-false entries were corrected to DB state, not stored false
    assert store["order_status"] == "placed"
    assert classify_write(store, hand_written, chain) == []


# ---- (c) correction event under each guard ------------------------------------

@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
@pytest.mark.parametrize("fault", EVIDENCE_VISIBLE)
def test_correction_under_m3_and_m4_guards(chain, fault):
    # The guard already prevented the false memory, so even a stubborn
    # agent's Session 4 is correct — there is nothing to resist with.
    traj = run_origin(chain, NAIVE, fault, seed=1)
    for policy in (M3EvidenceContract(), M4StateVerified()):
        store = policy.write(traj, chain, traj.world)
        assert _dependent_correct(chain, traj.world, store, 4, mode="stubborn"), \
            f"{chain['chain_id']}/{fault}/{policy.name}"
        assert _dependent_correct(chain, traj.world, store, 4, mode="revising")


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_correction_resistance_survives_m2_and_m3_but_not_m4(chain):
    # Correction resistance is only testable where the corrupted fact IS
    # the chain's primary fact (the Session-4 task's fact): silent faults
    # corrupt the primary fact on every chain; stale faults only on
    # some. Under the faults whose lie reaches the store (verbatim under
    # M2, verified-by-corrupted-evidence under M3), a stubborn agent
    # still resists the correction and a revising agent is repaired —
    # the Week-1 M1 pattern. Under M4 the false memory never formed.
    spec = _spec(chain, 4)
    assert spec["fact"] == chain["primary_fact"]
    applicable = [f for f in EVIDENCE_HIDDEN
                  if _corrupted_fact(chain, f) == chain["primary_fact"]]
    assert SILENT_WRONG_VALUE in applicable  # holds on all 12 chains
    for fault in applicable:
        traj = run_origin(chain, NAIVE, fault, seed=1)
        for policy in (M2Disclosure(), M3EvidenceContract()):
            store = policy.write(traj, chain, traj.world)
            assert not _dependent_correct(chain, traj.world, store, 4, mode="stubborn"), \
                f"{chain['chain_id']}/{fault}/{policy.name} stubborn"
            assert _dependent_correct(chain, traj.world, store, 4, mode="revising"), \
                f"{chain['chain_id']}/{fault}/{policy.name} revising"
        store = M4StateVerified().write(traj, chain, traj.world)
        assert _dependent_correct(chain, traj.world, store, 4, mode="stubborn")


# ---- runner wiring: any policy x any fault, end to end -------------------------

@pytest.mark.parametrize("policy_name", sorted(POLICIES))
@pytest.mark.parametrize("fault", FAULT_CONDITIONS)
def test_run_chain_executes_every_policy_under_every_fault(policy_name, fault):
    result = run_chain(CHAINS[0], NAIVE, fault, POLICIES[policy_name], seed=1)
    assert result["policy"] == policy_name
    assert len(result["dependents"]) == 3
    assert result["write_log"] is not None
    assert result["write_log"]["policy"] == policy_name


@pytest.mark.parametrize("chain", CHAINS, ids=CHAIN_IDS)
def test_run_chain_m4_end_to_end(chain):
    result = run_chain(chain, NAIVE, SILENT_WRONG_VALUE, M4StateVerified(), seed=1)
    assert result["write_falsified"] == []
    assert all(d["grade"].correct for d in result["dependents"])
