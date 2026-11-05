"""Gate G2: the live-model dependent agent (LLMDependentAgent).

Everything here runs on the deterministic fake backend — no model
download, no GPU. The fake replies play the role of the model: they
answer from the memory the agent is shown, so the tests prove the
plumbing (memory rendering, tool fallback, correction handling,
conservative parsing, run_chain wiring) — not what a real model will
do with a false memory. That is what the Kaggle smoke measures.
"""
import json

import pytest

from persistbench.agents import NaiveAgent
from persistbench.chains import CHAINS_BY_ID
from persistbench.faults import SILENT_WRONG_VALUE
from persistbench.llm_agent import (
    FakeLLMBackend,
    LLMDependentAgent,
    parse_dependent_reply,
)
from persistbench.memory import M1SelfSummary, M2Disclosure
from persistbench.runner import (
    correction_event,
    run_chain,
    run_dependent,
    run_origin,
)
from persistbench.scorer import grade_dependent

CHAIN01 = CHAINS_BY_ID["chain01_pay_ship_laptop"]
CHAIN05 = CHAINS_BY_ID["chain05_restock_lamp"]


def _m1_memory(chain, fault, seed=1):
    """(M1 store, trajectory) for a scripted naive origin run."""
    traj = run_origin(chain, NaiveAgent(), fault, seed=seed)
    return M1SelfSummary().write(traj, chain, traj.world), traj


def _spec(chain, session):
    return next(d for d in chain["dependents"] if d["session"] == session)


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,exemplar,expected", [
    ('{"answer": "TRK-000001"}', "", "TRK-000001"),
    ('{"answer": 40}', 0, 40),
    ('{"answer": "40"}', 0, 40),
    ('{"answer": true}', True, True),
    ('{"answer": "false"}', True, False),
    ('{"answer": null}', "", None),
    ('<think>checking memory</think>\n{"answer": "TRK-000001"}', "",
     "TRK-000001"),
    ('```json\n{"answer": "TRK-000001"}\n```', "", "TRK-000001"),
])
def test_parse_dependent_reply_answers(text, exemplar, expected):
    reply, error = parse_dependent_reply(text, exemplar)
    assert error is None
    assert reply == {"answer": expected}


def test_parse_dependent_reply_tool_call():
    reply, error = parse_dependent_reply(
        '{"tool": "get_order", "args": {"order_id": "O1001"}}', "")
    assert error is None
    assert reply == {"tool": "get_order", "args": {"order_id": "O1001"}}


@pytest.mark.parametrize("bad,exemplar", [
    ("", ""),
    ("just prose, no JSON", ""),
    ('{"done": true}', ""),
    ('{"answer": "lots"}', 0),
    ('{"answer": 42}', ""),
    ('{"answer": true}', 0),
    ('{"tool": "drop_database", "args": {}}', ""),
    ('{"tool": "charge_payment", "args": {"order_id": 42}}', ""),
])
def test_parse_dependent_reply_rejects_malformed(bad, exemplar):
    reply, error = parse_dependent_reply(bad, exemplar)
    assert reply is None and error


# ---------------------------------------------------------------------------
# Memory reading: true memory, false memory, correction
# ---------------------------------------------------------------------------

def test_live_dependent_reads_true_memory_fact():
    memory, traj = _m1_memory(CHAIN01, "no_fault")
    agent = LLMDependentAgent(
        FakeLLMBackend(['{"answer": "TRK-000001"}']), memory)
    spec = _spec(CHAIN01, 2)
    outcome = agent.run(CHAIN01, traj.world, spec)
    assert outcome["believed"] == "TRK-000001"
    assert outcome["answer"] == "TRK-000001"
    assert grade_dependent(CHAIN01, traj.world, spec, outcome).correct is True
    assert outcome["invalid_replies"] == 0
    assert agent.belief(CHAIN01, traj.world, "tracking", None) == "TRK-000001"


def test_false_stored_fact_propagates_into_believed_value():
    memory, traj = _m1_memory(CHAIN01, SILENT_WRONG_VALUE)
    assert memory["tracking"] == "TRK-999999"  # the false memory
    agent = LLMDependentAgent(
        FakeLLMBackend(['{"answer": "TRK-999999"}']), memory)
    spec = _spec(CHAIN01, 2)
    outcome = agent.run(CHAIN01, traj.world, spec)
    assert outcome["believed"] == "TRK-999999"
    assert grade_dependent(CHAIN01, traj.world, spec, outcome).correct is False


def test_session4_correction_overrides_false_memory():
    memory, traj = _m1_memory(CHAIN01, SILENT_WRONG_VALUE)
    seen = {}

    def reply(messages):
        seen["messages"] = messages
        return '{"answer": "TRK-000001"}'

    agent = LLMDependentAgent(FakeLLMBackend(reply), memory, mode="revising")
    spec = _spec(CHAIN01, 4)
    outcome = agent.run(
        CHAIN01, traj.world, spec,
        correction=correction_event(CHAIN01, traj.world),
    )
    assert outcome["believed"] == "TRK-000001"
    assert grade_dependent(CHAIN01, traj.world, spec, outcome).correct is True
    prompt = "\n".join(m["content"] for m in seen["messages"])
    # Both sides of the conflict reach the model: the false memory and
    # the verified correction that overrides it.
    assert "TRK-999999" in prompt
    assert "Verified observation" in prompt and "TRK-000001" in prompt


def test_correction_ignored_before_session4():
    # run_dependent passes the correction only to Session 4; a Session-2
    # run must not see it even when the caller hands it over.
    memory, traj = _m1_memory(CHAIN01, SILENT_WRONG_VALUE)
    seen = {}

    def reply(messages):
        seen["messages"] = messages
        return '{"answer": "TRK-999999"}'

    agent = LLMDependentAgent(FakeLLMBackend(reply), memory, mode="revising")
    spec = _spec(CHAIN01, 2)
    outcome = agent.run(
        CHAIN01, traj.world, spec,
        correction=correction_event(CHAIN01, traj.world),
    )
    assert outcome["believed"] == "TRK-999999"
    prompt = "\n".join(m["content"] for m in seen["messages"])
    assert "Verified observation" not in prompt


def test_stubborn_live_agent_is_never_shown_the_correction():
    memory, traj = _m1_memory(CHAIN01, SILENT_WRONG_VALUE)
    seen = {}

    def reply(messages):
        seen["messages"] = messages
        return '{"answer": "TRK-999999"}'

    agent = LLMDependentAgent(FakeLLMBackend(reply), memory, mode="stubborn")
    spec = _spec(CHAIN01, 4)
    outcome = agent.run(
        CHAIN01, traj.world, spec,
        correction=correction_event(CHAIN01, traj.world),
    )
    assert outcome["believed"] == "TRK-999999"
    assert grade_dependent(CHAIN01, traj.world, spec, outcome).correct is False
    prompt = "\n".join(m["content"] for m in seen["messages"])
    assert "Verified observation" not in prompt


def test_live_dependent_unknown_mode_rejected():
    with pytest.raises(ValueError):
        LLMDependentAgent(FakeLLMBackend([]), {}, mode="gullible")


def test_memory_annotations_are_never_rendered():
    # M2 stores carry _provenance / _claims annotations; the reader
    # does not condition on provenance, so they must not reach it.
    traj = run_origin(CHAIN01, NaiveAgent(), "no_fault", seed=1)
    memory = M2Disclosure().write(traj, CHAIN01, traj.world)
    assert any(key.startswith("_") for key in memory)
    seen = {}

    def reply(messages):
        seen["messages"] = messages
        return '{"answer": "TRK-000001"}'

    agent = LLMDependentAgent(FakeLLMBackend(reply), memory)
    agent.run(CHAIN01, traj.world, _spec(CHAIN01, 2))
    prompt = "\n".join(m["content"] for m in seen["messages"])
    assert "_provenance" not in prompt and "_claims" not in prompt
    assert "tracking: TRK-000001" in prompt


# ---------------------------------------------------------------------------
# M0 / partial memory: read-tool fallback, write tools refused
# ---------------------------------------------------------------------------

def test_m0_dependent_uses_read_tools():
    traj = run_origin(CHAIN01, NaiveAgent(), "no_fault", seed=1)
    calls = []

    def reply(messages):
        calls.append(1)
        if len(calls) == 1:
            return json.dumps({"tool": "get_order", "args": {"order_id": "O1001"}})
        return '{"answer": "TRK-000001"}'

    agent = LLMDependentAgent(FakeLLMBackend(reply), None)
    spec = _spec(CHAIN01, 2)
    outcome = agent.run(CHAIN01, traj.world, spec)
    assert outcome["believed"] == "TRK-000001"
    assert grade_dependent(CHAIN01, traj.world, spec, outcome).correct is True
    assert len(calls) == 2
    assert outcome["invalid_replies"] == 0


def test_write_tool_call_is_rejected_before_the_world():
    traj = run_origin(CHAIN01, NaiveAgent(), "no_fault", seed=1)
    calls = []

    def reply(messages):
        calls.append(1)
        if len(calls) == 1:
            return json.dumps(
                {"tool": "charge_payment", "args": {"order_id": "O1001"}})
        if len(calls) == 2:
            return json.dumps({"tool": "get_order", "args": {"order_id": "O1001"}})
        return '{"answer": "TRK-000001"}'

    agent = LLMDependentAgent(FakeLLMBackend(reply), None)
    spec = _spec(CHAIN01, 2)
    outcome = agent.run(CHAIN01, traj.world, spec)
    assert outcome["believed"] == "TRK-000001"
    assert outcome["invalid_replies"] == 1
    assert any("not available" in entry["reason"]
               for entry in outcome["parse_log"])
    # The world is untouched: the payment was never re-charged.
    assert traj.world.state["orders"]["O1001"]["payment_status"] == "captured"
    assert traj.world.state["counters"]["payment"] == 1


def test_tool_not_offered_when_memory_records_the_fact():
    memory, traj = _m1_memory(CHAIN01, "no_fault")
    calls = []

    def reply(messages):
        calls.append(1)
        if len(calls) == 1:
            return json.dumps({"tool": "get_order", "args": {"order_id": "O1001"}})
        return '{"answer": "TRK-000001"}'

    agent = LLMDependentAgent(FakeLLMBackend(reply), memory)
    spec = _spec(CHAIN01, 2)
    outcome = agent.run(CHAIN01, traj.world, spec)
    assert outcome["believed"] == "TRK-000001"
    assert outcome["invalid_replies"] == 1  # the tool call was refused


# ---------------------------------------------------------------------------
# Decision + action kinds, and failure modes
# ---------------------------------------------------------------------------

def test_live_decision_dependent():
    memory, traj = _m1_memory(CHAIN05, "no_fault")
    assert memory["stock"] == 40
    agent = LLMDependentAgent(FakeLLMBackend(['{"answer": 40}']), memory)
    spec = _spec(CHAIN05, 3)  # decision: stock >= 25
    outcome = agent.run(CHAIN05, traj.world, spec)
    assert outcome["believed"] == 40
    assert outcome["answer"] is True
    assert grade_dependent(CHAIN05, traj.world, spec, outcome).correct is True

    agent_false = LLMDependentAgent(
        FakeLLMBackend(['{"answer": 4}']), {"stock": 4})
    outcome_false = agent_false.run(CHAIN05, traj.world, spec)
    assert outcome_false["believed"] == 4
    assert outcome_false["answer"] is False
    assert grade_dependent(
        CHAIN05, traj.world, spec, outcome_false).correct is False


def test_live_action_dependent_uses_believed_tracking():
    memory, traj = _m1_memory(CHAIN01, SILENT_WRONG_VALUE)
    agent = LLMDependentAgent(
        FakeLLMBackend(['{"answer": "TRK-999999"}']), memory)
    spec = _spec(CHAIN01, 3)  # action: return label for the tracking
    outcome = agent.run(CHAIN01, traj.world, spec)
    assert outcome["acted_value"] == "TRK-999999"
    assert "error" in outcome["answer"]  # no such shipment exists
    assert grade_dependent(CHAIN01, traj.world, spec, outcome).correct is False


def test_unparseable_replies_end_in_believed_none_not_a_crash():
    memory, traj = _m1_memory(CHAIN05, "no_fault")
    agent = LLMDependentAgent(
        FakeLLMBackend(lambda messages: "lots of stock, definitely enough"),
        memory,
    )
    spec = _spec(CHAIN05, 3)  # decision: stock >= 25
    outcome = agent.run(CHAIN05, traj.world, spec)
    assert outcome["believed"] is None
    assert outcome["answer"] is None  # grades incorrect, never crashes
    assert grade_dependent(CHAIN05, traj.world, spec, outcome).correct is False
    assert outcome["invalid_replies"] == 5  # max_tool_calls + 2 turns
    assert outcome["parse_log"]


def test_abstention_null_answer_grades_incorrect():
    memory, traj = _m1_memory(CHAIN01, "no_fault")
    agent = LLMDependentAgent(FakeLLMBackend(['{"answer": null}']), memory)
    spec = _spec(CHAIN01, 2)
    outcome = agent.run(CHAIN01, traj.world, spec)
    assert outcome["believed"] is None
    assert grade_dependent(CHAIN01, traj.world, spec, outcome).correct is False
    assert any("abstained" in entry["reason"]
               for entry in outcome["parse_log"])


# ---------------------------------------------------------------------------
# run_chain wiring
# ---------------------------------------------------------------------------

def test_run_chain_live_end_to_end():
    # One fake backend serves all three dependent sessions in order:
    # Session 2 repeats the false memory, Session 3 acts on it, and
    # Session 4 (correction shown) answers with the verified value.
    backend = FakeLLMBackend([
        '{"answer": "TRK-999999"}',
        '{"answer": "TRK-999999"}',
        '{"answer": "TRK-000001"}',
    ])
    result = run_chain(
        CHAIN01, NaiveAgent(), SILENT_WRONG_VALUE, M1SelfSummary(),
        seed=1, dependent_agent="live", dependent_backend=backend,
    )
    assert backend.calls == 3
    assert result["write_falsified"] == ["tracking"]
    grades = [d["grade"] for d in result["dependents"]]
    assert [g.session for g in grades] == [2, 3, 4]
    assert grades[0].correct is False  # propagation: false memory misleads
    assert grades[1].correct is False  # action on the false tracking fails
    assert grades[2].correct is True   # correction repairs the revising agent
    assert result["dependents"][0]["outcome"]["believed"] == "TRK-999999"
    assert result["dependents"][2]["outcome"]["believed"] == "TRK-000001"


def test_run_chain_default_is_still_scripted():
    # No new arguments: the scripted path, unchanged.
    result = run_chain(CHAIN01, NaiveAgent(), SILENT_WRONG_VALUE,
                       M1SelfSummary(), seed=1)
    assert result["dependents"][0]["grade"].correct is False
    assert result["dependents"][0]["outcome"]["believed"] == "TRK-999999"
    assert result["dependents"][2]["grade"].correct is True


def test_run_chain_live_requires_a_backend():
    with pytest.raises(ValueError):
        run_chain(CHAIN01, NaiveAgent(), SILENT_WRONG_VALUE, M1SelfSummary(),
                  seed=1, dependent_agent="live")
    with pytest.raises(ValueError):
        run_chain(CHAIN01, NaiveAgent(), SILENT_WRONG_VALUE, M1SelfSummary(),
                  seed=1, dependent_agent="telepathic",
                  dependent_backend=FakeLLMBackend([]))


def test_run_dependent_rejects_a_live_agent_without_believed():
    class BrokenLive:
        is_live_dependent = True

        def run(self, chain, world, spec, correction=None):
            return {"answer": "TRK-000001"}  # no "believed" key

    traj = run_origin(CHAIN01, NaiveAgent(), "no_fault", seed=1)
    with pytest.raises(ValueError):
        run_dependent(CHAIN01, traj.world, BrokenLive(), _spec(CHAIN01, 2))
