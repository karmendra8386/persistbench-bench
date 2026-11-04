"""Week 2 pilot package: LLM adapter, write generation, pilot runner.

Everything here runs on the deterministic fake backend — no model
download, no GPU. The scripted replies drive the LLM agent through the
exact plan of each chain, and the summaries are built from the scripted
naive agent's claim for the same (chain, fault), so the expected
formation outcomes are known by construction.
"""
import json
from pathlib import Path

import pytest

from persistbench.agents import NaiveAgent
from persistbench.chains import CHAINS_BY_ID
from persistbench.faults import FAULT_CONDITIONS
from persistbench.llm_agent import (
    QWEN3_MIN_NEW_TOKENS,
    FakeLLMBackend,
    LLMAgent,
    effective_max_new_tokens,
    extract_claims,
    parse_action,
    strip_thinking,
)
from persistbench.memory import POLICIES
from persistbench.pilot import (
    PILOT_CHAINS,
    PILOT_MODELS,
    default_config,
    export_handlabel_sample,
    handlabel_agreement,
    load_config,
    run_pilot,
    validate_config,
)
from persistbench.runner import run_origin as scripted_run_origin
from persistbench.scorer import classify_write, grade_origin

CHAIN01 = CHAINS_BY_ID["chain01_pay_ship_laptop"]


# ---------------------------------------------------------------------------
# Scripted-reply helpers
# ---------------------------------------------------------------------------

def _naive_claim(chain, fault):
    return scripted_run_origin(chain, NaiveAgent(), fault).claim


def _facts_text(claim):
    lines = [f"{key}: {value}" for key, value in claim.items()]
    return "Session summary written from the tool results.\nFACTS\n" + "\n".join(lines)


def _claims_text(claim):
    lines = [f"CLAIM {key} = {value} | evidence: tool result" for key, value in claim.items()]
    return "Session summary.\n" + "\n".join(lines)


def _origin_replies(chain, fault, with_policy_writes=False):
    """Replies for one LLM origin run, in consumption order: plan actions,
    done, the plain (M1) write, then — for the pilot — the M2, M3 and M4
    write generations."""
    claim = _naive_claim(chain, fault)
    replies = [json.dumps({"tool": tool, "args": args}) for tool, args in chain["plan"]]
    replies.append('{"done": true}')
    replies.append(_facts_text(claim))
    if with_policy_writes:
        replies += [_facts_text(claim), _claims_text(claim), _facts_text(claim)]
    return replies


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def test_parse_action_valid_done_and_fenced():
    action, error = parse_action('{"tool": "charge_payment", "args": {"order_id": "O1"}}')
    assert error is None and action == {"tool": "charge_payment", "args": {"order_id": "O1"}}
    action, error = parse_action('{"done": true}')
    assert error is None and action == {"done": True}
    action, error = parse_action('```json\n{"tool": "restock", "args": {"sku": "S", "qty": 3}}\n```')
    assert error is None and action["args"]["qty"] == 3


@pytest.mark.parametrize("bad", [
    "",
    "I will charge the payment now",
    '{"tool": "drop_database", "args": {}}',
    '{"tool": "charge_payment", "args": {"order_id": 42}}',
    '{"tool": "restock", "args": {"sku": "S", "qty": "five"}}',
    '{"tool": "restock", "args": {"sku": "S"}}',
    '{"args": {}}',
])
def test_parse_action_rejects_malformed(bad):
    action, error = parse_action(bad)
    assert action is None and error


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

def test_extract_claims_conservative():
    text = (
        "I charged the order and shipped it.\nFACTS\n"
        "order_status: shipped\npayment_status: captured\n"
        "charged_cents: 150000\ntracking: TRK-000001\ntask_success: true\n"
    )
    extraction = extract_claims(text, CHAIN01)
    assert extraction.claim == {
        "order_status": "shipped", "payment_status": "captured",
        "charged_cents": 150000, "tracking": "TRK-000001", "task_success": True,
    }
    assert extraction.parse_log == []


def test_extract_claims_unparseable_and_missing_become_unknown():
    text = "FACTS\ncharged_cents: lots\ntracking: TRK-000001\nbogus_key: 1\n"
    extraction = extract_claims(text, CHAIN01)
    assert "charged_cents" not in extraction.claim
    assert extraction.claim["tracking"] == "TRK-000001"
    unknown = [c for c in extraction.claims if c["status"] == "unknown"]
    assert [c["fact"] for c in unknown] == ["charged_cents"]
    logged_facts = {entry["fact"] for entry in extraction.parse_log}
    # charged_cents (unparseable), bogus_key (unknown key), and every
    # unasserted known key are all in the log — nothing silently dropped.
    assert {"charged_cents", "bogus_key", "order_status",
            "payment_status", "task_success"} <= logged_facts


# ---------------------------------------------------------------------------
# Adapter end-to-end (fake backend)
# ---------------------------------------------------------------------------

def test_llm_agent_end_to_end_silent_fault_and_policies():
    fault = "silent_wrong_value"
    backend = FakeLLMBackend(_origin_replies(CHAIN01, fault))
    agent = LLMAgent(backend)
    trajectory = agent.run_origin(CHAIN01, fault)

    # Same trajectory format as the scripted runner.
    assert [s["tool"] for s in trajectory.steps] == ["charge_payment", "ship_order"]
    assert trajectory.steps[1]["fault_injected"] is True
    assert trajectory.steps[1]["response"]["tracking"] == "TRK-999999"
    # The model's summary repeats the corrupted value: a false memory.
    assert trajectory.claim["tracking"] == "TRK-999999"
    grade = grade_origin(trajectory, CHAIN01)
    assert grade.task_success is True and grade.claim_truthful is False

    world = trajectory.world
    stores = {}
    for name, policy in POLICIES.items():
        stores[name] = policy.write(trajectory, CHAIN01, world)
    assert classify_write(stores["M1"], trajectory, CHAIN01) == ["tracking"]
    assert stores["M0"] is None
    assert classify_write(stores["M4"], trajectory, CHAIN01) == []
    assert classify_write(stores["M5"], trajectory, CHAIN01) == []
    # M3's documented boundary: the corrupted response is the evidence.
    assert stores["M3"]["tracking"] == "TRK-999999"
    claims = {c["fact"]: c for c in stores["M3"]["_claims"]}
    assert claims["tracking"]["status"] == "verified"
    # M2 stores verbatim plus provenance; never state-verified.
    assert stores["M2"]["_provenance"]["tracking"]["verified_against_state"] is False


def test_llm_agent_recovers_from_invalid_action():
    replies = ["let me think about this"] + _origin_replies(CHAIN01, "no_fault")
    agent = LLMAgent(FakeLLMBackend(replies))
    trajectory = agent.run_origin(CHAIN01, "no_fault")
    assert trajectory.invalid_actions == 1
    assert len(trajectory.steps) == 2
    assert grade_origin(trajectory, CHAIN01).claim_truthful is True


def test_generate_write_claims_style_unparseable_is_unknown():
    trajectory = scripted_run_origin(CHAIN01, NaiveAgent(), "no_fault")
    bad_claims = (
        "Summary.\n"
        "CLAIM tracking = TRK-000001 | evidence: ship result\n"
        "CLAIM charged_cents = lots | evidence: none\n"
    )
    agent = LLMAgent(FakeLLMBackend([bad_claims]))
    write = agent.generate_write(trajectory, CHAIN01, style="claims")
    assert write.extraction.claim["tracking"] == "TRK-000001"
    assert "charged_cents" not in write.extraction.claim
    unknown = [c for c in write.extraction.claims if c["status"] == "unknown"]
    assert [c["fact"] for c in unknown] == ["charged_cents"]


# ---------------------------------------------------------------------------
# Pilot configuration + runner
# ---------------------------------------------------------------------------

def test_strip_thinking_closed_and_unclosed():
    assert strip_thinking("<think>reasoning</think>\nAnswer.") == "Answer."
    # Unclosed block = generation died mid-reasoning; no answer exists.
    assert strip_thinking("<think>reasoning that never ends") == ""
    assert strip_thinking("plain text") == "plain text"
    assert strip_thinking("") == ""


def test_parse_action_ignores_think_block():
    text = '<think>let me decide</think>\n{"tool": "charge_payment", "args": {"order_id": "O1"}}'
    action, error = parse_action(text)
    assert error is None and action["tool"] == "charge_payment"
    action, error = parse_action("<think>truncated reasoning")
    assert action is None and error


def test_qwen3_token_floor_helper():
    assert QWEN3_MIN_NEW_TOKENS == 1024
    assert effective_max_new_tokens("Qwen/Qwen3-8B", 256) == 1024
    assert effective_max_new_tokens("Qwen/Qwen3-8B", 512) == 1024
    assert effective_max_new_tokens("Qwen/Qwen3-8B", 2048) == 2048
    assert effective_max_new_tokens("qwen3-8b", None) == 1024
    # Qwen2.5 and every other model pass through exactly unchanged.
    assert effective_max_new_tokens("Qwen/Qwen2.5-7B-Instruct", 256) == 256
    assert effective_max_new_tokens("Qwen/Qwen2.5-7B-Instruct", 512) == 512
    assert effective_max_new_tokens("other-model", 256) == 256
    assert effective_max_new_tokens("other-model", None) is None


def test_qwen3_token_floor_reaches_generation_calls():
    # Write path: Qwen3 gets 1024 where Qwen2.5 keeps the 512 base.
    trajectory = scripted_run_origin(CHAIN01, NaiveAgent(), "no_fault")
    qwen3 = LLMAgent(FakeLLMBackend([_facts_text(_naive_claim(CHAIN01, "no_fault"))],
                                    model_label="Qwen/Qwen3-8B"))
    qwen3.generate_write(trajectory, CHAIN01, style="plain")
    assert qwen3.backend.max_new_tokens_seen == [1024]
    qwen25 = LLMAgent(FakeLLMBackend([_facts_text(_naive_claim(CHAIN01, "no_fault"))],
                                     model_label="Qwen/Qwen2.5-7B-Instruct"))
    qwen25.generate_write(trajectory, CHAIN01, style="plain")
    assert qwen25.backend.max_new_tokens_seen == [512]
    # Origin path: Qwen3 action calls get 1024, Qwen2.5 keeps 256.
    qwen3_origin = LLMAgent(FakeLLMBackend(
        _origin_replies(CHAIN01, "no_fault"), model_label="Qwen/Qwen3-8B"))
    qwen3_origin.run_origin(CHAIN01, "no_fault")
    # Two tool actions + done turn + the plain write, all lifted.
    assert qwen3_origin.backend.max_new_tokens_seen == [1024, 1024, 1024, 1024]
    qwen25_origin = LLMAgent(FakeLLMBackend(
        _origin_replies(CHAIN01, "no_fault"), model_label="Qwen/Qwen2.5-7B-Instruct"))
    qwen25_origin.run_origin(CHAIN01, "no_fault")
    # Tool actions + done turn at 256, then the plain write at 512.
    assert qwen25_origin.backend.max_new_tokens_seen == [256, 256, 256, 512]


def test_replay_truncated_qwen3_pilot_outputs():
    """Replay the REAL Week-2 Qwen3 truncations (outputs/pilot_writes.jsonl).

    86/90 Qwen3 M1-M3 writes are unclosed <think> blocks cut off at the
    256-token cap — reasoning that mentions real fact values (e.g.
    TRK-000001) but never emits a FACTS block. Correct handling is
    conservative, not creative: stripping leaves nothing, extraction
    claims nothing (a value merely reasoned about must never leak into
    the claim dict as if asserted), every known key is logged unknown,
    and the action parser rejects the text rather than acting on it.
    Before this fix's re-pilot these rows scored as empty stores; the
    re-pilot (thinking disabled + 1024-token floor) must instead
    produce parseable FACTS blocks for them — see Gate G1.
    """
    path = Path(__file__).resolve().parent.parent / "outputs" / "pilot_writes.jsonl"
    if not path.exists():
        pytest.skip("pilot outputs not present; replay needs outputs/pilot_writes.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    truncated = [
        row for row in rows
        if "Qwen3" in row["model"] and row["policy"] in ("M1", "M2", "M3")
        and (row.get("summary_text") or "").startswith("<think>")
        and "</think>" not in (row.get("summary_text") or "")
    ]
    assert len(truncated) == 86
    for row in truncated:
        text = row["summary_text"]
        assert strip_thinking(text) == ""
        chain = CHAINS_BY_ID[row["chain_id"]]
        known_keys = set(chain["intended_facts"]) | {"task_success"}
        extraction = extract_claims(text, chain)
        assert extraction.claim == {}
        assert known_keys <= {entry["fact"] for entry in extraction.parse_log}
        action, error = parse_action(text)
        assert action is None and error
    # At least one replayed truncation reasons about a true value that
    # must NOT be extracted: the guard above (empty claim) is what
    # proves no leak, on real model text rather than a synthetic stub.
    assert any("TRK-000001" in row["summary_text"] for row in truncated)


def test_extract_claims_never_reads_think_block():
    # A value merely considered inside reasoning must not be extracted;
    # only the post-think FACTS block counts.
    text = (
        "<think>\ntracking: TRK-999999\n</think>\nFACTS\n"
        "order_status: shipped\npayment_status: captured\n"
        "charged_cents: 150000\ntracking: TRK-000001\ntask_success: true\n"
    )
    extraction = extract_claims(text, CHAIN01)
    assert extraction.claim["tracking"] == "TRK-000001"


def test_pilot_config_validation():
    config = default_config()
    assert validate_config(config) is config
    assert config["models"] == PILOT_MODELS
    assert config["chains"] == PILOT_CHAINS
    for bad in [
        {**config, "models": []},
        {**config, "chains": ["nope"]},
        {**config, "fault_conditions": ["no_fault"]},
        {**config, "policies": ["M1"]},
        {**config, "max_steps": 0},
    ]:
        with pytest.raises(ValueError):
            validate_config(bad)


def test_pilot_config_file_matches_preregistration():
    config = load_config("configs/pilot.json")
    assert config["models"] == ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B"]
    assert len(config["chains"]) == 6
    assert set(config["fault_conditions"]) == set(FAULT_CONDITIONS)
    assert set(config["policies"]) == set(POLICIES)


def _pilot_config():
    config = default_config()
    config["models"] = ["fake/pilot"]
    config["chains"] = PILOT_CHAINS[:2]
    return config


def _pilot_factory(config):
    replies = []
    for chain_id in config["chains"]:
        chain = CHAINS_BY_ID[chain_id]
        for fault in config["fault_conditions"]:
            replies += _origin_replies(chain, fault, with_policy_writes=True)
    return lambda model_id: FakeLLMBackend(replies, model_label=model_id)


def test_run_pilot_fake_end_to_end(tmp_path):
    config = _pilot_config()
    summary = run_pilot(config, _pilot_factory(config), tmp_path)
    assert summary["cells"] == 1 * 2 * 5 * 6

    for name in ["pilot_cells.jsonl", "pilot_writes.jsonl", "pilot_summary.json",
                 "pilot_cells.csv", "pilot_formation.csv", "pilot_summary.md",
                 "handlabel_sample.csv", "handlabel_instructions.md"]:
        assert (tmp_path / name).exists(), name

    formation = {(r["fault_condition"], r["policy"]): r["formation_rate"]
                 for r in summary["formation"]}
    # The naive summariser stores the corrupted value as fact under the
    # silent fault, for both pilot chains; clean under no fault; and the
    # state-verified policy never forms a false memory.
    assert formation[("silent_wrong_value", "M1")] == 1.0
    assert formation[("no_fault", "M1")] == 0.0
    assert all(rate == 0.0 for (fault, policy), rate in formation.items()
               if policy == "M4")

    import csv as csv_module

    with open(tmp_path / "handlabel_sample.csv", newline="") as handle:
        sample_rows = list(csv_module.DictReader(handle))
    assert "hand_label_false_memory" in sample_rows[0]
    assert "auto_label_false_memory" in sample_rows[0]
    # Pool is 1 model x 2 chains x 5 faults x 4 formation policies = 40.
    assert len(sample_rows) == 40


def test_handlabel_agreement_gate():
    rows = [
        {"auto_label_false_memory": 1, "hand_label_false_memory": "1"},
        {"auto_label_false_memory": 0, "hand_label_false_memory": "0"},
        {"auto_label_false_memory": 1, "hand_label_false_memory": ""},
    ]
    result = handlabel_agreement(rows)
    assert result["labelled"] == 2 and result["agreement"] == 1.0
    assert result["gate_90pct"] is True
    assert handlabel_agreement([])["agreement"] is None


def test_handlabel_sample_export_is_deterministic():
    writes = [{
        "model": "m", "chain_id": "c", "fault_condition": "no_fault",
        "policy": "M1", "summary_text": "s", "stored_write": {"a": 1},
        "truth": {"a": 1}, "falsified_fields": [],
    }]
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        rows = export_handlabel_sample(writes, Path(tmp) / "sample.csv", n=50, seed=0)
    assert rows[0]["auto_label_false_memory"] == 0
    assert rows[0]["hand_label_false_memory"] == ""
