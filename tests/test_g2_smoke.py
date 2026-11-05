"""Gate G2 propagation smoke runner (persistbench/g2_smoke.py).

Everything here runs on the deterministic fake backend — no model
download, no GPU. The fake replies play the role of a *correct*
memory reader: they answer from the stored memory when it records
the fact, from the Session-4 correction when one is delivered, and
from a read tool otherwise. The tests therefore prove the plumbing
(store loading, world replay + repair, dependent driving, grading,
result writing, gate arithmetic) — not what a real model will do
with a false memory. That is what the Kaggle smoke measures.
"""
import json
import re
from pathlib import Path

import pytest

from persistbench.chains import CHAINS_BY_ID
from persistbench.g2_smoke import (
    ReuseError,
    analyse_cells,
    default_config,
    fallback_pilot_config,
    load_config,
    load_smoke_stores,
    reconcile_world,
    replay_origin_world,
    run_smoke,
    set_fact,
    validate_config,
)
from persistbench.llm_agent import FakeLLMBackend

REPO = Path(__file__).resolve().parent.parent
QWEN25 = "Qwen/Qwen2.5-7B-Instruct"
QWEN3 = "Qwen/Qwen3-8B"


# ---------------------------------------------------------------------------
# A correct-reader fake backend (test-only stand-in for the live model)
# ---------------------------------------------------------------------------

_FACT_KEYS = ("order_status", "payment_status", "charged_cents", "tracking",
              "stock", "address", "task_success")


def _reader_reply(messages):
    """Answer from correction > memory > read tool, like a good reader."""
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    prompt = user_msgs[0]
    last = user_msgs[-1]
    fact_match = re.search(r"Report the value of ([A-Za-z_]+)", prompt)
    assert fact_match, f"no fact in prompt: {prompt[:200]!r}"
    fact = fact_match.group(1)

    def fmt(value):
        return json.dumps({"answer": value})

    # A read-tool observation is in hand: answer from it.
    obs = re.search(r"Observation from (get_order|get_product|get_shipment): (\{.*\})",
                    last, re.DOTALL)
    if obs:
        data = json.loads(obs.group(2))
        if fact in data:
            return fmt(data[fact])
        if fact == "order_status":
            return fmt(data.get("status"))
        if fact == "tracking":
            return fmt(data.get("tracking"))
        if fact == "stock":
            return fmt(data.get("stock"))
        raise AssertionError(f"cannot read {fact} from {data}")

    # Session-4 correction overrides the memory.
    corr = re.search(r"Verified observation.*?overrides the stored memory for "
                     rf"{re.escape(fact)}: {re.escape(fact)}: (.+)\.",
                     prompt, re.DOTALL)
    if corr:
        raw = corr.group(1)
        try:
            return fmt(int(raw))
        except ValueError:
            return fmt(raw)

    # The stored memory, when it records the fact.
    mem = re.search(r"Stored memory from the earlier session:\n(.*?)(?:\n[A-Z]|\Z)",
                    prompt, re.DOTALL)
    if mem:
        for line in mem.group(1).splitlines():
            m = re.match(rf"{re.escape(fact)}: (.+)$", line.strip())
            if m:
                raw = m.group(1)
                try:
                    return fmt(int(raw))
                except ValueError:
                    return fmt(raw)

    # No memory for this fact: look it up with a read tool.
    order = re.search(r"Order id: (\S+)\.", prompt)
    if order:
        return json.dumps({"tool": "get_order",
                           "args": {"order_id": order.group(1)}})
    product = re.search(r"Product sku: (\S+)\.", prompt)
    if product:
        return json.dumps({"tool": "get_product",
                           "args": {"sku": product.group(1)}})
    raise AssertionError(f"no tool route for prompt: {prompt[:200]!r}")


def _factory(model_id):
    return FakeLLMBackend(_reader_reply, model_label=model_id)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_loads_and_validates():
    config = load_config(REPO / "configs" / "g2_smoke.json")
    assert config["models"] == [QWEN25, QWEN3]
    assert config["chains"] == ["chain01_pay_ship_laptop", "chain05_restock_lamp"]
    assert config["fault_conditions"] == ["silent_wrong_value", "stale_value"]
    assert config["policies"] == ["M0", "M1", "M5"]
    assert config["dependent_mode"] == "revising"
    assert config["seed"] == 0
    assert default_config()["writes"][QWEN25] == "outputs/pilot_writes.jsonl"
    assert default_config()["writes"][QWEN3] == "results/qwen3-repilot/pilot_writes.jsonl"


@pytest.mark.parametrize("bad", [
    {"models": []},
    {"chains": ["chain99_nope"]},
    {"fault_conditions": ["silent_wrong_value", "made_up"]},
    {"policies": ["M0", "M9"]},
    {"dependent_mode": "ignoring"},
    {"seed": "zero"},
    {"writes": {QWEN25: "outputs/pilot_writes.jsonl"}},  # Qwen3 missing
])
def test_config_validation_rejects(bad):
    config = default_config()
    config.update(bad)
    with pytest.raises(ValueError):
        validate_config(config)


# ---------------------------------------------------------------------------
# Store loading
# ---------------------------------------------------------------------------

def test_load_smoke_stores_24_cells_model_selected():
    config = load_config(REPO / "configs" / "g2_smoke.json")
    stores = load_smoke_stores(config)
    assert len(stores) == 24
    # The Nov 2 file holds both models; the Qwen3 cells must come from
    # the re-pilot, not the broken Nov 2 arm (empty M1 stores there).
    qwen3_m1 = stores[(QWEN3, "chain05_restock_lamp", "silent_wrong_value", "M1")]
    assert qwen3_m1["stored_write"] == {"stock": 4, "task_success": False}
    qwen25_m1 = stores[(QWEN25, "chain05_restock_lamp", "silent_wrong_value", "M1")]
    assert qwen25_m1["stored_write"] == {"stock": 76, "task_success": True}
    # Every cell carries the recorded DB truth the world reconciles to.
    for row in stores.values():
        assert isinstance(row["truth"], dict) and row["truth"]


def test_load_smoke_stores_missing_cell_is_reuse_error(tmp_path):
    config = default_config()
    # Qwen3 file with one smoke cell removed.
    rows = [json.loads(l) for l in
            (REPO / "results/qwen3-repilot/pilot_writes.jsonl").read_text().splitlines()]
    rows = [r for r in rows if not (
        r["chain_id"] == "chain05_restock_lamp"
        and r["fault_condition"] == "stale_value" and r["policy"] == "M1")]
    partial = tmp_path / "qwen3-partial.jsonl"
    partial.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ReuseError, match="missing"):
        load_smoke_stores(config, writes_paths={QWEN3: str(partial)})


def test_load_smoke_stores_duplicate_cell_is_reuse_error(tmp_path):
    config = default_config()
    rows = [json.loads(l) for l in
            (REPO / "results/qwen3-repilot/pilot_writes.jsonl").read_text().splitlines()]
    smoke_row = next(r for r in rows if r["chain_id"] == "chain01_pay_ship_laptop"
                     and r["fault_condition"] == "silent_wrong_value"
                     and r["policy"] == "M0")
    rows.append(dict(smoke_row))
    dup = tmp_path / "qwen3-dup.jsonl"
    dup.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ReuseError, match="duplicated"):
        load_smoke_stores(config, writes_paths={QWEN3: str(dup)})


def test_load_smoke_stores_missing_file_is_reuse_error():
    config = default_config()
    with pytest.raises(ReuseError, match="not found"):
        load_smoke_stores(config, writes_paths={QWEN3: "no/such/file.jsonl"})


# ---------------------------------------------------------------------------
# World replay + repair
# ---------------------------------------------------------------------------

def test_replay_matches_recorded_truth_where_plan_ran_once():
    config = load_config(REPO / "configs" / "g2_smoke.json")
    stores = load_smoke_stores(config)
    # chain01 (both models) and all Qwen3 cells: replay == recorded.
    for key, row in stores.items():
        if key[1] == "chain01_pay_ship_laptop" or key[0] == QWEN3:
            _, repair = reconcile_world(
                CHAINS_BY_ID[key[1]], key[2], 0, row["truth"])
            assert repair == [], key


def test_repair_qwen25_chain05_double_restock():
    config = load_config(REPO / "configs" / "g2_smoke.json")
    stores = load_smoke_stores(config)
    chain = CHAINS_BY_ID["chain05_restock_lamp"]
    for fault, recorded in (("silent_wrong_value", 76), ("stale_value", 80)):
        row = stores[(QWEN25, "chain05_restock_lamp", fault, "M1")]
        assert row["truth"] == {"stock": recorded}
        world, repair = reconcile_world(chain, fault, 0, row["truth"])
        assert repair == [{"fact": "stock", "replay_value": 40,
                           "recorded_value": recorded}]
        assert world.state["products"]["SKU-LAMP"]["stock"] == recorded
        # The read-tool fallback now returns the recorded truth.
        assert world.get_product("SKU-LAMP")["stock"] == recorded


def test_set_fact_unrepairable_is_reuse_error():
    # chain02 under explicit_error: the charge is suppressed, so the DB
    # holds no payment — a recorded charged_cents has nowhere to go.
    chain = CHAINS_BY_ID["chain02_charge_phone"]
    world = replay_origin_world(chain, "explicit_error", 0)
    assert world.state["orders"]["O1002"]["payment_id"] is None
    with pytest.raises(ReuseError, match="no payment"):
        set_fact(world, chain, "charged_cents", 160000)
    with pytest.raises(ReuseError):
        reconcile_world(chain, "explicit_error", 0,
                        {"payment_status": "captured",
                         "charged_cents": 160000, "order_status": "paid"})


def test_fallback_pilot_config_is_full_grid_narrowed():
    config = load_config(REPO / "configs" / "g2_smoke.json")
    fallback = fallback_pilot_config(config)
    assert fallback["models"] == config["models"]
    assert fallback["chains"] == config["chains"]
    # The pilot runner requires exactly the 5 frozen faults + all 6 policies.
    assert fallback["fault_conditions"] == [
        "no_fault", "explicit_error", "empty_partial",
        "silent_wrong_value", "stale_value"]
    assert fallback["policies"] == ["M0", "M1", "M2", "M3", "M4", "M5"]
    assert fallback["max_new_tokens"] == 256  # Qwen3 floor lifts it to 1024


# ---------------------------------------------------------------------------
# End-to-end fake run: driving, grading, writing, gates
# ---------------------------------------------------------------------------

def test_run_smoke_fake_backend_writes_results_and_passes(tmp_path):
    config = load_config(REPO / "configs" / "g2_smoke.json")
    out = tmp_path / "g2-smoke"
    summary = run_smoke(config, _factory, out)
    analysis = summary["analysis"]

    assert analysis["cells"] == 72
    assert analysis["errors"] == 0
    assert (out / "g2_smoke_cells.jsonl").exists()
    assert (out / "g2_smoke_cells.csv").exists()
    assert (out / "g2_smoke_summary.json").exists()
    assert (out / "g2_smoke_summary.md").exists()
    transcripts = sorted((out / "transcripts").glob("*.json"))
    assert len(transcripts) == 72

    cells = [json.loads(l) for l in
             (out / "g2_smoke_cells.jsonl").read_text().splitlines()]
    assert len(cells) == 72
    assert all(c["parseable"] for c in cells)
    assert all(c["transcript_file"] for c in cells)
    # A correct reader: floor and ceiling hold, yield is 100%.
    assert analysis["answer_yield"]["rate"] == 1.0
    assert analysis["m0_correct"]["rate"] == 1.0
    assert analysis["m5_correct"]["rate"] == 1.0
    assert analysis["correction_in_all_s4_prompts"] is True
    assert analysis["gates"]["pass"] is True

    # The false memories on the dependent fact propagate; the ones on
    # other facts do not. chain01 silent (tracking TRK-999999) goes
    # wrong at S2 and S3 in both models; Qwen3 chain05 silent (stock 4)
    # does the same. Qwen2.5 chain05's stored stock (76/80) is the true
    # DB value — the over-restock is a task failure, not a propagated
    # false belief — so it stays correct. Session 4's correction fixes
    # every propagated case.
    expected_incorrect = {QWEN25: 2, QWEN3: 4}
    for model in (QWEN25, QWEN3):
        row = analysis["per_model"][model]
        assert row["propagation_eligible"]["k"] == expected_incorrect[model]
        assert row["propagation_eligible"]["n"] == 12
        assert row["persistence_s2_to_s3"]["k"] == row["persistence_s2_to_s3"]["n"]
        assert row["correction_resistance_s3_to_s4"]["k"] == 0

    # Session-4 transcripts carry the prompt with the correction in it.
    s4 = [json.loads(p.read_text()) for p in transcripts
          if p.name.endswith("_s4.json")]
    assert len(s4) == 24
    assert all("Verified observation" in t["prompt"] for t in s4)
    assert all(t["correction"] is not None for t in s4)

    # World repairs are recorded: the 6 Qwen2.5 chain05 cells only.
    assert len(summary["world_repairs"]) == 6
    assert all(r["model"] == QWEN25
               and r["chain_id"] == "chain05_restock_lamp"
               for r in summary["world_repairs"])
    assert "76" in (out / "g2_smoke_summary.md").read_text()

    # CSV mirrors the JSONL.
    csv_text = (out / "g2_smoke_cells.csv").read_text()
    assert csv_text.count("\n") == 73  # header + 72 rows


def test_run_smoke_reuse_failure_writes_fallback_config(tmp_path):
    config = default_config()
    rows = [json.loads(l) for l in
            (REPO / "results/qwen3-repilot/pilot_writes.jsonl").read_text().splitlines()]
    rows = rows[:100]  # drops smoke cells
    partial = tmp_path / "qwen3-partial.jsonl"
    partial.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "g2-smoke"
    with pytest.raises(ReuseError):
        run_smoke(config, _factory, out,
                  writes_paths={QWEN3: str(partial)})
    fallback = json.loads((out / "fallback_pilot_config.json").read_text())
    assert fallback["chains"] == config["chains"]
    assert fallback["policies"] == ["M0", "M1", "M2", "M3", "M4", "M5"]


def test_analyse_cells_gate_arithmetic():
    # A hand-built cell set: one M0 miss fails the floor gate, and a
    # group with no eligible session fails the contrast gate.
    cells = []
    for model in (QWEN25, QWEN3):
        for chain in ("chain01_pay_ship_laptop", "chain05_restock_lamp"):
            for fault in ("silent_wrong_value", "stale_value"):
                for session in (2, 3, 4):
                    for policy, correct in (
                            ("M0", True), ("M1", False), ("M5", True)):
                        cells.append({
                            "model": model, "chain_id": chain,
                            "fault_condition": fault, "policy": policy,
                            "session": session, "parseable": True,
                            "correct": correct,
                            "correction_in_prompt": True,
                            "error": None,
                        })
    cells[0]["correct"] = False  # one M0 miss
    analysis = analyse_cells(cells, default_config())
    assert analysis["gates"]["m0_floor_holds"] is False
    assert analysis["gates"]["pass"] is False
    assert analysis["answer_yield"]["rate"] == 1.0
