"""Gate G7 populated-store sensitivity arm (persistbench/g7_arm.py).

Everything here runs on the deterministic fake backend — no model
download, no GPU. As in tests/test_g2_smoke.py, the fake replies
play a *correct* memory reader: answer from the stored memory when
it records the fact, from the Session-4 correction when delivered,
from a read tool otherwise. The tests prove the seeding mechanism
(distractor construction, truth, disjointness, rendering, merge
semantics), that the scripted grading path is untouched by the
seed, and that the arm runner drives, grades, writes and compares
— not what a real model will do with a populated store. That is
what the Kaggle arm measures.
"""
import json
import re
from pathlib import Path

import pytest

from persistbench.agents import DependentAgent
from persistbench.chains import CHAINS_BY_ID, build_world, resolve_fact
from persistbench.g2_smoke import ReuseError, load_smoke_stores
from persistbench.g7_arm import (
    build_distractor_seed,
    build_comparison,
    default_config,
    distractor_seeds,
    load_baseline,
    load_config,
    run_arm,
    seed_store,
    target_fact_keys,
    validate_config,
)
from persistbench.llm_agent import FakeLLMBackend, LLMDependentAgent
from persistbench.runner import correction_event, run_dependent
from persistbench.g2_smoke import reconcile_world

REPO = Path(__file__).resolve().parent.parent
QWEN25 = "Qwen/Qwen2.5-7B-Instruct"
QWEN3 = "Qwen/Qwen3-8B"
CHAIN01 = "chain01_pay_ship_laptop"
CHAIN05 = "chain05_restock_lamp"

EXPECTED_SEEDS = {
    CHAIN01: {
        "address": "88 Birch Ave",
        "refund_cents": 12000,
        "stock": 40,
        "cancellation_fee_cents": 0,
    },
    CHAIN05: {
        "order_status": "shipped",
        "payment_status": "captured",
        "charged_cents": 150000,
        "tracking": "TRK-000001",
        "address": "88 Birch Ave",
        "refund_cents": 12000,
        "cancellation_fee_cents": 0,
    },
}


# ---------------------------------------------------------------------------
# A correct-reader fake backend (same stand-in as tests/test_g2_smoke.py)
# ---------------------------------------------------------------------------

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

    corr = re.search(r"Verified observation.*?overrides the stored memory for "
                     rf"{re.escape(fact)}: {re.escape(fact)}: (.+)\.",
                     prompt, re.DOTALL)
    if corr:
        raw = corr.group(1)
        try:
            return fmt(int(raw))
        except ValueError:
            return fmt(raw)

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
    config = load_config(REPO / "configs" / "g7_arm.json")
    assert config["models"] == [QWEN25, QWEN3]
    assert config["chains"] == [CHAIN01, CHAIN05]
    assert config["fault_conditions"] == ["silent_wrong_value", "stale_value"]
    assert config["policies"] == ["M0", "M1", "M5"]
    assert config["dependent_mode"] == "revising"
    assert config["seed"] == 0
    assert config["populated_store"] is True
    assert config["baseline_summary"] == "results/g2b-smoke/g2_smoke_summary.json"


@pytest.mark.parametrize("bad", [
    {"populated_store": False},
    {"populated_store": None},
    {"policies": ["M0", "M1", "M3", "M5"]},   # M3 cannot carry seeds
    {"policies": ["M0", "M1", "M2", "M5"]},   # M2 cannot carry seeds
    {"policies": ["M1", "M5"]},               # M0 baseline required
    {"policies": ["M0", "M1", "M4", "M5"]},   # M4 cannot carry seeds
])
def test_config_validation_rejects(bad):
    config = default_config()
    config.update(bad)
    with pytest.raises(ValueError):
        validate_config(config)


# ---------------------------------------------------------------------------
# Distractor seeds: exact content, determinism, disjointness, truth
# ---------------------------------------------------------------------------

def test_seed_exact_content_and_provenance():
    for chain_id, expected in EXPECTED_SEEDS.items():
        seed, provenance = build_distractor_seed(CHAINS_BY_ID[chain_id])
        assert seed == expected
        assert {p["key"] for p in provenance} == set(expected)
        for item in provenance:
            source = CHAINS_BY_ID[item["source_chain"]]
            assert source["intended_facts"][item["key"]] == item["value"]
            assert item["source_chain"] != chain_id


def test_seed_is_deterministic():
    for chain_id in EXPECTED_SEEDS:
        first = build_distractor_seed(CHAINS_BY_ID[chain_id])
        second = build_distractor_seed(CHAINS_BY_ID[chain_id])
        assert first == second
    config = load_config(REPO / "configs" / "g7_arm.json")
    assert distractor_seeds(config) == distractor_seeds(config)


def test_seed_disjoint_from_target_keys_and_arm_stores():
    config = load_config(REPO / "configs" / "g7_arm.json")
    stores = load_smoke_stores(config)
    seeds = distractor_seeds(config)
    for chain_id, entry in seeds.items():
        chain = CHAINS_BY_ID[chain_id]
        assert not set(entry["seed"]) & target_fact_keys(chain)
        assert "task_success" not in entry["seed"]
    # And disjoint from every stored write the arm will actually seed:
    # merging can never shadow (or be shadowed by) the target entry.
    for (model, chain_id, fault, policy), row in stores.items():
        if policy == "M0":
            continue
        seed = seeds[chain_id]["seed"]
        stored_fact_keys = {k for k in row["stored_write"]
                            if not k.startswith("_")}
        assert not set(seed) & stored_fact_keys, (model, chain_id, fault, policy)


def test_seed_values_are_true_in_their_source_worlds():
    # Every distractor value is what the source chain's world holds
    # after its plan runs clean — the fact is true where it came
    # from, not merely copied out of a table.
    for chain_id in EXPECTED_SEEDS:
        _, provenance = build_distractor_seed(CHAINS_BY_ID[chain_id])
        for item in provenance:
            source = CHAINS_BY_ID[item["source_chain"]]
            world = build_world(source)
            for tool, kwargs in source["plan"]:
                getattr(world, tool)(**kwargs)
            assert resolve_fact(world, source, item["key"]) == item["value"], item


def test_seed_store_sizes():
    # Populated store = distractors + the target write's own fact
    # entries (intended facts + task_success): 9 entries per chain.
    config = load_config(REPO / "configs" / "g7_arm.json")
    stores = load_smoke_stores(config)
    seeds = distractor_seeds(config)
    for (model, chain_id, fault, policy), row in stores.items():
        if policy == "M0":
            assert seed_store(row["stored_write"], seeds[chain_id]["seed"]) is None
            continue
        merged = seed_store(row["stored_write"], seeds[chain_id]["seed"])
        fact_entries = {k: v for k, v in merged.items()
                        if not k.startswith("_")}
        assert len(fact_entries) == 9, (model, chain_id, fault, policy)
        # The target entry is byte-identical: seeding changed nothing
        # the target write recorded.
        for key, value in row["stored_write"].items():
            assert merged[key] == value


# ---------------------------------------------------------------------------
# seed_store merge semantics
# ---------------------------------------------------------------------------

def test_seed_store_merge_and_annotations_passthrough():
    stored = {"tracking": "TRK-999999", "task_success": False,
              "_claims": [{"fact": "tracking"}]}
    merged = seed_store(stored, {"stock": 40})
    assert merged["tracking"] == "TRK-999999"
    assert merged["stock"] == 40
    assert merged["_claims"] == [{"fact": "tracking"}]
    assert seed_store(None, {"stock": 40}) is None


def test_seed_store_collision_is_a_reuse_error():
    with pytest.raises(ReuseError, match="collides"):
        seed_store({"stock": 4, "task_success": False}, {"stock": 40})


# ---------------------------------------------------------------------------
# Rendering: the dependent sees distractors AND the target entry
# ---------------------------------------------------------------------------

def test_seeded_prompt_renders_distractors_and_target_entry():
    config = load_config(REPO / "configs" / "g7_arm.json")
    stores = load_smoke_stores(config)
    seeds = distractor_seeds(config)
    chain = CHAINS_BY_ID[CHAIN01]
    row = stores[(QWEN3, CHAIN01, "silent_wrong_value", "M1")]
    memory = seed_store(row["stored_write"], seeds[CHAIN01]["seed"])
    agent = LLMDependentAgent(_factory(QWEN3), memory, mode="revising", seed=0)
    spec = chain["dependents"][0]  # Session 2: report tracking
    prompt = agent._user_message(chain, spec, None, [])
    for key, value in seeds[CHAIN01]["seed"].items():
        assert f"{key}: {value}" in prompt
    # The false target entry is there too, unchanged.
    assert "tracking: TRK-999999" in prompt
    # And the live dependent resolves the target fact from memory:
    # seeding did not hide or alter the entry it must read.
    outcome, grade = run_dependent(chain, reconcile_world(
        chain, "silent_wrong_value", 0, row["truth"])[0], agent, spec,
        correction=None)
    assert outcome["believed"] == "TRK-999999"
    assert grade.correct is False  # the false memory propagates, as in G2b


def test_scripted_dependent_beliefs_unchanged_by_seed():
    # The scripted grading path reads only the target chain's fact
    # keys; seed keys are disjoint, so scripted beliefs (and hence
    # grades) with a seeded store equal the unseeded ones, cell by
    # cell, across the whole arm grid.
    config = load_config(REPO / "configs" / "g7_arm.json")
    stores = load_smoke_stores(config)
    seeds = distractor_seeds(config)
    for (model, chain_id, fault, policy), row in stores.items():
        if policy == "M0":
            continue
        chain = CHAINS_BY_ID[chain_id]
        world, _ = reconcile_world(chain, fault, 0, row["truth"])
        seeded = seed_store(row["stored_write"], seeds[chain_id]["seed"])
        correction = correction_event(chain, world)
        for spec in chain["dependents"]:
            plain_agent = DependentAgent(row["stored_write"], mode="revising")
            seeded_agent = DependentAgent(seeded, mode="revising")
            _, g_plain = run_dependent(chain, world, plain_agent, spec,
                                       correction=correction)
            _, g_seeded = run_dependent(chain, world, seeded_agent, spec,
                                        correction=correction)
            assert g_seeded.correct == g_plain.correct
            assert g_seeded.truth == g_plain.truth


# ---------------------------------------------------------------------------
# End-to-end fake run: driving, grading, writing, comparison
# ---------------------------------------------------------------------------

def test_run_arm_fake_backend_writes_results_and_compares(tmp_path):
    config = load_config(REPO / "configs" / "g7_arm.json")
    out = tmp_path / "g7-arm"
    summary = run_arm(config, _factory, out)
    analysis = summary["analysis"]

    assert summary["completed"] is True
    assert summary["baseline_available"] is True
    assert analysis["cells"] == 72
    assert analysis["errors"] == 0
    assert (out / "g7_arm_cells.jsonl").exists()
    assert (out / "g7_arm_cells.csv").exists()
    assert (out / "g7_arm_summary.json").exists()
    assert (out / "g7_arm_summary.md").exists()
    transcripts = sorted((out / "transcripts").glob("*.json"))
    assert len(transcripts) == 72

    cells = [json.loads(l) for l in
             (out / "g7_arm_cells.jsonl").read_text().splitlines()]
    assert len(cells) == 72
    assert all(c["populated_store"] is True for c in cells)
    seeds = distractor_seeds(config)
    for cell in cells:
        if cell["policy"] == "M0":
            assert cell["distractor_entries"] == 0
            assert cell["store_entries"] == 0
        else:
            assert cell["distractor_entries"] == len(seeds[cell["chain_id"]]["seed"])
            assert cell["store_entries"] == 9

    # A correct reader is not distracted: it reads the target key,
    # so the populated numbers equal the fake-backend G2b numbers
    # (tests/test_g2_smoke.py) exactly — seeding changed the store,
    # not the reader's path to the target entry.
    expected_incorrect = {QWEN25: 2, QWEN3: 4}
    for model in (QWEN25, QWEN3):
        row = analysis["per_model"][model]
        assert row["propagation_eligible"]["k"] == expected_incorrect[model]
        assert row["propagation_eligible"]["n"] == 12
        assert row["persistence_s2_to_s3"]["k"] == row["persistence_s2_to_s3"]["n"]
        assert row["correction_resistance_s3_to_s4"]["k"] == 0
    assert analysis["answer_yield"]["rate"] == 1.0
    assert analysis["m0_correct"]["rate"] == 1.0
    assert analysis["m5_correct"]["rate"] == 1.0

    # The comparison is against the recorded G2b summary: the
    # empty-store rates are the frozen ones, and with a correct
    # reader the deltas are zero.
    comparison = summary["comparison"]
    for model, k in ((QWEN25, 2), (QWEN3, 4)):
        prop = comparison[model]["propagation_eligible"]
        assert prop["empty_store"]["k"] == k
        assert prop["empty_store"]["n"] == 12
        assert prop["populated_store"]["k"] == k
        assert prop["delta"] == 0.0

    # Transcripts: seeded cells show the dependent the seeded store
    # and record both forms; M0 transcripts carry no store.
    seeded_t = [json.loads(p.read_text()) for p in transcripts
                if "_M1_" in p.name or "_M5_" in p.name]
    assert seeded_t and all(
        t["distractor_seed"] for t in seeded_t)
    assert all(
        "stock: 40" in t["prompt"] or "cancellation_fee_cents: 0" in t["prompt"]
        for t in seeded_t if t["chain_id"] == CHAIN01)
    m0_t = [json.loads(p.read_text()) for p in transcripts if "_M0_" in p.name]
    assert m0_t and all(t["stored_write"] is None for t in m0_t)
    assert all(t["distractor_seed"] == [] for t in m0_t)

    md = (out / "g7_arm_summary.md").read_text()
    assert "Populated store (G7)" in md
    assert "Empty store (G2b)" in md
    assert "0.167" in md and "0.333" in md  # recorded G2b rates

    # CSV mirrors the JSONL.
    csv_text = (out / "g7_arm_cells.csv").read_text()
    assert csv_text.count("\n") == 73  # header + 72 rows


def test_run_arm_without_baseline_still_completes(tmp_path):
    config = load_config(REPO / "configs" / "g7_arm.json")
    config["baseline_summary"] = "results/no-such-summary.json"
    out = tmp_path / "g7-arm"
    summary = run_arm(config, _factory, out)
    assert summary["completed"] is True
    assert summary["baseline_available"] is False
    for measures in summary["comparison"].values():
        for measure in measures.values():
            assert measure["empty_store"] is None
            assert measure["delta"] is None
    assert load_baseline(config) is None


def test_run_arm_reuse_failure_writes_fallback_config(tmp_path):
    config = default_config()
    rows = [json.loads(l) for l in
            (REPO / "results/qwen3-repilot/pilot_writes.jsonl").read_text().splitlines()]
    rows = rows[:100]  # drops arm cells
    partial = tmp_path / "qwen3-partial.jsonl"
    partial.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "g7-arm"
    with pytest.raises(ReuseError):
        run_arm(config, _factory, out, writes_paths={QWEN3: str(partial)})
    fallback = json.loads((out / "fallback_pilot_config.json").read_text())
    assert fallback["chains"] == config["chains"]


def test_build_comparison_handles_missing_rates():
    analysis = {"per_model": {QWEN25: {
        "propagation_eligible": {"k": 0, "n": 0, "rate": None,
                                 "ci_lo": None, "ci_hi": None},
        "persistence_s2_to_s3": {"k": 0, "n": 0, "rate": None,
                                 "ci_lo": None, "ci_hi": None},
        "correction_resistance_s3_to_s4": {"k": 0, "n": 0, "rate": None,
                                           "ci_lo": None, "ci_hi": None},
    }}}
    comparison = build_comparison(analysis, None)
    prop = comparison[QWEN25]["propagation_eligible"]
    assert prop["empty_store"] is None
    assert prop["delta"] is None
