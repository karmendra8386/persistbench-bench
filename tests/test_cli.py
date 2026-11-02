"""CLI smoke tests — fake backend only (plumbing, not results)."""
import json

import pytest

from persistbench import cli


def test_resolve_chain_exact_and_prefix():
    assert cli.resolve_chain("chain01_pay_ship_laptop") == "chain01_pay_ship_laptop"
    assert cli.resolve_chain("chain01") == "chain01_pay_ship_laptop"
    assert cli.resolve_chain("chain08") == "chain08_cancel_desk"


def test_resolve_chain_unknown():
    with pytest.raises(SystemExit):
        cli.resolve_chain("chain99")


def test_run_cell_fake_backend_scores_a_cell():
    config = cli.default_config()
    factory = cli._backend_factory("fake")
    cell = cli.run_cell(
        config, "chain01_pay_ship_laptop", "silent_wrong_value", "M1", factory
    )
    assert cell["model"].startswith("fake/")
    assert cell["chain_id"] == "chain01_pay_ship_laptop"
    assert cell["fault_condition"] == "silent_wrong_value"
    assert cell["policy"] == "M1"
    assert cell["formation"] in (0, 1)
    assert isinstance(cell["stored_write"], dict)
    assert isinstance(cell["truth"], dict)


def test_run_cell_m0_has_no_formation_rate():
    config = cli.default_config()
    factory = cli._backend_factory("fake")
    cell = cli.run_cell(
        config, "chain02_charge_phone", "no_fault", "M0", factory
    )
    assert cell["formation"] is None


def test_main_run_prints_json(capsys):
    rc = cli.main([
        "run", "--chain", "chain01", "--policy", "M1",
        "--fault", "silent_wrong_value", "--backend", "fake",
    ])
    assert rc == 0
    cell = json.loads(capsys.readouterr().out)
    assert cell["chain_id"] == "chain01_pay_ship_laptop"
    assert cell["policy"] == "M1"


def test_main_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert "persistaibench" in capsys.readouterr().out
