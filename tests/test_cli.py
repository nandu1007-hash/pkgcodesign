"""CLI end-to-end: validate, run (with device-power hand-off), and the `rom` subcommand."""
import json
import os

from pkgcodesign.cli import main
from pkgcodesign.io_formats import load_results_h5

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "..", "examples", "stacked_2tier_tsv.json")
DEV = os.path.join(HERE, "..", "examples", "device_power_example.json")


def test_validate_command(capsys):
    assert main(["validate", MODEL]) == 0
    assert "OK:" in capsys.readouterr().out


def test_run_command_writes_results_with_provenance(tmp_path, capsys):
    out = str(tmp_path / "op.h5")
    assert main(["run", MODEL, "--out", out]) == 0
    assert "warpage =" in capsys.readouterr().out
    back = load_results_h5(out)
    assert back["scalars"]["warpage_m"] > 0
    assert json.loads(back["provenance"]["materials_json"])["schema"] == "xtco.materials.v1"


def test_run_with_device_power(tmp_path):
    out = str(tmp_path / "dp.h5")
    assert main(["run", MODEL, "--device-power", DEV, "--out", out]) == 0
    assert os.path.exists(out)


def test_run_reflow(tmp_path, capsys):
    out = str(tmp_path / "rf.h5")
    assert main(["run", MODEL, "--reflow", "--out", out]) == 0
    assert "warpage =" in capsys.readouterr().out


def test_rom_command_reports_and_writes(tmp_path, capsys):
    report = str(tmp_path / "rom.json")
    assert main(["rom", MODEL, "--samples", "30", "--test", "10", "--out", report]) == 0
    out = capsys.readouterr().out
    assert "max_err=" in out and "speedup=" in out
    rep = json.loads(open(report).read())
    assert rep["max_abs_err_K"] < 1e-3
    assert 0 < rep["rank"] <= 30
    assert rep["n_dofs"] > 0
