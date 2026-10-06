import json
import os
import numpy as np
from pkgcodesign import build_grid, ThermalROM, solve_coupled
from pkgcodesign.io_formats import (load_package_model, merge_device_power, save_results_h5,
                                    load_results_h5, SchemaError, validate_package_model)
import pytest

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "..", "examples", "stacked_2tier_tsv.json")
DEV = os.path.join(HERE, "..", "examples", "device_power_example.json")


def test_rom_accuracy_and_speedup():
    spec = load_package_model(MODEL)
    g = build_grid(spec)
    rom = ThermalROM(g, spec["heat_sources"]).train(n_samples=60)
    rep = rom.validate(n_test=10)
    assert rep["max_abs_err_K"] < 1e-3
    assert rep["rank"] < 60 and rep["speedup"] > 10


def test_hdf5_roundtrip(tmp_path):
    spec = load_package_model(MODEL)
    g = build_grid(spec)
    res = solve_coupled(g, spec)
    p = str(tmp_path / "r.h5")
    save_results_h5(p, g, res, spec)
    back = load_results_h5(p)
    assert np.allclose(back["fields"]["T"], res["T"])
    assert back["scalars"]["warpage_m"] == pytest.approx(res["warpage_m"])


def test_device_power_merge_and_schema_errors():
    spec = load_package_model(MODEL)
    merged = merge_device_power(spec, json.load(open(DEV)))
    assert len(merged["heat_sources"]) == 3 and merged["heat_sources"][0]["power_W"] == 0.40
    bad = dict(spec, heat_sources=[{"layer": "nope", "rect_um": [0, 0, 1, 1], "power_W": 1}])
    with pytest.raises(SchemaError):
        validate_package_model(bad)


def test_monolithic_not_yet_implemented():
    spec = load_package_model(MODEL)
    with pytest.raises(NotImplementedError):
        solve_coupled(build_grid(spec), spec, scheme="monolithic")
