"""Solver convergence diagnostics are written to the HDF5 results (C5)."""
import os

from pkgcodesign import build_grid, solve_coupled
from pkgcodesign.io_formats import load_package_model, save_results_h5, load_results_h5

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "..", "examples", "stacked_2tier_tsv.json")


def test_hdf5_stores_thermal_iterations_and_residual(tmp_path):
    spec = load_package_model(MODEL)
    grid = build_grid(spec)
    res = solve_coupled(grid, spec)
    p = str(tmp_path / "r.h5")
    save_results_h5(p, grid, res, spec)
    scalars = load_results_h5(p)["scalars"]
    assert scalars["thermal_iterations"] == res["thermal_info"]["iterations"]
    assert scalars["thermal_residual_K"] == res["thermal_info"]["residual"]
    assert scalars["thermal_converged"] == 1


def test_hdf5_records_reproducible_provenance(tmp_path):
    """Regression: results used to carry only a model JSON - no library versions,
    no git commit, and nothing about which materials were actually used."""
    import json
    import platform

    import numpy as np
    import scipy

    from pkgcodesign import __version__, MaterialDB

    spec = load_package_model(MODEL)
    grid = build_grid(spec)
    res = solve_coupled(grid, spec)
    p = str(tmp_path / "r.h5")
    save_results_h5(p, grid, res, spec)
    prov = load_results_h5(p)["provenance"]

    assert prov["python"] == platform.python_version()
    assert prov["pkgcodesign"] == __version__
    assert prov["numpy"] == np.__version__
    assert prov["scipy"] == scipy.__version__
    assert isinstance(prov["git_commit"], str)
    assert json.loads(prov["model_json"]) == spec

    # the embedded materials table must reload into an equivalent database
    mats = json.loads(prov["materials_json"])
    db = MaterialDB.from_dict(mats)
    # only materials the model actually uses are embedded, exactly as the solver saw them
    assert db.names() == sorted(m.name for m in grid.materials)
    assert set(db.names()) <= set(MaterialDB.default().names())
    assert db.get("silicon").k == 148.0
