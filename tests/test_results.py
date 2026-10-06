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
