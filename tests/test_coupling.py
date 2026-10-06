"""Coupling scheme naming and behaviour (C4): the Phase 1 driver is one-way, thermal -> mechanical."""
import os

import numpy as np
import pytest

from pkgcodesign import build_grid, solve_coupled
from pkgcodesign.io_formats import load_package_model
from pkgcodesign.thermal import solve_thermal
from pkgcodesign.boundary import BoundaryConditions

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "..", "examples", "stacked_2tier_tsv.json")


def test_one_way_scheme_runs():
    spec = load_package_model(MODEL)
    res = solve_coupled(build_grid(spec), spec, scheme="one_way")
    assert {"T", "warpage_m", "von_mises"} <= set(res)


def test_temperature_is_independent_of_mechanics():
    """One-way coupling: T equals the thermal solve alone, because nothing feeds u back into T."""
    spec = load_package_model(MODEL)
    grid = build_grid(spec)
    res = solve_coupled(grid, spec, scheme="one_way")
    bc = BoundaryConditions.from_spec(spec)
    T_alone, _ = solve_thermal(grid, spec["heat_sources"], t_ambient=bc.t_ambient_C,
                               h_top=bc.h_top, h_bottom=bc.h_bottom)
    assert np.array_equal(res["T"], T_alone)


def test_iterative_alias_warns_and_matches_one_way():
    """Regression for C4: 'iterative' overstated the scheme. It still works but warns."""
    spec = load_package_model(MODEL)
    grid = build_grid(spec)
    with pytest.warns(DeprecationWarning, match="one_way"):
        old = solve_coupled(grid, spec, scheme="iterative")
    new = solve_coupled(grid, spec, scheme="one_way")
    assert np.array_equal(old["T"], new["T"])


def test_unknown_scheme_is_rejected():
    spec = load_package_model(MODEL)
    with pytest.raises(ValueError):
        solve_coupled(build_grid(spec), spec, scheme="staggered")


def test_monolithic_is_still_a_stub():
    spec = load_package_model(MODEL)
    with pytest.raises(NotImplementedError):
        solve_coupled(build_grid(spec), spec, scheme="monolithic")
