"""Boundary conditions come from the model spec, and the FOM and ROM share them (C2)."""
import copy
import json
import os

import numpy as np

from pkgcodesign import ThermalROM, build_grid
from pkgcodesign.boundary import BoundaryConditions
from pkgcodesign.io_formats import validate_package_model
from pkgcodesign.thermal import solve_thermal

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "..", "examples", "stacked_2tier_tsv.json")


def _spec(t_ambient_C=25.0):
    with open(MODEL) as f:
        spec = json.load(f)
    spec = copy.deepcopy(spec)
    spec["boundary"]["t_ambient_C"] = t_ambient_C
    validate_package_model(spec)
    return spec


def test_from_spec_reads_all_boundary_fields():
    spec = _spec(t_ambient_C=40.0)
    bc = BoundaryConditions.from_spec(spec)
    assert bc == BoundaryConditions(t_ambient_C=40.0, h_top=15000.0, h_bottom=100.0)


def test_rom_matches_fom_at_non_default_ambient():
    """Regression for C2: the ROM used to hard-code t_ambient=25, giving a 25 K offset."""
    spec = _spec(t_ambient_C=50.0)
    bc = BoundaryConditions.from_spec(spec)
    grid = build_grid(spec)
    rom = ThermalROM(grid, spec["heat_sources"], bc=bc).train(n_samples=30)

    scale, h = np.ones(len(spec["heat_sources"])), bc.h_top
    T_fom, _ = solve_thermal(grid, spec["heat_sources"], t_ambient=bc.t_ambient_C, h_top=h,
                             h_bottom=bc.h_bottom, temperature_dependent=False)
    T_rom = rom.predict(scale, h).reshape(grid.shape)
    assert np.max(np.abs(T_rom - T_fom)) < 1e-3
    assert T_fom.min() > 50.0 - 1e-9  # every cell sits above the 50 C ambient, so the case is really non-default
