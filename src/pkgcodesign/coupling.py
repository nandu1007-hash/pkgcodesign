"""Thermo-mechanical coupling driver.

Phase 1 is one-way (scheme "one_way"): the steady temperature field is solved first, then
thermal stress and warpage are computed from it. Nothing feeds displacement back into
temperature, so there is no coupling iteration to converge. The only iteration is the
Picard loop on k(T) inside solve_thermal. A monolithic (two-way) solve is a Phase 2 item.

Operating case : steady T(x) with Picard iteration on k(T)  ->  thermal stress / warpage.
Reflow case    : uniform cool-down from solder solidification to room temperature.
"""
from __future__ import annotations

import warnings

import numpy as np

from .boundary import BoundaryConditions
from .mesh import StackGrid
from .thermal import solve_thermal
from .mechanical import solve_mechanical


def solve_coupled(grid: StackGrid, spec: dict, scheme: str = "one_way") -> dict:
    if scheme == "iterative":
        warnings.warn("scheme='iterative' is deprecated; the Phase 1 driver is one-way, use "
                      "scheme='one_way'", DeprecationWarning, stacklevel=2)
        scheme = "one_way"
    if scheme == "monolithic":
        raise NotImplementedError("Monolithic coupled solve is a Phase 2 deliverable; use scheme='one_way'.")
    if scheme != "one_way":
        raise ValueError(f"unknown scheme '{scheme}'")
    bc = BoundaryConditions.from_spec(spec)
    T, tinfo = solve_thermal(grid, spec["heat_sources"], t_ambient=bc.t_ambient_C,
                             h_top=bc.h_top, h_bottom=bc.h_bottom)
    t_ref = spec["thermomech"]["t_ref_C"]
    mech = solve_mechanical(grid, T, t_ref)
    return {"T": T, "thermal_info": tinfo, **mech}


def solve_reflow(grid: StackGrid, spec: dict) -> dict:
    tm = spec["thermomech"]
    t_hot, t_cold = tm["t_reflow_C"], tm["t_room_C"]
    T = np.full(grid.shape, t_cold)
    return {"T": T, **solve_mechanical(grid, T, t_ref=t_hot)}
