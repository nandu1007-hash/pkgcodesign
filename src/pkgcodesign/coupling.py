"""Thermo-mechanical coupling framework (Phase 1: iterative/staggered; monolithic is Phase 2).

Operating case : steady T(x) with Picard iteration on k(T)  ->  thermal stress / warpage.
Reflow case    : uniform cool-down from solder solidification to room temperature.
"""
from __future__ import annotations

import numpy as np

from .mesh import StackGrid
from .thermal import solve_thermal
from .mechanical import solve_mechanical


def solve_coupled(grid: StackGrid, spec: dict, scheme: str = "iterative") -> dict:
    if scheme == "monolithic":
        raise NotImplementedError("Monolithic coupled solve is a Phase 2 deliverable; use scheme='iterative'.")
    if scheme != "iterative":
        raise ValueError(f"unknown scheme '{scheme}'")
    bc = spec.get("boundary", {})
    T, tinfo = solve_thermal(grid, spec["heat_sources"], t_ambient=bc.get("t_ambient_C", 25.0),
                             h_top=bc.get("h_top", 3000.0), h_bottom=bc.get("h_bottom", 100.0))
    t_ref = spec.get("thermomech", {}).get("t_ref_C", 25.0)
    mech = solve_mechanical(grid, T, t_ref)
    return {"T": T, "thermal_info": tinfo, **mech}


def solve_reflow(grid: StackGrid, spec: dict) -> dict:
    tm = spec.get("thermomech", {})
    t_hot, t_cold = tm.get("t_reflow_C", 217.0), tm.get("t_room_C", 25.0)
    T = np.full(grid.shape, t_cold)
    return {"T": T, **solve_mechanical(grid, T, t_ref=t_hot)}
