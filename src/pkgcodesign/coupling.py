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
    bc = spec["boundary"]
    T, tinfo = solve_thermal(grid, spec["heat_sources"], t_ambient=bc["t_ambient_C"],
                             h_top=bc["h_top"], h_bottom=bc["h_bottom"])
    t_ref = spec["thermomech"]["t_ref_C"]
    mech = solve_mechanical(grid, T, t_ref)
    return {"T": T, "thermal_info": tinfo, **mech}


def solve_reflow(grid: StackGrid, spec: dict) -> dict:
    tm = spec["thermomech"]
    t_hot, t_cold = tm["t_reflow_C"], tm["t_room_C"]
    T = np.full(grid.shape, t_cold)
    return {"T": T, **solve_mechanical(grid, T, t_ref=t_hot)}
