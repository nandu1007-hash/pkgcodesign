"""Steady-state 3D heat conduction, finite-volume on the structured stack grid.

K T = q + hA*T_amb, with face conductances from harmonic (series half-cell) averaging.
Convective film (top/bottom) is applied at the cell centre: G_b = h * A. This keeps the
operator affine in h, which the ROM relies on. Side walls are adiabatic.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import StackGrid


def cell_volumes(grid: StackGrid) -> np.ndarray:
    return grid.dx[:, None, None] * grid.dy[None, :, None] * grid.dz[None, None, :]


def assemble_conduction(grid: StackGrid, k: np.ndarray) -> sp.csr_matrix:
    """Pure conduction operator (symmetric positive semi-definite)."""
    nx, ny, nz = grid.shape
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    rows, cols, vals = [], [], []

    def faces(i, j, G):
        rows.extend([i.ravel(), j.ravel(), i.ravel(), j.ravel()])
        cols.extend([i.ravel(), j.ravel(), j.ravel(), i.ravel()])
        g = G.ravel()
        vals.extend([g, g, -g, -g])

    A = dy[None, :, None] * dz[None, None, :]
    R = dx[:-1, None, None] / (2 * k[:-1]) + dx[1:, None, None] / (2 * k[1:])
    faces(idx[:-1], idx[1:], A / R)

    A = dx[:, None, None] * dz[None, None, :]
    R = dy[None, :-1, None] / (2 * k[:, :-1]) + dy[None, 1:, None] / (2 * k[:, 1:])
    faces(idx[:, :-1], idx[:, 1:], A / R)

    A = dx[:, None, None] * dy[None, :, None]
    R = dz[None, None, :-1] / (2 * k[:, :, :-1]) + dz[None, None, 1:] / (2 * k[:, :, 1:])
    faces(idx[:, :, :-1], idx[:, :, 1:], A / R)

    n = nx * ny * nz
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n)).tocsr()


def boundary_conductance(grid: StackGrid, h_top: float, h_bottom: float) -> np.ndarray:
    """Vector hA for each cell (zero for interior cells)."""
    nx, ny, nz = grid.shape
    A = (grid.dx[:, None] * grid.dy[None, :])
    g = np.zeros((nx, ny, nz))
    g[:, :, -1] += h_top * A
    g[:, :, 0] += h_bottom * A
    return g.ravel()


def distribute_sources(grid: StackGrid, sources: List[dict]) -> np.ndarray:
    """Convert rectangular power sources (W) to a per-cell heat vector [W].

    Source: {"layer": str, "rect_um": [x0,y0,x1,y1], "power_W": float, "cell_layer": "bottom"|"top"|"all"}
    Power is spread over overlapped cells in proportion to overlap area.
    """
    nx, ny, nz = grid.shape
    q = np.zeros((nx, ny, nz))
    for s in sources:
        x0, y0, x1, y1 = (v * 1e-6 for v in s["rect_um"])
        ox = np.clip(np.minimum(x1, grid.x_edges[1:]) - np.maximum(x0, grid.x_edges[:-1]), 0, None)
        oy = np.clip(np.minimum(y1, grid.y_edges[1:]) - np.maximum(y0, grid.y_edges[:-1]), 0, None)
        w = np.outer(ox, oy)
        if w.sum() == 0:
            raise ValueError(f"Heat source {s.get('name', '?')} lies outside the domain")
        w = w / w.sum()
        ks, ke = grid.layer_cells[s["layer"]]
        where = s.get("cell_layer", "bottom")
        ks_sel = {"bottom": [ks], "top": [ke - 1], "all": list(range(ks, ke))}[where]
        for kk in ks_sel:
            q[:, :, kk] += s["power_W"] * w / len(ks_sel)
    return q


def solve_thermal(grid: StackGrid, sources: List[dict], t_ambient: float = 25.0,
                  h_top: float = 3000.0, h_bottom: float = 100.0,
                  T_guess: Optional[np.ndarray] = None, k_cell: Optional[np.ndarray] = None,
                  temperature_dependent: bool = True, tol: float = 1e-6, max_iter: int = 30):
    """Solve steady-state T [deg C]. Picard iteration on k(T) when temperature_dependent.

    Returns (T, info) with T of shape (nx, ny, nz).
    """
    q = distribute_sources(grid, sources).ravel()
    g = boundary_conductance(grid, h_top, h_bottom)
    rhs = q + g * t_ambient

    def solve_with(kc):
        K = assemble_conduction(grid, kc) + sp.diags(g)
        return spla.spsolve(K.tocsc(), rhs).reshape(grid.shape)

    if k_cell is not None:
        return solve_with(k_cell), {"iterations": 1, "residual": 0.0}

    T = np.full(grid.shape, t_ambient) if T_guess is None else T_guess
    if not temperature_dependent:
        return solve_with(grid.cell_property(lambda m: m.k)), {"iterations": 1, "residual": 0.0}

    res = np.inf
    for it in range(1, max_iter + 1):
        kc = np.empty(grid.shape)
        for mi, m in enumerate(grid.materials):
            sel = grid.mat_id == mi
            kc[sel] = m.k_of_T(T[sel])
        T_new = solve_with(kc)
        res = float(np.max(np.abs(T_new - T)))
        T = T_new
        if res < tol:
            break
    return T, {"iterations": it, "residual": res}
