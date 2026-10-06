"""Reduced-order modelling: POD-Galerkin parametric thermal ROM (Phase 1 prototype).

Parameters mu = [power scale of each heat source..., h_top].  With the film applied at cell
centres the full operator is affine:  K(mu) = K_cond + h_top * Htop + h_bot * Hbot, and the
load is linear in the power scales, so the online solve is a tiny r x r system.
Constant (temperature-independent) k is used for the ROM; nonlinear k(T) needs
hyper-reduction (DEIM) - Phase 3 together with Track 5.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .boundary import DEFAULT_BC, BoundaryConditions
from .mesh import StackGrid
from .thermal import assemble_conduction, boundary_conductance, distribute_sources


class ThermalROM:
    def __init__(self, grid: StackGrid, sources: List[dict], bc: BoundaryConditions = DEFAULT_BC):
        """`bc` must be the same object the full-order solve uses (h_top is a sampled parameter)."""
        self.grid, self.sources = grid, sources
        self.t_amb, self.h_bot = bc.t_ambient_C, bc.h_bottom
        self.K0 = assemble_conduction(grid, grid.cell_property(lambda m: m.k)).tocsr()
        self.Htop = sp.diags(boundary_conductance(grid, 1.0, 0.0))
        self.Hbot = sp.diags(boundary_conductance(grid, 0.0, 1.0))
        self.Q = np.stack([distribute_sources(grid, [dict(s, power_W=1.0)]).ravel() for s in sources], axis=1)
        self.p0 = np.array([s["power_W"] for s in sources])
        self.V = None

    # -- full-order model ------------------------------------------------------------
    def _K(self, h_top):
        return (self.K0 + h_top * self.Htop + self.h_bot * self.Hbot).tocsc()

    def full_solve(self, power_scale, h_top):
        q = self.Q @ (np.asarray(power_scale) * self.p0)
        g = (h_top * self.Htop + self.h_bot * self.Hbot) @ np.ones(self.K0.shape[0])
        return spla.spsolve(self._K(h_top), q + g * self.t_amb)

    # -- offline ---------------------------------------------------------------------
    def train(self, n_samples=40, scale_range=(0.2, 2.0), h_range=(1e3, 2e4), energy=1 - 1e-14, seed=0):
        rng = np.random.default_rng(seed)
        p = len(self.sources)
        mus = np.column_stack([rng.uniform(*scale_range, size=(n_samples, p)),
                               np.exp(rng.uniform(np.log(h_range[0]), np.log(h_range[1]), n_samples))])
        t0 = time.perf_counter()
        S = np.stack([self.full_solve(m[:p], m[p]) - self.t_amb for m in mus], axis=1)
        U, s, _ = np.linalg.svd(S, full_matrices=False)
        r = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), energy) + 1)
        self.V = U[:, :r]
        self.K0r = self.V.T @ (self.K0 @ self.V)
        self.Htr = self.V.T @ (self.Htop @ self.V)
        self.Hbr = self.V.T @ (self.Hbot @ self.V)
        self.Qr = self.V.T @ self.Q
        self.rank, self.train_time = r, time.perf_counter() - t0
        self.singular_values = s
        return self

    # -- online ----------------------------------------------------------------------
    def predict(self, power_scale, h_top):
        """Temperature field (flattened, deg C) from the reduced system."""
        a = np.linalg.solve(self.K0r + h_top * self.Htr + self.h_bot * self.Hbr,
                            self.Qr @ (np.asarray(power_scale) * self.p0))
        return self.t_amb + self.V @ a

    def validate(self, n_test=20, scale_range=(0.2, 2.0), h_range=(1e3, 2e4), seed=123) -> dict:
        rng = np.random.default_rng(seed)
        p = len(self.sources)
        errs, t_full, t_rom = [], 0.0, 0.0
        for _ in range(n_test):
            sc = rng.uniform(*scale_range, size=p); h = rng.uniform(*h_range)
            t0 = time.perf_counter(); Tf = self.full_solve(sc, h); t_full += time.perf_counter() - t0
            t0 = time.perf_counter(); Tr = self.predict(sc, h); t_rom += time.perf_counter() - t0
            errs.append(np.max(np.abs(Tf - Tr)))
        return {"rank": self.rank, "max_abs_err_K": float(max(errs)), "mean_abs_err_K": float(np.mean(errs)),
                "speedup": t_full / max(t_rom, 1e-12), "n_dofs": self.K0.shape[0]}
