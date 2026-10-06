"""Structured hexahedral mesh builder for layered package stacks (FEM Solver Abstraction sub-team).

Phase 1 scope: rectilinear grid, one material per cell, optional TSV / copper-pillar
arrays marked on a cell-index lattice. Unstructured meshing (Gmsh/netgen) plugs in
behind the same `StackGrid` interface in Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .materials import Material, MaterialDB


@dataclass
class StackGrid:
    x_edges: np.ndarray                 # [m], length nx+1
    y_edges: np.ndarray
    z_edges: np.ndarray
    mat_id: np.ndarray                  # (nx, ny, nz) int index into `materials`
    materials: List[Material]
    layer_cells: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # name -> (k_start, k_stop)
    name: str = "package"

    @property
    def shape(self):
        return self.mat_id.shape

    @property
    def n_cells(self):
        return int(np.prod(self.shape))

    @property
    def dx(self): return np.diff(self.x_edges)
    @property
    def dy(self): return np.diff(self.y_edges)
    @property
    def dz(self): return np.diff(self.z_edges)

    def cell_property(self, fn) -> np.ndarray:
        """Map a per-material scalar function onto cells, e.g. lambda m: m.E."""
        vals = np.array([fn(m) for m in self.materials])
        return vals[self.mat_id]

    def cell_centers(self):
        xc = 0.5 * (self.x_edges[1:] + self.x_edges[:-1])
        yc = 0.5 * (self.y_edges[1:] + self.y_edges[:-1])
        zc = 0.5 * (self.z_edges[1:] + self.z_edges[:-1])
        return xc, yc, zc


def build_grid(spec: dict, db: MaterialDB | None = None) -> StackGrid:
    """Build a StackGrid from a `xtco.package_model.v1` dict (see docs/DATA_FORMATS.md)."""
    db = db or MaterialDB.default()
    Lx, Ly = (v * 1e-6 for v in spec["domain_um"])
    nx, ny = spec["mesh"]["nx"], spec["mesh"]["ny"]
    x_edges = np.linspace(0.0, Lx, nx + 1)
    y_edges = np.linspace(0.0, Ly, ny + 1)

    mats: List[Material] = []

    def mid(name: str) -> int:
        m = db.get(name)
        if m not in mats:
            mats.append(m)
        return mats.index(m)

    z_edges = [0.0]
    layer_cells: Dict[str, Tuple[int, int]] = {}
    slabs = []  # (nz, base_mat, tsv_dict or None)
    k0 = 0
    for layer in spec["layers"]:
        nz = int(layer["nz"])
        t = layer["thickness_um"] * 1e-6
        z_edges.extend(z_edges[-1] + t * np.arange(1, nz + 1) / nz)
        layer_cells[layer["name"]] = (k0, k0 + nz)
        slabs.append((nz, mid(layer["material"]), layer.get("tsv")))
        k0 += nz

    mat_id = np.zeros((nx, ny, k0), dtype=int)
    k = 0
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    for nz, base, tsv in slabs:
        mat_id[:, :, k:k + nz] = base
        if tsv:
            p, o = int(tsv["pitch_cells"]), int(tsv.get("offset_cells", 0))
            mask = ((I - o) % p == 0) & ((J - o) % p == 0) & (I >= o) & (J >= o)
            mat_id[mask, k:k + nz] = mid(tsv["material"])
        k += nz

    return StackGrid(x_edges, y_edges, np.array(z_edges), mat_id, mats, layer_cells,
                     name=spec.get("name", "package"))
