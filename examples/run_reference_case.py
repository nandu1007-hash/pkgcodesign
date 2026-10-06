"""Reproduce the Phase-1 reference demo: operating thermal map, reflow warpage, ROM accuracy.

    python examples/run_reference_case.py   ->  docs/figures/*.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pkgcodesign import build_grid, solve_coupled, solve_reflow, ThermalROM
from pkgcodesign.boundary import BoundaryConditions
from pkgcodesign.io_formats import load_package_model

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

spec = load_package_model(os.path.join(HERE, "stacked_2tier_tsv.json"))
grid = build_grid(spec)
op = solve_coupled(grid, spec)
rf = solve_reflow(grid, spec)
print(f"Operating: Tmax={op['T'].max():.1f} C, warpage={op['warpage_m']*1e6:.2f} um")
print(f"Reflow   : warpage={rf['warpage_m']*1e6:.2f} um")

xc, yc, zc = grid.cell_centers()
# 1) vertical cross-section through the hotspot
j = 5
fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
im = ax.pcolormesh(grid.x_edges * 1e3, grid.z_edges * 1e3, op["T"][:, j, :].T, cmap="inferno", shading="flat")
for name, (k0, k1) in grid.layer_cells.items():
    ax.axhline(grid.z_edges[k0] * 1e3, color="w", lw=0.4, alpha=0.6)
    ax.text(0.02, 0.5 * (grid.z_edges[k0] + grid.z_edges[k1 - 1 + 1]) * 1e3, name, color="w", fontsize=5, va="center")
ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]"); ax.set_title("Steady-state temperature, cross-section (deg C)", fontsize=9)
fig.colorbar(im, ax=ax, label="T [C]"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "thermal_section.png")); plt.close(fig)

# 2) reflow warpage map
fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
im = ax.pcolormesh(grid.x_edges * 1e3, grid.y_edges * 1e3, rf["bottom_uz_residual"].T * 1e6, cmap="coolwarm", shading="gouraud" if False else "auto")
ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
ax.set_title("Reflow warpage, substrate bottom (um)", fontsize=9)
fig.colorbar(im, ax=ax, label="plane-fit residual [um]"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "warpage_map.png")); plt.close(fig)

# 3) ROM
rom = ThermalROM(grid, spec["heat_sources"], bc=BoundaryConditions.from_spec(spec)).train(n_samples=60)
rep = rom.validate(n_test=20)
print("ROM:", rep)
fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=200)
s = rom.singular_values / rom.singular_values[0]
ax.semilogy(np.arange(1, len(s) + 1), s, "o-", ms=3)
ax.axvline(rom.rank, color="r", ls="--", lw=0.8); ax.text(rom.rank + 0.5, 0.3, f"rank {rom.rank}", color="r", fontsize=7)
ax.set_xlabel("POD mode"); ax.set_ylabel("normalised singular value"); ax.set_title("POD spectrum (60 snapshots)", fontsize=9)
ax.set_ylim(1e-17, 2); fig.tight_layout(); fig.savefig(os.path.join(OUT, "rom_pod.png")); plt.close(fig)

import json
json.dump({"Tmax_C": float(op["T"].max()), "operating_warpage_um": op["warpage_m"] * 1e6,
           "reflow_warpage_um": rf["warpage_m"] * 1e6, "rom": rep, "n_cells": grid.n_cells},
          open(os.path.join(OUT, "demo_metrics.json"), "w"), indent=2)
