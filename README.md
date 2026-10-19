# pkgcodesign (working name)

Open-source framework for **system-level package co-design** — Track: *System & Packaging*.
This repository is the **Phase 1 foundation + a Phase 2 vertical slice** so every sub-team has
something runnable to extend.

> **Status: research prototype.** Material values are *indicative placeholders* and nothing here has
> been validated against Ansys/COMSOL yet. Validation is a Phase 2 deliverable.

## What works today
| Sub-team | Module | State |
|---|---|---|
| FEM Solver Abstraction | `mesh.py` | Structured hex stack builder, TSV/Cu-pillar lattice. Unstructured meshing: planned |
| Material Database | `materials.py`, `data/materials_initial.json` | 8 materials, schema + validation. Values need NIST/CINDAS traceability |
| Thermo-Mechanical Coupling | `thermal.py`, `mechanical.py`, `coupling.py` | FV heat conduction with k(T); Hex8 + incompatible-modes linear elasticity; one-way thermal→mechanical coupling (`scheme="one_way"`). Monolithic: stub |
| Reduced-Order Modeling | `rom.py` | POD-Galerkin parametric thermal ROM (power scales, h_top) |
| Interfaces | `io_formats.py`, `cli.py` | JSON in, HDF5 out, schema validation, device-power hand-off from Tracks 1/3 |

## Quick start
```bash
pip install -e ".[dev]"
pytest -q                                   # full suite (energy balance, 1-D analytic, Timoshenko bilayer, ROM, I/O, validation, units, BCs, coupling)
python examples/run_reference_case.py       # figures -> docs/figures
pkgcodesign run examples/stacked_2tier_tsv.json --out op.h5
pkgcodesign run examples/stacked_2tier_tsv.json --reflow --out reflow.h5
pkgcodesign run examples/stacked_2tier_tsv.json --device-power examples/device_power_example.json
pkgcodesign rom examples/stacked_2tier_tsv.json --out rom_report.json   # train + validate the thermal ROM
```

## Verification done so far
* Thermal: global energy balance to 1e-9 W; 1-D uniform-generation case matches analytic solution to machine precision.
* Mechanical: free thermal expansion is stress-free; bi-material plate curvature agrees with the Timoshenko formula (~0.2 % error, 1-6 cells per layer).
* ROM: rank 5 reproduces full-order temperatures to ~1e-6 K on held-out parameters, ~500x faster online (5,200 cells).

## Known limitations (be honest in reviews)
* Isotropic linear materials; no plasticity/creep (Phase 3). Real organic substrates are anisotropic.
* TSVs are resolved cell-by-cell on a coarse lattice, not homogenised; no sub-grid effects.
* Convective film is applied at the cell centre (keeps the operator affine for the ROM).
* ROM is for constant-k thermal problems; nonlinear k(T) needs hyper-reduction.
* Coupling is one-way: temperature is solved first, then stress and warpage. Displacement does not feed back into temperature, so there is no coupling iteration. The only iteration is the Picard loop on k(T). Two-way (monolithic) coupling is a Phase 2 item and needs a feedback physics to justify it.
* Python/NumPy backend. C++ backend and GPU assembly are Phase 2/3 items.

## Layout
```
src/pkgcodesign/   library          tests/      verification
examples/          models + demo    docs/       data formats, literature plan
```
