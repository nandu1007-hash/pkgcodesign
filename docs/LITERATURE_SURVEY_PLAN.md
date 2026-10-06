# Phase 1 literature survey - division of work

Output of each area: 2-3 page summary + comparison table + recommendation (adopt / wrap / build).

## Area 1 - Open-source FEM/FEA frameworks
Candidates: FEniCS(x), deal.II, MOOSE, FreeFEM, Firedrake, OpenFOAM (FV), CalculiX, Elmer, Kratos.
Criteria (score 0-3): hex/tet meshing & Gmsh import | thermal+elasticity | multiphysics coupling |
per-region material assignment | licence | Python/C++ API | HPC/GPU | community health | package-scale (10^6-10^8 DOF) evidence.

## Area 2 - Material-property databases
Sources: NIST, CINDAS/TPRC, MatWeb, vendor datasheets (substrate/EMC/underfill/TIM), literature for solder.
Fields: k(T), CTE(T), E(T), nu, rho, cp, Tg, anisotropy, source, uncertainty, licence/redistribution rights.
Deliverable: gap list vs. `materials_initial.json` and a provenance policy.

## Area 3 - Data formats & interoperability
JSON/HDF5 (+XDMF/CGNS), GDSII/OASIS, ODB++/IPC-2581, device->package hand-off for Tracks 1 & 3.
See `docs/DATA_FORMATS.md` open questions.
