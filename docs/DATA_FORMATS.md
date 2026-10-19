# Data formats v1 (draft for design-spec v1.0)

| Schema | Container | Purpose |
|---|---|---|
| `xtco.package_model.v1` | JSON | domain, mesh, layers (+TSV lattice), heat sources, BCs, thermo-mech reference temps |
| `xtco.device_power.v1`  | JSON | power blocks handed over from Tracks 1 & 3 (device-level) |
| `xtco.materials.v1`     | JSON | material property tables with `source` and `confidence` per entry |
| `xtco.results.v1`       | HDF5 | `/mesh`, `/fields/{T,von_mises,u,...}`, `/scalars` attrs, embedded model JSON for provenance |

## Result-file provenance (root attrs of `xtco.results.v1`)
Each HDF5 results file is self-describing: `model_json` (exact input model), `materials_json`
(the `xtco.materials.v1` table actually used, including any custom `--materials` DB), `python`,
`pkgcodesign`, `numpy`, `scipy`, `git_commit` (best-effort; `GITHUB_SHA`/`GIT_COMMIT` env or
`git rev-parse`, empty string when unavailable), and `name`.

## Conventions
* Geometry in **micrometres** in JSON, **SI** inside the solver and in HDF5.
* Temperatures in **deg C**; CTE in **ppm/K**; E in **GPa** (JSON only).
* Every HDF5 result embeds the exact input model (`attrs["model_json"]`) for reproducibility.

## Open questions for the Area-3 survey
1. GDSII/OASIS -> power-map extraction: layer/datatype mapping, density-to-power conversion (needs Track 1/3 input).
2. Package-level formats (ODB++, IPC-2581, Gerber) vs. our layered-stack abstraction.
3. Whether to adopt an existing HDF5/XDMF or CGNS convention for fields to ease ParaView/VisIt use.
4. Versioning policy and a JSON-Schema file per format.
