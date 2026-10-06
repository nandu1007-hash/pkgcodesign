"""Standardised I/O: JSON for model + device-level interfaces, HDF5 for fields (Phase 1 spec v1).

Schemas (documented in docs/DATA_FORMATS.md):
  xtco.package_model.v1  - geometry/materials/BCs  (JSON)
  xtco.device_power.v1   - device-level power blocks from Tracks 1 & 3 (JSON)
  xtco.results.v1        - fields + metadata (HDF5)
"""
from __future__ import annotations

import json
import platform
from importlib import resources
from functools import lru_cache
from typing import Optional

import h5py
import jsonschema
import numpy as np

from .mesh import StackGrid

PACKAGE_SCHEMA = "xtco.package_model.v1"
DEVICE_SCHEMA = "xtco.device_power.v1"
RESULTS_SCHEMA = "xtco.results.v1"


class SchemaError(ValueError):
    pass


@lru_cache(maxsize=None)
def _package_model_schema() -> dict:
    path = resources.files("pkgcodesign.schemas").joinpath("package_model.v1.schema.json")
    with path.open("r") as f:
        return json.load(f)


def validate_package_model(spec: dict) -> None:
    """Structural (JSON Schema, unknown keys rejected) then semantic checks.

    Unknown keys and missing required sections raise SchemaError; nothing is silently defaulted.
    """
    validator = jsonschema.Draft202012Validator(_package_model_schema())
    errors = sorted(validator.iter_errors(spec), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        where = "/".join(str(p) for p in e.path) or "<root>"
        raise SchemaError(f"{where}: {e.message}")

    names = [l["name"] for l in spec["layers"]]
    if len(set(names)) != len(names):
        raise SchemaError("layer names must be unique")
    Lx, Ly = spec["domain_um"]
    for s in spec["heat_sources"]:
        if s["layer"] not in names:
            raise SchemaError(f"heat source references unknown layer '{s['layer']}'")
        x0, y0, x1, y1 = s["rect_um"]
        if not (0 <= x0 < x1 <= Lx and 0 <= y0 < y1 <= Ly):
            raise SchemaError(f"heat source rect outside domain: {s}")


def load_package_model(path: str) -> dict:
    with open(path) as f:
        spec = json.load(f)
    validate_package_model(spec)
    return spec


def merge_device_power(spec: dict, device: dict) -> dict:
    """Apply an xtco.device_power.v1 document (from Tracks 1/3) to a package model."""
    if device.get("schema") != DEVICE_SCHEMA:
        raise SchemaError(f"schema must be '{DEVICE_SCHEMA}'")
    out = dict(spec)
    out["heat_sources"] = [
        {"name": b["name"], "layer": b["layer"], "rect_um": b["rect_um"], "power_W": b["power_W"],
         "cell_layer": b.get("cell_layer", "bottom")} for b in device["blocks"]]
    validate_package_model(out)
    return out


def save_results_h5(path: str, grid: StackGrid, result: dict, spec: Optional[dict] = None) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = RESULTS_SCHEMA
        f.attrs["python"] = platform.python_version()
        f.attrs["name"] = grid.name
        if spec is not None:
            f.attrs["model_json"] = json.dumps(spec)
        g = f.create_group("mesh")
        g["x_edges"], g["y_edges"], g["z_edges"] = grid.x_edges, grid.y_edges, grid.z_edges
        g["material_id"] = grid.mat_id
        g.attrs["materials"] = json.dumps([m.name for m in grid.materials])
        g.attrs["layer_cells"] = json.dumps(grid.layer_cells)
        r = f.create_group("fields")
        for key in ("T", "von_mises", "u", "bottom_uz_residual"):
            if key in result:
                r.create_dataset(key, data=result[key], compression="gzip")
        s = f.create_group("scalars")
        for key in ("warpage_m", "warpage_signed_m"):
            if key in result:
                s.attrs[key] = result[key]
        if "T" in result:
            s.attrs["T_max_C"] = float(np.max(result["T"]))


def load_results_h5(path: str) -> dict:
    out = {}
    with h5py.File(path, "r") as f:
        out["schema"] = f.attrs["schema"]
        out["fields"] = {k: v[...] for k, v in f["fields"].items()}
        out["scalars"] = dict(f["scalars"].attrs)
        out["mesh"] = {k: v[...] for k, v in f["mesh"].items()}
    return out
