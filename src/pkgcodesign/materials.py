"""Material property database (Phase 1 deliverable: initial material property database).

All properties are isotropic, linear. Temperature dependence is currently limited
to a power law for thermal conductivity: k(T) = k0 * (T/300 K) ** k_exp.
Plasticity/creep/anisotropy are Phase 3 items.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from importlib import resources
from typing import Dict, Iterable

import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    k: float          # thermal conductivity at 300 K [W/(m.K)]
    cte: float        # coefficient of thermal expansion [ppm/K]
    E: float          # Young's modulus [GPa]
    nu: float         # Poisson's ratio [-]
    rho: float        # density [kg/m3]
    cp: float         # specific heat [J/(kg.K)]
    k_exp: float = 0.0
    source: str = ""
    confidence: str = "unverified"

    def k_of_T(self, T_celsius):
        """Thermal conductivity at temperature T [deg C] (scalar or ndarray)."""
        T_K = np.asarray(T_celsius, dtype=float) + 273.15
        return self.k * (T_K / 300.0) ** self.k_exp

    @property
    def alpha(self) -> float:
        """CTE in 1/K."""
        return self.cte * 1e-6

    @property
    def E_Pa(self) -> float:
        return self.E * 1e9

    def validate(self) -> Iterable[str]:
        """Plausibility checks in the units of UNITS (the stated units, not SI).

        The bounds are deliberately wide, so they only catch unit mistakes (1/K written where
        ppm/K is expected, g/cm3 written where kg/m3 is expected, Pa written where GPa is expected).
        They are not a substitute for provenance checks.
        """
        problems = []
        if not (0 < self.k <= 3000): problems.append(f"{self.name}: k out of range [W/(m.K)]")
        if not (0.01 <= self.E < 1500): problems.append(f"{self.name}: E out of range [GPa]")
        if not (-0.99 < self.nu < 0.5): problems.append(f"{self.name}: nu must be in (-1, 0.5)")
        if not (0.01 <= self.cte < 500): problems.append(f"{self.name}: cte out of range [ppm/K]")
        if not (100 <= self.rho <= 30000): problems.append(f"{self.name}: rho out of range [kg/m3]")
        if not (100 <= self.cp <= 10000): problems.append(f"{self.name}: cp out of range [J/(kg.K)]")
        return problems


# Units every entry in a materials file is written in. A file that declares different units is
# rejected, because the numbers would be silently wrong.
UNITS = {"k": "W/(m.K)", "cte": "ppm/K", "E": "GPa", "nu": "-", "rho": "kg/m3", "cp": "J/(kg.K)"}
MATERIALS_SCHEMA = "xtco.materials.v1"


class MaterialDB:
    def __init__(self, materials: Dict[str, Material] | None = None):
        self._m: Dict[str, Material] = dict(materials or {})

    @classmethod
    def default(cls) -> "MaterialDB":
        with resources.files("pkgcodesign.data").joinpath("materials_initial.json").open("r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_json(cls, path: str) -> "MaterialDB":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "MaterialDB":
        if d.get("schema") != MATERIALS_SCHEMA:
            raise ValueError(f"materials schema must be '{MATERIALS_SCHEMA}', got {d.get('schema')!r}")
        if d.get("units") != UNITS:
            raise ValueError(f"materials units must be {UNITS}, got {d.get('units')}")
        db = cls()
        for name, p in d["materials"].items():
            db.add(Material(name=name, **p))
        return db

    def add(self, mat: Material) -> None:
        problems = list(mat.validate())
        if problems:
            raise ValueError("; ".join(problems))
        self._m[mat.name] = mat

    def get(self, name: str) -> Material:
        try:
            return self._m[name]
        except KeyError:
            raise KeyError(f"Unknown material '{name}'. Available: {sorted(self._m)}") from None

    def names(self):
        return sorted(self._m)

    def to_json(self, path: str) -> None:
        out = {"schema": MATERIALS_SCHEMA, "units": UNITS,
               "materials": {n: {k: v for k, v in asdict(m).items() if k != "name"} for n, m in self._m.items()}}
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
