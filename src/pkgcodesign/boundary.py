"""Thermal boundary conditions shared by the full-order solver, the coupled driver, and the ROM.

The numeric defaults live here and nowhere else. Models always provide these values through
the package-model JSON (`boundary` section, required by the schema), so the defaults only
apply when a caller builds a solver directly from Python.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryConditions:
    t_ambient_C: float = 25.0   # ambient temperature [deg C]
    h_top: float = 3000.0       # film coefficient on the top face [W/(m2.K)]
    h_bottom: float = 100.0     # film coefficient on the bottom face [W/(m2.K)]

    @classmethod
    def from_spec(cls, spec: dict) -> "BoundaryConditions":
        """Read the `boundary` section of a validated xtco.package_model.v1 dict."""
        bc = spec["boundary"]
        return cls(t_ambient_C=float(bc["t_ambient_C"]), h_top=float(bc["h_top"]),
                   h_bottom=float(bc["h_bottom"]))


DEFAULT_BC = BoundaryConditions()
