"""pkgcodesign: open-source package-level thermo-mechanical co-design framework."""
from .materials import Material, MaterialDB
from .mesh import StackGrid, build_grid
from .thermal import solve_thermal
from .mechanical import solve_mechanical
from .coupling import solve_coupled, solve_reflow
from .rom import ThermalROM

__version__ = "0.1.0"
__all__ = [
    "Material", "MaterialDB", "StackGrid", "build_grid", "solve_thermal",
    "solve_mechanical", "solve_coupled", "solve_reflow", "ThermalROM",
]
