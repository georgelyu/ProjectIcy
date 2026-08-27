"""IceFlow2D flow solver and CPU thermal validation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import IceFlowConfig, RigidBoundaryScheme, create_iceflow_config
from .stefan2d import Stefan2DConfig, Stefan2DSnapshot, Stefan2DSolver
from .thermal import (
    LatticeScales,
    PhaseChangeProperties,
    ThermalBoundary,
    ThermalBoundarySet,
    ThermalConfig,
    phase_change_active_water_target_cells,
    phase_change_enthalpy_numpy,
    phase_change_water_target_cells,
    recover_temperature_and_liquid_fraction_numpy,
)

if TYPE_CHECKING:
    from .simulator import IceFlow2D

__all__ = [
    "IceFlowConfig",
    "LatticeScales",
    "PhaseChangeProperties",
    "RigidBoundaryScheme",
    "IceFlow2D",
    "Stefan2DConfig",
    "Stefan2DSnapshot",
    "Stefan2DSolver",
    "ThermalBoundary",
    "ThermalBoundarySet",
    "ThermalConfig",
    "create_iceflow_config",
    "phase_change_active_water_target_cells",
    "phase_change_enthalpy_numpy",
    "phase_change_water_target_cells",
    "recover_temperature_and_liquid_fraction_numpy",
]


def __getattr__(name: str):
    """Load the CUDA solver only when it is actually requested.

    This keeps the NumPy-only Stefan validation usable on machines without a
    Taichi/CUDA runtime while preserving ``from iceflow2d import IceFlow2D``.
    """

    if name == "IceFlow2D":
        from .simulator import IceFlow2D

        return IceFlow2D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
