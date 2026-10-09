"""Coupled falling-ice melting solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import IceBodyConfig, IceFlowConfig, create_iceflow_config
from .config import (
    LatticeScales,
    MovingBodyThermalScheme,
    PhaseChangeProperties,
    ThermalBoundary,
    ThermalBoundarySet,
    ThermalConfig,
    ThermalScheme,
)

if TYPE_CHECKING:
    from .simulator import IceFlow2D

__all__ = [
    "IceBodyConfig",
    "IceFlowConfig",
    "LatticeScales",
    "MovingBodyThermalScheme",
    "PhaseChangeProperties",
    "IceFlow2D",
    "ThermalBoundary",
    "ThermalBoundarySet",
    "ThermalConfig",
    "ThermalScheme",
    "create_iceflow_config",
]


def __getattr__(name: str):
    """Load Taichi kernels only when the CUDA solver is requested."""

    if name == "IceFlow2D":
        from .simulator import IceFlow2D

        return IceFlow2D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
