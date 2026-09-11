"""Coupled falling-ice melting solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import IceFlowConfig, create_iceflow_config
from .config import (
    LatticeScales,
    MovingBodyThermalScheme,
    MovingBodyThermalTotals,
    PhaseChangeProperties,
    ThermalBoundary,
    ThermalBoundarySet,
    ThermalConfig,
)

if TYPE_CHECKING:
    from .simulator import IceFlow2D

__all__ = [
    "IceFlowConfig",
    "LatticeScales",
    "MovingBodyThermalScheme",
    "MovingBodyThermalTotals",
    "PhaseChangeProperties",
    "IceFlow2D",
    "ThermalBoundary",
    "ThermalBoundarySet",
    "ThermalConfig",
    "create_iceflow_config",
]


def __getattr__(name: str):
    """Load Taichi kernels only when the CUDA solver is requested."""

    if name == "IceFlow2D":
        from .simulator import IceFlow2D

        return IceFlow2D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
