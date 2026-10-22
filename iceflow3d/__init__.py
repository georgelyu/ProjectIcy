"""Three-dimensional disappearing-melt extension of iceflow2d_v3."""

from .config import (
    IceBodyConfig,
    IceFlowConfig,
    LatticeScales,
    PhaseChangeProperties,
    ThermalConfig,
    create_iceflow_config,
)

__all__ = [
    "IceFlow3D",
    "IceBodyConfig",
    "IceFlowConfig",
    "LatticeScales",
    "PhaseChangeProperties",
    "ThermalConfig",
    "create_iceflow_config",
]


def __getattr__(name):
    if name == "IceFlow3D":
        from .simulator import IceFlow3D

        return IceFlow3D
    raise AttributeError(name)
