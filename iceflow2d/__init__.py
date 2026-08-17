"""Sharp-boundary, non-thermal 2D rigid ice--air--water coupling."""

from .config import (
    DamBreakConfig,
    IceFlowConfig,
    Material,
    Mode,
    RigidBoundaryScheme,
    create_dambreak_config,
    create_iceflow_config,
)
from .simulator import IceFlow2D


Simulator2D = IceFlow2D

__all__ = [
    "DamBreakConfig",
    "IceFlowConfig",
    "Material",
    "Mode",
    "RigidBoundaryScheme",
    "IceFlow2D",
    "Simulator2D",
    "create_dambreak_config",
    "create_iceflow_config",
]

