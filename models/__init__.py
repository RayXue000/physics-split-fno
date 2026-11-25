"""PSC-PSR-FNO v5：可微已知物理一步算子 + 等变 FNO 修正。"""

from models.equivariant_correction import EquivariantSpectralCorrection
from models.frozen_branch import FrozenPhysicsBranch
from models.physics_steps import (
    AdvectionDiffusionStep,
    BurgersETD1Step,
    IdentityPhysicsStep,
    LinearSpectralStep,
    ZeroPhysicsStep,
    build_physics_step,
)
from models.psr_fno import DDPSRFNO, PSRFNO

__all__ = [
    "PSRFNO",
    "DDPSRFNO",
    "EquivariantSpectralCorrection",
    "FrozenPhysicsBranch",
    "LinearSpectralStep",
    "AdvectionDiffusionStep",
    "BurgersETD1Step",
    "ZeroPhysicsStep",
    "IdentityPhysicsStep",
    "build_physics_step",
]
