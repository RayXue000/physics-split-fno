# -*- coding: utf-8 -*-
"""耗散-色散约束的物理分裂残差 Fourier 神经算子。

.. math::

    G_\theta(u,z) = \Psi_{K,\Delta t}(u,z) + C_\theta(u,z).

``known_step`` 可以是线性 Fourier 乘子，也可以是可微的非线性 RK/ETD
一步算子。训练阶段的 JVP 同时作用于完整模型和该已知物理分支。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.equivariant_correction import EquivariantSpectralCorrection
from models.physics_steps import LinearSpectralStep, ZeroPhysicsStep


class PSRFNO(nn.Module):
    """v5 主模型，同时兼容 v4 的 ``symbol_fn, dt`` 构造方式。"""

    def __init__(
        self,
        symbol_fn=None,
        dt: Optional[float] = None,
        length: float = 1.0,
        modes: int = 16,
        width: int = 32,
        n_layers: int = 4,
        param_channels=(),
        z_channels: int = 0,
        use_correction: bool = True,
        *,
        known_step: Optional[nn.Module] = None,
        correction_scale: float = 1.0,
        correction_output_init_scale=None,
    ):
        super().__init__()
        if known_step is None:
            if symbol_fn is None:
                known_step = ZeroPhysicsStep()
            else:
                if dt is None:
                    raise ValueError("使用 symbol_fn 时必须提供 dt")
                known_step = LinearSpectralStep(symbol_fn, dt, length)
        if not isinstance(known_step, nn.Module):
            raise TypeError("known_step 必须是 torch.nn.Module")

        self.known_step = known_step
        self.use_correction = bool(use_correction)
        self.correction_scale = float(correction_scale)
        self.correction = (
            EquivariantSpectralCorrection(
                modes=modes,
                width=width,
                n_layers=n_layers,
                param_channels=param_channels,
                z_channels=z_channels,
                output_init_scale=correction_output_init_scale,
            )
            if self.use_correction
            else None
        )

    @property
    def frozen(self):
        """旧代码兼容别名；v5 中它可能是非线性已知物理分支。"""
        return self.known_step

    def physics_only(self, u: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.known_step(u, z)

    def correction_only(
        self,
        u: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.use_correction:
            return torch.zeros_like(u)
        return self.correction_scale * self.correction(u, z)

    def forward(
        self,
        u: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.physics_only(u, z) + self.correction_only(u, z)

    def step_fn(self, z: Optional[torch.Tensor] = None):
        """返回单样本 ``[N] -> [N]`` 完整模型闭包，供 JVP 使用。"""

        def function(u_single: torch.Tensor) -> torch.Tensor:
            z_batch = None if z is None else z.unsqueeze(0)
            return self.forward(u_single.unsqueeze(0), z_batch).squeeze(0)

        return function

    def physics_step_fn(self, z: Optional[torch.Tensor] = None):
        """返回单样本已知物理闭包，与模型共享同一个 Fourier 探针。"""

        def function(u_single: torch.Tensor) -> torch.Tensor:
            z_batch = None if z is None else z.unsqueeze(0)
            return self.physics_only(u_single.unsqueeze(0), z_batch).squeeze(0)

        return function

    @torch.no_grad()
    def rollout(
        self,
        u0: torch.Tensor,
        n_steps: int,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """自回归 rollout，返回 ``[n_steps+1, batch, grid]``。"""
        trajectory = [u0]
        state = u0
        for _ in range(n_steps):
            state = self.forward(state, z)
            trajectory.append(state)
        return torch.stack(trajectory, dim=0)


DDPSRFNO = PSRFNO
