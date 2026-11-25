# -*- coding: utf-8 -*-
"""无训练参数、分辨率无关、可微的已知物理一步算子。"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Optional

import torch
import torch.nn as nn

from physics import angular_wavenumbers


class ZeroPhysicsStep(nn.Module):
    """普通 FNO 基线使用：不注入物理分支。"""

    name = "none"

    def forward(self, u: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.zeros_like(u)


class IdentityPhysicsStep(nn.Module):
    """可选残差网络基线：以恒等映射为固定分支。"""

    name = "identity"

    def forward(self, u: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        return u


class LinearSpectralStep(nn.Module):
    """``u -> F^-1[exp(dt*m(k))*F(u)]`` 的固定线性传播。"""

    def __init__(
        self,
        symbol_fn: Callable[[torch.Tensor], torch.Tensor],
        dt: float,
        length: float = 1.0,
        name: str = "linear_spectral",
    ):
        super().__init__()
        self.symbol_fn = symbol_fn
        self.dt = float(dt)
        self.length = float(length)
        self.name = name
        self._cache: dict[tuple, torch.Tensor] = {}

    def propagator(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (n, str(device), dtype)
        if key not in self._cache:
            k = angular_wavenumbers(n, self.length, device=device, dtype=dtype)
            symbol = self.symbol_fn(k)
            if symbol.dim() == 1:
                propagator = torch.exp(self.dt * symbol)
            else:
                propagator = torch.linalg.matrix_exp(self.dt * symbol)
            self._cache[key] = propagator.detach()
        return self._cache[key]

    def forward(self, u: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = u.shape[-1]
        propagator = self.propagator(n, u.device, u.dtype)
        u_hat = torch.fft.rfft(u, dim=-1)
        if propagator.dim() == 1:
            shape = (1,) * (u_hat.dim() - 1) + (-1,)
            return torch.fft.irfft(u_hat * propagator.view(shape), n=n, dim=-1)
        if u.dim() != 3:
            raise ValueError("矩阵 Fourier 符号要求输入 [batch, channel, grid]")
        out_hat = torch.einsum("kij,bjk->bik", propagator, u_hat)
        return torch.fft.irfft(out_hat, n=n, dim=-1)


class AdvectionDiffusionStep(LinearSpectralStep):
    """周期平流扩散方程的精确线性一步。"""

    def __init__(self, velocity: float, nu: float, dt: float, length: float = 1.0):
        self.velocity = float(velocity)
        self.nu = float(nu)

        def symbol(k: torch.Tensor) -> torch.Tensor:
            complex_dtype = torch.complex128 if k.dtype == torch.float64 else torch.complex64
            kc = k.to(complex_dtype)
            return -1j * self.velocity * kc - self.nu * kc.square()

        super().__init__(symbol, dt, length, name="advection_diffusion_exact")


class BurgersETD1Step(nn.Module):
    """周期黏性 Burgers 已知部分的一阶 ETD 步。

    ``advection_scale=1`` 表示名义完整 Burgers 物理；设为 0 时只保留扩散。
    全部运算均为 PyTorch 张量运算，因此可直接对输入做 JVP。
    """

    name = "burgers_etd1"

    def __init__(
        self,
        nu: float,
        dt: float,
        length: float = 1.0,
        advection_scale: float = 1.0,
        dealias: bool = True,
    ):
        super().__init__()
        self.nu = float(nu)
        self.dt = float(dt)
        self.length = float(length)
        self.advection_scale = float(advection_scale)
        self.dealias = bool(dealias)
        self._cache: dict[tuple, tuple[torch.Tensor, ...]] = {}

    def _operators(self, n: int, device: torch.device, dtype: torch.dtype):
        key = (n, str(device), dtype)
        if key not in self._cache:
            k = 2.0 * math.pi * torch.fft.fftfreq(
                n,
                d=self.length / n,
                device=device,
                dtype=dtype,
            )
            argument = -self.nu * k.square() * self.dt
            regular = argument.abs() > 1e-6
            denominator = torch.where(regular, argument, torch.ones_like(argument))
            quotient = torch.expm1(argument) / denominator
            taylor = 1.0 + 0.5 * argument + argument.square() / 6.0
            phi1 = torch.where(regular, quotient, taylor)
            propagator = torch.exp(argument)
            if self.dealias:
                cutoff = (2.0 / 3.0) * k.abs().max()
                mask = (k.abs() <= cutoff).to(dtype)
            else:
                mask = torch.ones_like(k)
            self._cache[key] = (k, propagator, phi1, mask)
        return self._cache[key]

    def forward(self, u: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = u.shape[-1]
        k, propagator, phi1, mask = self._operators(n, u.device, u.dtype)
        u_hat = torch.fft.fft(u, dim=-1)
        nonlinear_hat = (
            -0.5j
            * self.advection_scale
            * k
            * torch.fft.fft(u.square(), dim=-1)
            * mask
        )
        next_hat = propagator * u_hat + self.dt * phi1 * nonlinear_hat
        return torch.fft.ifft(next_hat, dim=-1).real


def build_physics_step(
    pde: str,
    physics_cfg: Mapping,
    dt: float,
    *,
    enabled: bool = True,
) -> nn.Module:
    """从 YAML 配置构造已知物理分支。"""
    if not enabled:
        return ZeroPhysicsStep()

    pde_key = pde.lower().replace("-", "_")
    length = float(physics_cfg.get("length", 1.0))
    mode = str(physics_cfg.get("mode", "full")).lower()

    if pde_key == "advection":
        if mode not in {"full", "linear"}:
            raise ValueError("advection 的 physics.mode 应为 full/linear")
        return AdvectionDiffusionStep(
            velocity=float(physics_cfg.get("velocity", 1.0)),
            nu=float(physics_cfg.get("nu", 0.0)),
            dt=dt,
            length=length,
        )

    if pde_key == "burgers":
        if mode not in {"full", "diffusion", "partial"}:
            raise ValueError("burgers 的 physics.mode 应为 full/diffusion/partial")
        advection_scale = (
            float(physics_cfg.get("advection_scale", 1.0))
            if mode == "full"
            else 0.0
        )
        return BurgersETD1Step(
            nu=float(physics_cfg.get("nu", physics_cfg.get("nu_bar", 1e-3))),
            dt=dt,
            length=length,
            advection_scale=advection_scale,
            dealias=bool(physics_cfg.get("dealias", True)),
        )

    if pde_key == "kdv":
        # u_t + 6 u u_x + delta u_xxx = 0，周期域。
        #
        # 线性色散部分的符号：对 u_xxx，F[u_xxx] = (i k)^3 u_hat = -i k^3 u_hat，
        # 代入 u_t = -delta u_xxx 得 m(k) = +i delta k^3，故
        #     A_K(k) = exp(dt * i * delta * k^3),   |A_K(k)| = 1。
        #
        # 这正是本方法需要的**色散测试床**：模长恒为 1（无耗散），复方向随
        # 波数按 k^3 变化（频率依赖色散）。
        #
        # mode 的含义：
        #   linear  已知物理只含线性色散项，非线性项 6 u u_x 属未知，交给
        #           修正分支。这是"部分已知物理"设定，也是唯一能真正检验
        #           色散约束的设定。
        #   full    已知物理同时含非线性项（ETD1），此时 Psi_K 接近真实一步
        #           算子，修正分支只需学离散误差 —— 与 advection 的精确物理
        #           分支一样会使任务退化，只应作为机制上限实验。
        if mode not in {"linear", "full"}:
            raise ValueError("kdv 的 physics.mode 应为 linear（部分物理，推荐）或 full（机制上限）")
        delta = float(physics_cfg.get("delta", 0.02))
        if mode == "linear":
            def kdv_symbol(k: torch.Tensor) -> torch.Tensor:
                return 1j * delta * k.to(torch.cfloat) ** 3

            return LinearSpectralStep(
                symbol_fn=kdv_symbol,
                dt=dt,
                length=length,
                name="kdv_linear_dispersion",
            )
        raise NotImplementedError(
            "kdv 的 physics.mode='full'（含非线性 ETD1）尚未实现。"
            "第一轮只需要 mode='linear' 的部分物理设定——它才是能检验色散"
            "约束的设定；full 只会像 advection 的精确物理分支那样使任务退化。"
        )

    raise ValueError(f"尚未实现 {pde!r} 的已知物理一步算子")

