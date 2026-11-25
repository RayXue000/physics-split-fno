# -*- coding: utf-8 -*-
"""训练状态上的随机 Fourier JVP 局部响应估计。"""
from __future__ import annotations

import math
from typing import Optional

import torch

try:
    from torch.func import jvp as _func_jvp
except ImportError:  # pragma: no cover
    _func_jvp = None


def _run_jvp(function, state: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """保留参数梯度的输入 JVP。"""
    if _func_jvp is not None:
        _, tangent = _func_jvp(function, (state,), (direction,))
        return tangent
    _, tangent = torch.autograd.functional.jvp(
        function,
        (state,),
        (direction,),
        create_graph=True,
    )
    return tangent


def make_random_phase_probe(
    n_grid: int,
    device=None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
):
    """构造实值、全频带、等幅随机相位探针。

    DC 和偶数网格 Nyquist 只有实值自由度，不能表示任意复相位。旧实现把
    两者都固定为 ``+1``，于是

        E[xi_Nyq * conj(xi_DC)] = 1 != 0

    破坏了命题 2 要求的正交条件 ``E[xi_q conj(xi_k)] = delta_qk``。对一般
    非平移不变的局部 Jacobian，这会让 DC 的响应估计混入 ``J_{0,Nyq}``、
    Nyquist 的估计混入 ``J_{Nyq,0}``。两个端点虽已被排除出色散损失，却仍
    进入耗散掩码，因此耗散端点用的是有偏估计。

    修复：给这两个实自由度各自独立采样 Rademacher ``+-1``。此时

        E[xi_0] = E[xi_Nyq] = 0,
        E[xi_0 conj(xi_Nyq)] = E[xi_0] E[xi_Nyq] = 0,
        E[|xi_0|^2] = E[|xi_Nyq|^2] = 1,

    与内部模态的均匀随机相位一起满足同一组正交条件，命题 2 对全部 rFFT
    模态成立。探针仍是实值（irfft 的共轭对称性不受影响）。
    """
    n_modes = n_grid // 2 + 1
    phase = 2.0 * math.pi * torch.rand(
        n_modes,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    spectrum = torch.exp(1j * phase.to(complex_dtype))

    # 只有实自由度的模态数：DC 恒有；偶数网格还多一个 Nyquist。
    n_real_modes = 2 if n_grid % 2 == 0 else 1
    rademacher = torch.randint(
        0,
        2,
        (n_real_modes,),
        device=device,
        generator=generator,
    ).to(dtype) * 2.0 - 1.0
    spectrum[0] = rademacher[0].to(complex_dtype)
    if n_grid % 2 == 0:
        spectrum[-1] = rademacher[1].to(complex_dtype)

    probe = torch.fft.irfft(spectrum, n=n_grid)
    return probe, spectrum


def response_from_probe(
    step_fn,
    state: torch.Tensor,
    probe: torch.Tensor,
    probe_spectrum: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """由一次 JVP 得到 Fourier Jacobian 对角的随机估计样本。"""
    tangent = _run_jvp(step_fn, state, probe)
    tangent_spectrum = torch.fft.rfft(tangent)
    denominator = probe_spectrum.abs().square().clamp_min(eps)
    return tangent_spectrum * probe_spectrum.conj() / denominator


def random_phase_response(
    step_fn,
    state: torch.Tensor,
    n_probes: int = 1,
    generator: Optional[torch.Generator] = None,
    return_samples: bool = False,
):
    """用 ``R`` 个随机探针估计一个一步算子的局部 Fourier 对角响应。"""
    if n_probes < 1:
        raise ValueError("n_probes 必须 >= 1")
    samples = []
    for _ in range(n_probes):
        probe, probe_hat = make_random_phase_probe(
            state.shape[-1],
            device=state.device,
            dtype=state.dtype,
            generator=generator,
        )
        samples.append(response_from_probe(step_fn, state, probe, probe_hat))
    stacked = torch.stack(samples, dim=0)
    mean = stacked.mean(dim=0)
    if not return_samples:
        return mean
    variance = (stacked - mean).abs().square().mean(dim=0)
    return mean, {"samples": stacked, "variance": variance}


def paired_random_phase_response(
    model_step_fn,
    physics_step_fn,
    state: torch.Tensor,
    n_probes: int = 1,
    generator: Optional[torch.Generator] = None,
    return_samples: bool = False,
):
    """用同一组探针同时估计模型与已知物理响应。

    配对探针不会消除非对角耦合造成的有限 ``R`` 方差，但能减少比较两条
    响应时由不同随机方向引入的额外方差。
    """
    if n_probes < 1:
        raise ValueError("n_probes 必须 >= 1")
    model_samples = []
    physics_samples = []
    for _ in range(n_probes):
        probe, probe_hat = make_random_phase_probe(
            state.shape[-1],
            device=state.device,
            dtype=state.dtype,
            generator=generator,
        )
        model_samples.append(
            response_from_probe(model_step_fn, state, probe, probe_hat)
        )
        physics_samples.append(
            response_from_probe(physics_step_fn, state, probe, probe_hat)
        )
    model_stack = torch.stack(model_samples, dim=0)
    physics_stack = torch.stack(physics_samples, dim=0)
    model_mean = model_stack.mean(dim=0)
    physics_mean = physics_stack.mean(dim=0)
    if not return_samples:
        return model_mean, physics_mean
    diagnostics = {
        "model_samples": model_stack,
        "physics_samples": physics_stack,
        "model_variance": (model_stack - model_mean).abs().square().mean(dim=0),
        "physics_variance": (
            physics_stack - physics_mean
        ).abs().square().mean(dim=0),
    }
    return model_mean, physics_mean, diagnostics


# ---------------------------------------------------------------------------
# v4 历史接口。常数锚点/平移等变时它仍是精确单探针恢复。


def response_single_probe(step_fn, u_anchor, probe=None):
    if probe is None:
        direction, direction_hat = make_random_phase_probe(
            u_anchor.shape[-1],
            device=u_anchor.device,
            dtype=u_anchor.dtype,
        )
    else:
        direction, direction_hat = probe
    return response_from_probe(step_fn, u_anchor, direction, direction_hat)


def response_per_mode(step_fn, u, k_indices, length=1.0):
    n_grid = u.shape[-1]
    x = torch.arange(n_grid, device=u.device, dtype=u.dtype) / n_grid * length
    responses = []
    leakages = []
    for mode in k_indices:
        wavenumber = 2.0 * math.pi * mode / length
        cosine_jvp = _run_jvp(step_fn, u, torch.cos(wavenumber * x))
        sine_jvp = _run_jvp(step_fn, u, torch.sin(wavenumber * x))
        combined = cosine_jvp + 1j * sine_jvp
        spectrum = torch.fft.fft(combined) / n_grid
        selected = spectrum[mode].abs().square()
        total = spectrum.abs().square().sum()
        responses.append(spectrum[mode])
        leakages.append(1.0 - selected / (total + 1e-12))
    return torch.stack(responses), torch.stack(leakages)


def leakage_weight(base_weight, leakage):
    return base_weight * torch.clamp(1.0 - leakage, min=0.0).detach()

