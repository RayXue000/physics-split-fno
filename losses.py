# -*- coding: utf-8 -*-
"""PSC-PSR-FNO v5 的数据、耗散与色散损失。"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


def _sample_spectral_energy(value_hat: torch.Tensor) -> torch.Tensor:
    energy = value_hat.abs().square()
    if energy.dim() > 2:
        energy = energy.sum(dim=tuple(range(1, energy.dim() - 1)))
    return energy


def _masked_mean(value: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return value.mean()
    mask = mask.to(device=value.device, dtype=torch.bool)
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(0)
    mask = mask.expand_as(value)
    if not bool(mask.any()):
        return value.sum() * 0.0
    return value.masked_select(mask).mean()


def spatial_relative_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """逐样本空间相对平方误差。"""
    reduce_dims = tuple(range(1, prediction.dim()))
    numerator = (prediction - target).square().sum(dim=reduce_dims)
    denominator = target.square().sum(dim=reduce_dims)
    return (numerator / (denominator + eps)).mean()


def relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """逐样本相对 L2，供评估报告使用。"""
    difference = (prediction - target).reshape(prediction.shape[0], -1)
    reference = target.reshape(target.shape[0], -1)
    return (
        torch.linalg.vector_norm(difference, dim=1)
        / torch.linalg.vector_norm(reference, dim=1).clamp_min(eps)
    ).mean()


def frequency_band_masks(
    n_grid: int,
    edges: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    names: Sequence[str] = ("low", "mid", "high"),
    device=None,
    reference: str = "nyquist",
    modes: Optional[int] = None,
) -> "OrderedDict[str, torch.Tensor]":
    """构造互不相交的 1D 频带。

    ``reference`` 决定归一化基准：

    - ``"nyquist"``（默认，保持历史行为）：按 Nyquist 模态归一化。
    - ``"modes"``：按修正分支**可学习的最高模态** ``modes-1`` 归一化，并把
      超出该范围的模态单独归入 ``"beyond"`` 带。

    为什么需要 ``"modes"``：``SpectralConv1d`` 只对最低 ``modes`` 个模态持有
    可学习复权重，更高模态被硬置零。而按 Nyquist 划带时（edges 0/.25/.5/1，
    modes=16）：

        n=64 : low 0-7,  mid 8-15,  high 16-32  -> high 带 0/17 个模态可学习
        n=128: low 0-15, mid 16-31, high 32-64  -> mid 与 high 全部不可学习

    也就是说 ``L_band`` 把 1/3 的权重压在修正分支没有任何频率选择性参数的
    频带上；模型只能通过 1x1 旁路（空间逐点，频域全通，无法做频率相关滤波）
    和 GELU 谐波间接影响它。优化器为满足该项，只能扭曲它真正能控制的低/中
    频——这与实测到的"高频改善、低频退化 40 倍"一致。

    改用 ``"modes"`` 后三个受约束频带全部落在可学习范围内，``"beyond"`` 段
    只作诊断上报、不参与加权（其权重恒为 0）。

    注意语义变化：``"modes"`` 下的 "high" 指"可学习模态中的高频"，不再是
    数据绝对意义上的高频。数据的真实高频误差仍需通过 ``beyond`` 带监控。
    """
    if len(edges) != len(names) + 1:
        raise ValueError("edges 长度必须比 names 多 1")
    if edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("频带边界必须从 0.0 覆盖到 1.0")
    if any(right <= left for left, right in zip(edges[:-1], edges[1:])):
        raise ValueError("频带边界必须严格递增")

    n_modes = n_grid // 2 + 1
    if reference == "modes":
        if modes is None:
            raise ValueError("reference='modes' 时必须提供 modes")
        maximum_mode = max(min(int(modes), n_modes) - 1, 1)
    elif reference == "nyquist":
        maximum_mode = max(n_grid // 2, 1)
    else:
        raise ValueError("band reference 应为 nyquist 或 modes")

    normalized = torch.arange(n_modes, device=device) / maximum_mode
    masks = OrderedDict()
    for index, name in enumerate(names):
        left, right = float(edges[index]), float(edges[index + 1])
        if index == len(names) - 1:
            mask = (normalized >= left) & (normalized <= right)
        else:
            mask = (normalized >= left) & (normalized < right)
        masks[name] = mask
    if reference == "modes":
        # 谱卷积够不着的模态：只诊断、不加权
        masks["beyond"] = normalized > 1.0
    return masks


def band_relative_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    masks: Optional[Mapping[str, torch.Tensor]] = None,
    edges: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    weights: Optional[Sequence[float]] = None,
    energy_profile: Optional[torch.Tensor] = None,
    band_energy_floor: float = 1e-10,
    denominator_floor: float = 1e-8,
    denominator_mode: str = "original",
    band_tau: Optional[Mapping[str, float]] = None,
):
    """逐频带归一化误差 ``sum_b beta_b E_b/D_b``。

    返回 ``(loss, details)``。若训练集某频带能量低于总能量的
    ``band_energy_floor``，该频带自动关闭并对剩余权重重新归一化。

    ``denominator_mode`` 是 v6a-r2 §9.5.1 的**唯一**受控干预量。两个分母
    逐字实现理论式，不做任何等价改写：

    ``"original"``（A2-ORIGINAL，与历史行为逐位相同）::

        D_orig[i,b] = Y[i,b] + eps_E * Y_tot[i]

    ``"capped"``（A2-CAPPED）::

        D_cap[i,b] = max(Y[i,b], tau_b * Y_tot[i]) + eps_E * Y_tot[i]

    其中 ``eps_E = denominator_floor``，``tau_b = band_tau[b]``。

    **A2-CAPPED 只改分母。** 频带 mask、``modes``、active 判据、``beta_b``
    权重与归一化流程全部走同一段代码，与 A2-ORIGINAL 共用；否则 §9.5.3 的
    ``W_m`` 就不是"分母截断效应"，而是若干改动的混合。``tau_b`` 只允许由
    ``band_calibration`` 从 **train** 切分的目标能量比例导出（§9.5.1：
    "不得根据预测误差选择"），本函数只消费它，不产生它。
    """
    denominator_mode = str(denominator_mode).lower()
    if denominator_mode not in {"original", "capped"}:
        raise ValueError("denominator_mode 应为 original 或 capped")
    n_grid = prediction.shape[-1]
    masks = masks or frequency_band_masks(n_grid, edges=edges, device=prediction.device)
    names = list(masks)
    # "beyond" 是 reference='modes' 下"谱卷积够不着"的诊断段：只上报比值，
    # 权重恒为 0，不进入优化目标。
    weighted_names = [n for n in names if n != "beyond"]
    if weights is None:
        weights = [1.0 / len(weighted_names)] * len(weighted_names)
    if len(weights) != len(weighted_names) or any(weight < 0 for weight in weights):
        raise ValueError(
            f"band weights 必须与受约束频带数一致（{len(weighted_names)}）且非负"
        )
    weights = list(weights)
    if "beyond" in names:
        weights.insert(names.index("beyond"), 0.0)

    error_hat = torch.fft.rfft(prediction - target, dim=-1)
    target_hat = torch.fft.rfft(target, dim=-1)
    error_energy = _sample_spectral_energy(error_hat)
    target_energy = _sample_spectral_energy(target_hat)
    total_target = target_energy.sum(dim=-1).clamp_min(torch.finfo(target.dtype).eps)

    if energy_profile is not None:
        profile = energy_profile.to(prediction.device)
        profile_total = profile.sum().clamp_min(torch.finfo(profile.dtype).eps)
    else:
        profile = None
        profile_total = None

    if denominator_mode == "capped":
        if band_tau is None:
            raise ValueError(
                "denominator_mode='capped' 必须提供 band_tau（由 "
                "band_calibration.derive_band_tau 从 train 切分导出）"
            )
        missing = [name for name in masks if name != "beyond" and name not in band_tau]
        if missing:
            raise ValueError(f"band_tau 缺少频带 {missing}")

    ratios = []
    active = []
    details = OrderedDict()
    capped_fraction = OrderedDict()
    for name, mask in masks.items():
        mask = mask.to(prediction.device)
        error_band = error_energy[:, mask].sum(dim=-1)
        target_band = target_energy[:, mask].sum(dim=-1)
        if denominator_mode == "capped" and name != "beyond":
            tau = float(band_tau[name])
            floor_band = tau * total_target
            effective_band = torch.maximum(target_band, floor_band)
            # 诊断：该 batch 里有多少比例的样本真正被下限接管。全 0 说明
            # tau 对当前数据无效，全 1 说明该频带的分母已与 Y_b 无关。
            capped_fraction[name] = float((floor_band > target_band).float().mean())
        else:
            effective_band = target_band
        denominator = effective_band + denominator_floor * total_target
        ratio = error_band / denominator
        is_active = True
        if profile is not None:
            fraction = profile[mask].sum() / profile_total
            is_active = bool(fraction >= band_energy_floor)
        ratios.append(ratio)
        active.append(is_active)
        details[name] = ratio.mean()

    ratio_matrix = torch.stack(ratios, dim=-1)
    weight_tensor = torch.tensor(weights, device=prediction.device, dtype=prediction.dtype)
    active_tensor = torch.tensor(active, device=prediction.device, dtype=prediction.dtype)
    effective = weight_tensor * active_tensor
    if float(effective.sum()) <= 0.0:
        raise ValueError("所有频带均被能量阈值关闭")
    effective = effective / effective.sum()
    details["active"] = dict(zip(names, active))
    details["weights"] = effective.detach()
    details["denominator_mode"] = denominator_mode
    details["band_tau"] = dict(band_tau) if band_tau is not None else None
    details["capped_sample_fraction"] = dict(capped_fraction)
    return (ratio_matrix * effective).sum(dim=-1).mean(), details


def response_energy_mask(
    energy_profile: torch.Tensor,
    *,
    relative_floor: float = 1e-8,
    max_normalized_mode: float = 1.0,
) -> torch.Tensor:
    """由训练集能谱和可信最高模态构造物理响应活动掩码。"""
    n_modes = energy_profile.numel()
    maximum = max(n_modes - 1, 1)
    normalized = torch.arange(n_modes, device=energy_profile.device) / maximum
    threshold = relative_floor * energy_profile.max().clamp_min(
        torch.finfo(energy_profile.dtype).eps
    )
    return (energy_profile >= threshold) & (normalized <= max_normalized_mode)


def phase_active_mask(
    n_grid: int,
    model_response: torch.Tensor,
    physics_response: torch.Tensor,
    *,
    energy_mask: Optional[torch.Tensor] = None,
    response_floor: float = 1e-3,
) -> torch.Tensor:
    """色散活动带：有数据能量、物理响应非零，并排除 DC/Nyquist。

    掩码故意不由模型模长决定，否则模型可把自身响应压到阈值以下来关闭
    色散损失。模型响应接近零时，方向余弦按 0 处理，对应损失为 1。
    """
    mask = physics_response.abs() >= response_floor
    if energy_mask is not None:
        mask = mask & energy_mask.to(mask.device)
    mask = mask.clone()
    mask[..., 0] = False
    if n_grid % 2 == 0:
        mask[..., -1] = False
    return mask


def dissipation_envelope_loss(
    model_response: torch.Tensor,
    physics_response: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    slack: float = 1.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """单侧对数模长约束，只惩罚模型响应超过可信物理上界。"""
    model_log_gain = torch.log(model_response.abs() + eps)
    upper_log_gain = torch.log(slack * physics_response.abs().detach() + eps)
    violation = F.relu(model_log_gain - upper_log_gain).square()
    return _masked_mean(violation, mask)


def dissipation_target_loss(
    model_response: torch.Tensor,
    physics_response: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """机制上限实验使用的双侧对数模长匹配。"""
    difference = (
        torch.log(model_response.abs() + eps)
        - torch.log(physics_response.abs().detach() + eps)
    ).square()
    return _masked_mean(difference, mask)


def dispersion_direction_loss(
    model_response: torch.Tensor,
    physics_response: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """无需 ``arg``/相位解缠的复方向损失 ``1-cos(delta_phi)``。"""
    reference = physics_response.detach()
    denominator = (model_response.abs() * reference.abs()).clamp_min(eps)
    cosine = (model_response * reference.conj()).real / denominator
    direction_error = 1.0 - cosine.clamp(-1.0, 1.0)
    return _masked_mean(direction_error, mask)


def correction_response_shrink_loss(
    model_response: torch.Tensor,
    physics_response: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Correction-shrink 对照：``||A_G - A_K||^2``。

    由于一步算子写成 ``G = Psi_K + C_theta``，而随机探针响应算子
    ``T_{v,k}(J) = FFT(Jv)(k) * conj(v_hat(k)) / |v_hat(k)|^2`` 对 ``J``
    是线性的，且模型与物理分支使用**同一批探针**，因此逐样本、逐状态、
    逐模态精确成立

        A_G = A_K + A_C.

    于是本损失恒等于 ``||A_C||^2``，即修正分支局部对角响应的纯收缩，
    物理参考值被完全消掉。这正是 ARIS 审稿指出的代数退化，本函数把它
    实现成显式对照臂：若完整方法与本臂打平，说明"物理参考"没有提供
    超出残差收缩的信息。
    """
    difference = (model_response - physics_response.detach()).abs().square()
    return _masked_mean(difference, mask)


def permute_reference_modes(
    physics_response: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """DD-shuffled 对照：对参考响应做模态置换。

    置换保持 ``|A_K|`` 的多重集合与 active 模态数完全不变，只破坏"哪个
    模态对应哪个参考值"这一信息。前向分支中的 ``Psi_K`` 不受影响。
    """
    return physics_response.index_select(-1, permutation.to(physics_response.device))


def spectral_weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    exponent: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """机制 baseline：普通谱加权输出损失 ``sum_k (1+k)^p |e_hat(k)|^2``。

    只按波数放大高频误差，不做逐频带归一化，也不涉及任何 Jacobian 响应。
    用于检验 A2 的收益能否被"任何高频加权"复制（失败判据 1）。
    """
    error_hat = torch.fft.rfft(prediction - target, dim=-1)
    target_hat = torch.fft.rfft(target, dim=-1)
    modes = torch.arange(
        error_hat.shape[-1], device=prediction.device, dtype=prediction.dtype
    )
    weight = (1.0 + modes) ** exponent
    numerator = (weight * error_hat.abs().square()).sum(dim=-1)
    denominator = (weight * target_hat.abs().square()).sum(dim=-1) + eps
    while numerator.dim() > 1:
        numerator = numerator.sum(dim=-1)
        denominator = denominator.sum(dim=-1)
    return (numerator / denominator).mean()


def sobolev_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    order: int = 1,
    length: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """机制 baseline：H^order 相对 Sobolev 损失（谱方式计算导数）。

    通过空间导数放大高频误差，但**不区分**高频误差来自错误耗散还是
    错误相位推进——这正是本方法声称要分离的两种机制。
    """
    n_grid = prediction.shape[-1]
    wavenumbers = (
        2.0
        * math.pi
        / length
        * torch.arange(n_grid // 2 + 1, device=prediction.device, dtype=prediction.dtype)
    )
    error_hat = torch.fft.rfft(prediction - target, dim=-1)
    target_hat = torch.fft.rfft(target, dim=-1)
    total_numerator = 0.0
    total_denominator = 0.0
    for derivative in range(order + 1):
        factor = wavenumbers**derivative
        total_numerator = total_numerator + (factor * error_hat.abs()).square().sum(dim=-1)
        total_denominator = total_denominator + (factor * target_hat.abs()).square().sum(dim=-1)
    while total_numerator.dim() > 1:
        total_numerator = total_numerator.sum(dim=-1)
        total_denominator = total_denominator.sum(dim=-1)
    return (total_numerator / (total_denominator + eps)).mean()


def amplitude_phase_decomposition(
    model_response: torch.Tensor,
    reference_response: torch.Tensor,
):
    """复误差的耗散-色散恒等分解。"""
    dissipation = (model_response.abs() - reference_response.abs()).square()
    dispersion = 2.0 * (
        model_response.abs() * reference_response.abs()
        - (model_response * reference_response.conj()).real
    )
    return dissipation, dispersion


def physics_ramp(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    """物理正则渐增系数。"""
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, (epoch - warmup_epochs + 1) / ramp_epochs))


# ---------------------------------------------------------------------------
# v4 历史脚本的兼容函数。它们不再被 v5 默认训练入口调用。


def correction_loss(model, u_n, u_np1, z=None):
    with torch.no_grad():
        target = u_np1 - model.physics_only(u_n, z)
    return F.mse_loss(model.correction_only(u_n, z), target)


def psc_loss(A, sset, w_k=None, active_mask=None):
    distance = sset.distance_sq(A)
    if w_k is not None:
        distance = w_k * distance
    if active_mask is not None:
        distance = torch.where(active_mask, distance, torch.zeros_like(distance))
    return distance.sum()


def active_band_mask(
    reference,
    energy=None,
    mod_floor=1e-3,
    energy_floor=None,
    n_grid=None,
):
    mask = reference.abs() > mod_floor
    if energy is not None and energy_floor is not None:
        mask = mask & (energy > energy_floor)
    if n_grid is not None and n_grid % 2 == 0:
        mask = mask.clone()
        mask[..., -1] = False
    return mask


def spectral_gain(response):
    if response.dim() == 1:
        return response.abs()
    return torch.linalg.matrix_norm(response, ord=2)


def envelope_loss(
    response,
    rho,
    delta=0.02,
    mu_k=None,
    stab_mask=None,
    margin_mode="relative",
):
    gain = spectral_gain(response)
    threshold = (
        rho * (1.0 - delta)
        if margin_mode == "relative"
        else torch.clamp(rho - delta, min=0.0)
    )
    violation = torch.clamp(gain - threshold, min=0.0).square()
    if mu_k is not None:
        violation = mu_k * violation
    if stab_mask is not None:
        violation = torch.where(stab_mask, violation, torch.zeros_like(violation))
    return violation.sum()


def total_loss(
    l_data,
    l_psc=None,
    l_traj=None,
    l_env=None,
    lambda_psc=1.0,
    lambda_env=1.0,
):
    total = l_data
    if l_psc is not None:
        total = total + lambda_psc * l_psc
    if l_traj is not None:
        total = total + lambda_psc * l_traj
    if l_env is not None:
        total = total + lambda_env * l_env
    return total
