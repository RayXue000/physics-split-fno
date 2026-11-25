"""物理谱工具与 v4 历史参考求解器。

v5 的可微已知物理一步算子位于 ``models/physics_steps.py``。本文件保留
波数、解析符号和此前 K0-K3/V1-V3 复核所需的参考函数；v5 默认训练不会
调用参考求解器生成扰动对。

约定:周期域 [0, L),角波数 k_m = 2π m / L,m = 0..N/2(rfft 单边)。
本文件全部为 C=1(标量通道);矩阵情形(C>1)的符号应返回 [K,C,C],
冻结分支会自动走 matrix_exp 路径。
"""
import math

import torch


def angular_wavenumbers(n, length=1.0, device=None, dtype=torch.float32):
    """rfft 单边角波数 [K=n//2+1]: k_m = 2π m / L。"""
    m = torch.arange(n // 2 + 1, device=device, dtype=dtype)
    return 2.0 * math.pi * m / length


# ---------------------------------------------------------------- 符号 m(k)
def advection_diffusion_symbol(c, nu):
    """u_t + c u_x = ν u_xx  =>  m(k) = -i c k - ν k²。返回 callable(k)->complex。"""
    def m(k):
        kc = k.to(torch.cfloat)
        return -1j * c * kc - nu * kc ** 2
    return m


def burgers_linearized_symbol(a, nu):
    """Burgers 在常数态 a 处的线性化符号 m(k;a) = -i a k - ν k²(论文 Rem.1 示例)。"""
    return advection_diffusion_symbol(a, nu)


def diffusion_symbol(nu):
    """只信扩散(Burgers 灰箱的典型 L0):m0(k) = -ν̄ k²;对流 -u u_x 归入未知 R*。"""
    def m(k):
        return (-nu * k ** 2).to(torch.cfloat)
    return m


# ------------------------------------------------------- Allen-Cahn 反应扩散
def allen_cahn_linearized_symbol(a, nu, reaction_rate=1.0):
    """Allen-Cahn 在常数态 ``a`` 处的连续线性化符号。

    方程为 ``u_t = ν u_xx + λ(u-u³)``，因此
    ``m(k;a) = -νk² + λ(1-3a²)``。该符号用于连续方程解释；若参考
    求解器采用下面的 Strang 分裂，应使用
    :func:`allen_cahn_strang_tangent_target` 作为精确离散 oracle。
    """
    def m(k):
        return (-nu * k ** 2 + reaction_rate * (1.0 - 3.0 * a ** 2)).to(
            torch.cfloat)
    return m


def allen_cahn_reaction_flow(u, dt, reaction_rate=1.0):
    """精确推进点态反应 ODE ``u_t=λ(u-u³)`` 一个时间步。"""
    decay = math.exp(-2.0 * reaction_rate * dt)
    denominator = decay + (1.0 - decay) * u.square()
    return u / denominator.clamp_min(torch.finfo(u.dtype).tiny).sqrt()


def allen_cahn_reaction_derivative(a, dt, reaction_rate=1.0):
    """精确反应流对输入 ``a`` 的导数 ``R'_dt(a)``。"""
    decay = math.exp(-2.0 * reaction_rate * dt)
    denominator = decay + (1.0 - decay) * a ** 2
    return decay / denominator ** 1.5


def allen_cahn_reference_step(u, dt, nu, reaction_rate=1.0, length=1.0):
    """周期 Allen-Cahn 的 Strang 谱分裂参考一步。

    顺序为半步精确扩散、整步精确反应、半步精确扩散。该求解器没有
    Burgers 的 2/3 去混叠截断，因此它也用于检验 V3 是否能从具体求解器
    的可辨识子空间中解耦，而不是只对 Burgers 的频带规则有效。
    """
    n = u.shape[-1]
    k = 2.0 * math.pi * torch.fft.fftfreq(
        n, d=length / n, device=u.device, dtype=u.dtype)
    half_diffusion = torch.exp(-0.5 * nu * k.square() * dt)

    uh = torch.fft.fft(u)
    u_half = torch.fft.ifft(half_diffusion * uh).real
    reacted = allen_cahn_reaction_flow(u_half, dt, reaction_rate)
    return torch.fft.ifft(half_diffusion * torch.fft.fft(reacted)).real


def allen_cahn_strang_tangent_target(a, k, dt, nu, reaction_rate=1.0):
    """常数锚点处 Strang 一步映射的精确离散切向乘子。

    常数态通过第一半步扩散后仍为 ``a``，故每个傅里叶模态满足
    ``DΦ_dt[a](k)=R'_dt(a) exp(-νk²dt)``。这避免把时间离散误差误判为
    K2 经验集的欠覆盖。
    """
    reaction_gain = allen_cahn_reaction_derivative(a, dt, reaction_rate)
    return (reaction_gain * torch.exp(-nu * k.square() * dt)).to(torch.cfloat)


def allen_cahn_growth_envelope(k, dt, nu, reaction_rate=1.0, slack=1.05):
    """Allen-Cahn Strang 映射的全局谱增长包络。

    对 ``λ>0``，反应流最大导数在 ``a=0`` 处为 ``exp(λdt)``。
    """
    if reaction_rate < 0:
        raise ValueError("当前 Allen-Cahn 包络要求 reaction_rate >= 0")
    return slack * math.exp(reaction_rate * dt) * torch.exp(
        -nu * k.square() * dt)


def ks_symbol():
    """Kuramoto–Sivashinsky:u_t = -u u_x - u_xx - u_xxxx => 线性部 m(k)=k²-k⁴。
    低模 k<1 增长(真不稳定带)——包络 ρ 必须 ≥ ‖M*‖,见 envelope_from_symbol。"""
    def m(k):
        return (k ** 2 - k ** 4).to(torch.cfloat)
    return m


# ------------------------------------------------------- 常数锚点冻结目标
def frozen_symbol_target(symbol_fn, k, dt):
    """M*(k;a) = exp(Δt m(k;a))(Prop.3(i))。symbol_fn 已绑定基态 a。
    返回复张量 [K]。Burgers 常数态 f(a)=0 => 该目标精确。"""
    return torch.exp(dt * symbol_fn(k))


def midpoint_symbol_target(symbol_at, a, f_a, k, dt):
    """Prop.3(iii):非平衡常数锚点,在中点 a + Δt/2·f(a) 处取符号,目标偏差 O(Δt³)。
    symbol_at: callable(base)->callable(k);f_a = f(a) 为该常数态的逐点变化率。"""
    a_mid = a + 0.5 * dt * f_a
    return torch.exp(dt * symbol_at(a_mid)(k))


# ---------------------------------------------------------------- 包络 ρ(k)
def envelope_from_symbol(symbol_fn, k, dt, slack=1.05):
    """由已知(或可信)符号构造包络:ρ(k) = slack · |exp(Δt m(k))|
    = slack·exp(Δt Re m(k))。耗散带自动 <1,不稳定带自动 ≥‖M*‖(信息一致)。"""
    return slack * torch.exp(dt * symbol_fn(k).real)


def diffusion_envelope(k, nu, dt, slack=1.05):
    """纯耗散包络 ρ(k) = slack·exp(-ν̄ Δt k²)(K0 级唯一先验)。"""
    return slack * torch.exp(-nu * dt * k ** 2)


def burgers_dealias_rfft_mask(n, device=None):
    """Burgers 参考求解器 2/3 去混叠后可测的 rFFT 模态。

    `burgers_reference_step` 在完整 FFT 上保留 |m| <= (2/3)(N/2),
    因而单边 rFFT 的最大可信索引为 floor(N/3)。带外对流响应已被数值
    求解器主动删除,不能再拿连续全频带 oracle 校准经验符号。偶数网格的
    Nyquist 模也排除,其复相位对实值离散场不可表示。
    """
    mode = torch.arange(n // 2 + 1, device=device)
    mask = mode <= int(math.floor(n / 3.0))
    if n % 2 == 0:
        mask[-1] = False
    return mask


def project_real_rfft_symbol(symbol, n):
    """把连续复符号投影到实值 DFT 可实现子空间。

    偶数网格的 Nyquist 系数必须为实数。该投影用于 K3 集合和 delta_k
    报告,避免只在训练 mask 中隐藏不可达复目标。
    """
    if n % 2 != 0:
        return symbol
    projected = symbol.clone()
    projected[..., -1] = projected[..., -1].real.to(projected.dtype)
    return projected


# ------------------------------------------------- 参考求解器(测量协议用)
def burgers_reference_step(u, dt, nu, substeps=None, length=1.0):
    """真 Burgers 一步演化 Φ_Δt(伪谱 + 积分因子 RK4,与 MATLAB 生成器同法)。

    该函数仅保留给 v4 K2 历史实验复核。v5 训练使用
    ``models.physics_steps.BurgersETD1Step`` 的模型内 JVP，不调用本函数
    生成扰动标签。

    u: [..., N] 实张量(支持批量);返回同形状。substeps=None 按 CFL 自适应。
    """
    n = u.shape[-1]
    k = 2.0 * math.pi * torch.fft.fftfreq(n, d=length / n, device=u.device)
    if substeps is None:
        kmax = float(k.abs().max())
        adv = float(u.abs().max()) * kmax + 1e-12
        dif = nu * kmax ** 2 + 1e-12
        substeps = max(1, int(math.ceil(dt * max(adv, dif) / 0.25)))
    h = dt / substeps
    E = torch.exp(-nu * k ** 2 * h)
    E2 = torch.exp(-nu * k ** 2 * h / 2)
    mask = (k.abs() <= (2.0 / 3.0) * k.abs().max()).to(u.dtype)   # 2/3 去混叠
    g = -0.5j * k

    def nl(uh):
        ur = torch.fft.ifft(uh).real
        return g * torch.fft.fft(ur * ur) * mask

    uh = torch.fft.fft(u)
    for _ in range(substeps):
        a = h * nl(uh)
        b = h * nl(E2 * (uh + a / 2))
        c = h * nl(E2 * uh + b / 2)
        d = h * nl(E * uh + E2 * c)
        uh = E * uh + (E * a + 2 * E2 * (b + c) + d) / 6
    return torch.fft.ifft(uh).real
