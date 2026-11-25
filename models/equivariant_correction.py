"""平移等变 FNO 修正分支 C_θ(论文 §2.2,eq:psr 右项)。

论文的硬性结构要求(Prop.2 单探针恢复的前提):C_θ **联合平移等变**
    C_θ(τ_h u, τ_h z) = τ_h C_θ(u, z).
实现上意味着:
  - 只允许:谱卷积(SpectralConv)、1×1 卷积旁路、逐点非线性、
    标量参数 η̄ 以**常数通道**注入;
  - **禁止**:绝对坐标 / 位置编码 / 任何随 x 变化但不随状态平移的通道。
    ⚠ 这与本仓库 reproduce_deeponet_fno 的标准 FNO 不同 —— 那边把网格
    坐标 x 拼进输入,会破坏等变性,导致 DG_θ[a] 不再是傅里叶乘子,
    Prop.2 的单探针恢复随之失效。selfcheck.py 会做 roll 等变性数值验证。

条件场 z 若存在,必须"随状态一起平移"(如与 u 同网格的外源场),
以通道形式与 u 拼接;标量参数广播成常数通道。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """FNO 1D 谱卷积：rFFT → 低模复乘 → irFFT。

    低频块和零填充用 ``torch.cat`` 组装，避免原地切片写入破坏
    ``torch.func.jvp`` 的前向模式自动微分。
    """

    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.out_ch, self.modes = out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):                                  # x: [B, C, N]
        b, _, n = x.shape
        k = n // 2 + 1
        m = min(self.modes, k)
        x_ft = torch.fft.rfft(x)
        low = torch.einsum(
            "bik,iok->bok",
            x_ft[:, :, :m],
            self.weight[:, :, :m],
        )
        if m < k:
            high = torch.zeros(
                b,
                self.out_ch,
                k - m,
                dtype=low.dtype,
                device=x.device,
            )
            out_ft = torch.cat((low, high), dim=-1)
        else:
            out_ft = low
        return torch.fft.irfft(out_ft, n=n)


class EquivariantSpectralCorrection(nn.Module):
    """C_θ(u; η̄):输入状态 u [B,N](+可选条件场 z [B,Cz,N]),输出修正 [B,N]。

    参数:
      modes/width/n_layers : FNO 超参;
      param_channels       : 标量参数 η̄ 的数值列表(如 [ν̄]),以常数通道注入,
                             让修正网络"知道"名义参数(论文 §2.2)。
    """

    def __init__(
        self,
        modes=16,
        width=32,
        n_layers=4,
        param_channels=(),
        z_channels=0,
        output_init_scale=None,
    ):
        super().__init__()
        self.register_buffer(
            "param_values", torch.tensor(list(param_channels), dtype=torch.float32))
        in_ch = 1 + len(param_channels) + z_channels
        self.lift = nn.Conv1d(in_ch, width, 1)             # 1×1 提升(等变)
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.skip = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.proj1 = nn.Conv1d(width, 128, 1)
        self.proj2 = nn.Conv1d(128, 1, 1)
        self.output_init_scale = output_init_scale
        if output_init_scale is not None:
            nn.init.normal_(self.proj2.weight, mean=0.0, std=float(output_init_scale))
            nn.init.zeros_(self.proj2.bias)
        self.n_layers = n_layers

    def forward(self, u, z=None):                          # u: [B, N]
        b, n = u.shape
        chans = [u.unsqueeze(1)]                           # [B,1,N]
        if self.param_values.numel() > 0:                  # 标量参数 → 常数通道
            const = self.param_values.to(dtype=u.dtype).view(1, -1, 1).expand(b, -1, n)
            chans.append(const)
        if z is not None:
            if z.dim() != 3 or z.shape[0] != b or z.shape[-1] != n:
                raise ValueError("z 必须为 [batch, channel, grid] 并与 u 对齐")
            chans.append(z)                                # 条件场须与 u 同平移
        x = torch.cat(chans, dim=1)
        x = self.lift(x)
        for i in range(self.n_layers):
            x = self.spec[i](x) + self.skip[i](x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
        x = F.gelu(self.proj1(x))
        return self.proj2(x).squeeze(1)                    # [B, N]
