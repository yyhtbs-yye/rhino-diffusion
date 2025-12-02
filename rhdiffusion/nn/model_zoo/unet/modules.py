import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal timestep embedding, as in DDPM / Transformer PE.

    Input:  t   (B,) or (B,1)
    Output: emb (B, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B, 1)
        if t.dim() == 2 and t.size(1) == 1:
            t = t.squeeze(1)
        assert t.dim() == 1, f"t must be 1D (batch,), got {t.shape}"

        half_dim = self.dim // 2
        device = t.device
        emb_factor = math.log(10000) / (half_dim - 1)
        exps = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        # (B, half_dim)
        emb = t[:, None] * exps[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class CondFuseBlock(nn.Module):
    """
    Condition fusion block for a given resolution:
      - Resize cond to x resolution.
      - Concat along channels.
      - Conv + GN + SiLU to project back to x_channels.
    This is a separate nn.Module per resolution.
    """
    def __init__(self, x_channels: int, cond_in_channels: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(x_channels + cond_in_channels, x_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, x_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, C_cond, Hc, Wc)
        cond_resized = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat([x, cond_resized], dim=1)
        h = self.conv(h)
        h = self.norm(h)
        h = self.act(h)
        return h


class ResBlock(nn.Module):
    """
    Time-conditioned residual block:
        GN -> SiLU -> Conv
        + time MLP bias
        GN -> SiLU -> Conv
        + residual (1x1 conv if channels differ)
    """
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int | None = None, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim is not None else None
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # inject time embedding after first conv
        if self.time_mlp is not None and t_emb is not None:
            temb = self.time_mlp(t_emb)            # (B, out_ch)
            h = h + temb[..., None, None]          # broadcast to (B, out_ch, H, W)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.skip(x)


class Downsample(nn.Module):
    """Strided conv downsample (2x)."""
    def __init__(self, in_ch: int, out_ch: int | None = None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsample + conv (2x)."""
    def __init__(self, in_ch: int, out_ch: int | None = None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
