import torch
import torch.nn as nn
import torch.nn.functional as F

from rhdiffusion.nn.model_zoo.unet.modules import SinusoidalPosEmb, ResBlock, Downsample, Upsample, CondFuseBlock

def silu_kaiming_init_(module: nn.Module):
    """
    Kaiming/He init tailored for SiLU/Swish-style networks:
      - Treat SiLU as a ReLU-like activation (common in the literature).
      - Use fan_in so forward activations keep a stable variance.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_in",
            nonlinearity="relu",  # SiLU treated as ReLU-like
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# ----------------------------------------------------------
# U-Net for diffusion
# ----------------------------------------------------------

class UNet2DModel(nn.Module):
    """
    U-Net backbone for diffusion models.

    - Input:
        x    : (B, in_channels,  H,  W)
        t    : (B,) or (B,1) timesteps
        cond : (B, cond_in_channels, H, W) optional spatial condition

    - Conditioning:
        At each resolution (both down and up):
          x_res = Fuse_i( concat( x_res, resize(cond) ) )

        Each Fuse_i is a separate CondFuseBlock with its own parameters.
    """
    def __init__(self, in_channels, out_channels,
                 base_channels=64, channel_mults=(1, 2, 4, 8),
                 num_res_blocks=2, time_emb_dim=None, cond_in_channels=None,
                 groups=8,):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.cond_in_channels = cond_in_channels

        # ---- time embedding ----
        if time_emb_dim is None:
            time_emb_dim = base_channels * 4
        self.time_dim = time_emb_dim

        self.time_embed = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ---- channels per resolution ----
        channels = [base_channels * m for m in channel_mults]

        # ---- initial conv ----
        self.init_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # ---- down path ----
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.cond_fuse_down = nn.ModuleList() if cond_in_channels is not None else None

        in_ch = channels[0]
        for i, ch in enumerate(channels):
            if ch % groups != 0:
                raise ValueError(f"channels={ch} not divisible by groups={groups}")
            
            # residual blocks for this resolution
            resblocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                resblocks.append(ResBlock(in_ch, ch, time_emb_dim=time_emb_dim, groups=groups))
                in_ch = ch
            self.down_blocks.append(resblocks)

            # conditioning fuse at this resolution
            if cond_in_channels is not None:
                self.cond_fuse_down.append(CondFuseBlock(ch, cond_in_channels, groups=groups))

            # downsample, except at the last (lowest) resolution
            if i != len(channels) - 1:
                out_ch = channels[i + 1]
                self.downsamples.append(Downsample(in_ch, out_ch))
                in_ch = out_ch

        # ---- bottleneck ----
        self.mid_block1 = ResBlock(in_ch, in_ch, time_emb_dim=time_emb_dim, groups=groups)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_emb_dim=time_emb_dim, groups=groups)

        # ---- up path ----
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.cond_fuse_up = nn.ModuleList() if cond_in_channels is not None else None

        # we go from lowest resolution back to highest
        for i in reversed(range(len(channels))):
            ch = channels[i]

            # first resblock at this level sees [current + skip] channels
            resblocks = nn.ModuleList()
            resblocks.append(ResBlock(in_ch + ch, ch, time_emb_dim=time_emb_dim, groups=groups))
            cur_ch = ch
            # remaining resblocks just process ch -> ch
            for _ in range(1, num_res_blocks):
                resblocks.append(ResBlock(cur_ch, ch, time_emb_dim=time_emb_dim, groups=groups))
                cur_ch = ch
            self.up_blocks.append(resblocks)
            in_ch = ch

            if cond_in_channels is not None:
                self.cond_fuse_up.append(CondFuseBlock(ch, cond_in_channels, groups=groups))

            # upsample, except after the final (highest-res) block
            if i != 0:
                out_ch = channels[i - 1]
                self.upsamples.append(Upsample(in_ch, out_ch))
                in_ch = out_ch

        # ---- final projection ----
        self.final_norm = nn.GroupNorm(groups, channels[0])
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

        # ---- initialize weights ----
        self.reset_parameters()

    # Initialization
    def reset_parameters(self):
        for m in self.modules():
            silu_kaiming_init_(m)

    # ------------------------------------------------------
    # forward
    # ------------------------------------------------------
    def forward(self, x, t, conds=None):
        """
        x     : (B, in_channels, H, W)      noisy sample
        t     : (B,) or (B, 1)              diffusion timestep
        conds : List[(B, cond_in_channels, H, W)] optional spatial condition at each resolution
        """
        if conds is not None and self.cond_in_channels is None:
            raise ValueError("Conditioning tensor provided, but UNet was created with cond_in_channels=None")

        if self.cond_in_channels is not None:
            if conds is None:
                # either allow unconditional behaviour, or explicitly require conds
                conds = [None] * len(self.down_blocks)
            else:
                assert len(conds) == len(self.down_blocks), \
                    f"expected {len(self.down_blocks)} cond tensors, got {len(conds)}"

        # timestep embedding
        t = t.float()
        if t.dim() == 0:         # scalar -> (1,)
            t = t[None]
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)        # (B, time_dim)

        # initial conv
        h = self.init_conv(x)
        skips = []

        # ----- downsampling path -----
        for i, resblocks in enumerate(self.down_blocks):
            for block in resblocks:
                h = block(h, t_emb)

            if conds is not None and conds[i] is not None:
                h = self.cond_fuse_down[i](h, conds[i])

            skips.append(h)

            if i < len(self.down_blocks) - 1:
                h = self.downsamples[i](h)

        # ----- bottleneck -----
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # ----- upsampling path -----
        for i, resblocks in enumerate(self.up_blocks):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)

            for block in resblocks:
                h = block(h, t_emb)

            if conds is not None and conds[-i] is not None:
                h = self.cond_fuse_up[i](h, conds[-i])

            if i < len(self.up_blocks) - 1:
                h = self.upsamples[i](h)

        # final projection back to noise prediction
        h = self.final_norm(h)
        h = self.final_act(h)
        h = self.final_conv(h)
        return h


if __name__ == "__main__":
    # Create model
    model = UNet2DModel(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        cond_in_channels=3
    )
    
    # Example inputs
    batch_size = 2
    height, width = 32, 32
    x = torch.randn(batch_size, 3, height, width)  # noisy image
    t = torch.randint(0, 1000, (batch_size,))       # timestep
    conds = [torch.randn(batch_size, 3, height // (2**i), width // (2**i)) if i == 2 else None
                for i in range(len(model.down_blocks))]  # conditions at each resolution
    print(conds)
    # Forward pass
    output = model(x, t, conds=conds)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")