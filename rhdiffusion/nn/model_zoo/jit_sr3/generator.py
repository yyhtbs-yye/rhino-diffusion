import torch
import torch.nn as nn
import torch.nn.functional as F

from rhdiffusion.nn.model_zoo.jit.modules import (
    TimestepEmbedder,
    BottleneckPatchEmbed,
    VisionRotaryEmbeddingFast,
    JiTBlock,
    FinalLayer,
)
from rhdiffusion.nn.model_zoo.jit_sr3.modules import PixelRefineHead

from rhdiffusion.nn.model_zoo.jit.utils import get_2d_sincos_pos_embed

class JiTSR3(nn.Module):
    """
    JiT-based SR3 super-resolution model with spatial pixel-unshuffle fusion.

    x : (N, C, H, W)              - noisy HR image (H = W = input_size)
    t : (N,)                      - diffusion timesteps
    y : (N, C, H//scale, W//scale) - LR conditioning image

    Fusion:
      1) PixelUnshuffle(y) so that its spatial grid matches x's patch grid.
      2) y_unshuf tokens -> (mu, sigma) per token (spatial AdaLN on x).
      3) y_unshuf tokens -> y_tok_embed, concat [x_mod, y_tok_embed], then
         project back to hidden_size.
    """

    def __init__(self, input_size=256, patch_size=16, scale=2,
                 in_channels=3, hidden_size=1024, depth=24, num_heads=16, 
                 mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0,
                 bottleneck_dim=128, 
                 in_context_len=0, in_context_start=8,):
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.scale = scale
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start

        # size of patch grid (H/patch, W/patch)  # NEW
        self.grid_size = input_size // patch_size  # NEW

        # ---- basic sanity for pixel-unshuffle alignment ----
        # We want: spatial(x_embed) = H/patch_size
        # y is H/scale, and after PixelUnshuffle(r) -> H/scale/r.
        # Require H/scale/r == H/patch_size => r = patch_size/scale (integer).
        assert patch_size >= scale and patch_size % scale == 0, (
            "For pixel-unshuffle fusion: patch_size must be >= scale and "
            "divisible by scale (so that y_unshuffle has same spatial size "
            "as the x patch grid)."
        )
        self.y_unshuffle_factor = patch_size // scale

        # ---- time embedding ----
        self.t_embedder = TimestepEmbedder(hidden_size)

        # ---- patch embed for x ----
        self.x_embedder = BottleneckPatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            pca_dim=bottleneck_dim,
            embed_dim=hidden_size,
            bias=True,
        )

        # ---- fixed 2D sin-cos positional embedding for image tokens ----
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # ---- in-context tokens (same as original JiT) ----
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, hidden_size),
                requires_grad=True,
            )
            nn.init.normal_(self.in_context_posemb, std=0.02)

        # ---- RoPE (Lightning-DiT style) ----
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size

        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0,
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len,
        )

        # ---- pixel-unshuffle-based y encoder ----
        self.y_unshuffle = nn.PixelUnshuffle(self.y_unshuffle_factor)
        # channels after unshuffle
        y_channels = in_channels * (self.y_unshuffle_factor ** 2)

        # spatial AdaLN: y_token -> (mu, sigma) per token
        self.spatial_norm = nn.LayerNorm(hidden_size)
        self.y_to_mu_sigma = nn.Linear(y_channels, 2 * hidden_size)

        # token-wise y embedding for concatenation and global pooling
        self.y_token_proj = nn.Linear(y_channels, hidden_size)

        # fuse [x_mod, y_embed] back to hidden_size
        self.xy_fuse = nn.Linear(2 * hidden_size, hidden_size)

        # ---- JiT backbone ----
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,)
            for i in range(depth)
        ])

        # ---- final prediction layer ----
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.pixel_refine = PixelRefineHead(in_ch=self.out_channels)

        # ---- init ----
        self.initialize_weights()

    # ---------------------------------------------------------------------
    # initialization (mostly identical to original JiT) 
    # ---------------------------------------------------------------------
    def initialize_weights(self):
        # Initialize transformer-like linears
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # fixed sin-cos pos_embed (frozen)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5),
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # patch_embed as linear
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # time MLP init
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # zero-out AdaLN in blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Add this to initialize_weights()
        nn.init.constant_(self.y_to_mu_sigma.weight, 0)
        nn.init.constant_(self.y_to_mu_sigma.bias, 0)

        # zero-out final AdaLN + output linear
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        nn.init.zeros_(self.y_token_proj.weight)
        nn.init.zeros_(self.y_token_proj.bias)
        nn.init.zeros_(self.xy_fuse.weight)
        nn.init.zeros_(self.xy_fuse.bias)

        # 8. CRITICAL FIX: Pixel Refine Head
        # Assuming PixelRefineHead has a last convolution layer, zero it out.
        # If PixelRefineHead is a complex module, access its last layer specifically.
        # Example generic fix (check your specific module structure):
        if hasattr(self.pixel_refine, 'weight'):
             nn.init.constant_(self.pixel_refine.weight, 0)
             if self.pixel_refine.bias is not None:
                 nn.init.constant_(self.pixel_refine.bias, 0)
        else:
            # If it's a sequence of layers, iterate to find the last Conv2d
            for m in self.pixel_refine.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    last_layer = m
            # Zero the very last layer found
            if 'last_layer' in locals():
                nn.init.constant_(last_layer.weight, 0)
                if last_layer.bias is not None:
                    nn.init.constant_(last_layer.bias, 0)
    # ---------------------------------------------------------------------
    # unpatchify (same as original) 
    # ---------------------------------------------------------------------
    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # ---------------------------------------------------------------------
    # forward: spatial pixel-unshuffle fusion + JiT backbone
    # ---------------------------------------------------------------------
    def forward(self, x, t, y):
        """
        x: (N, C, H, W)               -> patch_embed -> (N, T, D)
        t: (N,)
        y: (N, C, H//scale, W//scale) -> PixelUnshuffle -> (N, C', H/patch, W/patch)

        Fusion:
          1) y_unshuf tokens -> μ, σ -> spatial AdaLN on x tokens
          2) y_unshuf tokens -> y_token_embed
          3) concat [x_mod, y_token_embed] and project back to D
        """
        N, C, H, W = x.shape
        assert (
            H == self.input_size and W == self.input_size
        ), f"Expected x spatial size {self.input_size}, got {H}x{W}"

        # -----------------------------------------------------------------
        # 1. time embedding (global, as in DiT/SR3)
        # -----------------------------------------------------------------
        t_emb = self.t_embedder(t)  # (N, D)

        # -----------------------------------------------------------------
        # 2. patch-embed HR x -> tokens
        # -----------------------------------------------------------------
        x_tok = self.x_embedder(x)  # (N, T, D), T = (H/patch)^2

        # -----------------------------------------------------------------
        # 3. pixel-unshuffle y to match x's patch grid
        #    y: (N, C, H/scale, W/scale)
        #    -> y_unshuf: (N, C', H/patch, W/patch)
        # -----------------------------------------------------------------
        expected_h_lr = self.input_size // self.scale
        assert y.shape[2] == expected_h_lr and y.shape[3] == expected_h_lr, (
            f"Expected y spatial size {(expected_h_lr, expected_h_lr)}, "
            f"got {(y.shape[2], y.shape[3])}"
        )

        y_unshuf = self.y_unshuffle(y)  # (N, C', H/patch, W/patch)
        N_y, C_y, H_p, W_p = y_unshuf.shape
        assert N_y == N
        assert H_p == self.input_size // self.patch_size and W_p == H_p, (
            "After PixelUnshuffle, y must share the same grid as x tokens."
        )

        # tokens: (N, T, C_y)
        y_tok = y_unshuf.flatten(2).transpose(1, 2)

        # -----------------------------------------------------------------
        # 4. spatial AdaLN from y to x (early step before concat)
        # -----------------------------------------------------------------
        # normalize x token-wise
        x_norm = self.spatial_norm(x_tok)  # (N, T, D)

        # y_tok -> (mu, sigma) per token
        mu_sigma = self.y_to_mu_sigma(y_tok)  # (N, T, 2D)
        mu, sigma = mu_sigma.chunk(2, dim=-1)

        # spatial AdaLN (token-wise) and tame the modulation
        scale = 0.1
        x_mod = x_norm * (1.0 + scale * sigma) + scale * mu

        # -----------------------------------------------------------------
        # 5. concatenation fusion: [x_mod, y_embed] -> D
        # -----------------------------------------------------------------
        y_tok_embed = self.y_token_proj(y_tok)      # (N, T, D)
        xy_cat = torch.cat([x_mod, y_tok_embed], dim=-1)  # (N, T, 2D)
        x_fused = self.xy_fuse(xy_cat)             # (N, T, D)

        # add fixed pos_embed
        x_fused = x_fused + self.pos_embed

        # -----------------------------------------------------------------
        # 6. global conditioning for JiT blocks: c = t_emb + global(y)
        # -----------------------------------------------------------------
        y_global = y_tok_embed.mean(dim=1)  # (N, D)
        c = t_emb + y_global                # (N, D)

        # -----------------------------------------------------------------
        # 7. JiT blocks with in-context tokens constructed from y_global
        # -----------------------------------------------------------------
        x_seq = x_fused
        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                # in-context tokens: global y embedding + learned posemb
                in_context_tokens = y_global.unsqueeze(1).repeat(
                    1, self.in_context_len, 1
                )
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x_seq = torch.cat([in_context_tokens, x_seq], dim=1)

            rope = (
                self.feat_rope
                if i < self.in_context_start
                else self.feat_rope_incontext
            )
            x_seq = block(x_seq, c, feat_rope=rope)

        # drop in-context tokens before decoding patches
        if self.in_context_len > 0:
            x_seq = x_seq[:, self.in_context_len :, :]

        # -----------------------------------------------------------------
        # 8. final layer + unpatchify -> HR prediction
        # -----------------------------------------------------------------
        patches = self.final_layer(x_seq, c)  # (N, T, p^2 * C)
        
        output = self.unpatchify(patches, self.patch_size)  # (N, C, H, W)

        output = self.pixel_refine(output)
        return output
