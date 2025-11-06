import math
import torch
import torch.nn as nn
from rhdiffusion.nn.model_zoo.dit.dit_utils import timestep_embedding
from rhdiffusion.nn.model_zoo.dit.dit_modules import PatchEmbed

# -------------------------------------
# DiT block: MHSA + MLP with AdaLayerNorm conditioning
# -------------------------------------

class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, cond_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        # Two AdaLN layers: each produces (scale, shift) for LayerNorm
        self.adaLN1 = nn.Linear(cond_dim, 2 * dim)
        self.adaLN2 = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x, cond):
        """
        x:    (B, N, D)
        cond: (B, cond_dim)
        """
        # ---- Self-attention branch ----
        gamma1, beta1 = self.adaLN1(cond).chunk(2, dim=-1)
        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # ---- MLP branch ----
        gamma2, beta2 = self.adaLN2(cond).chunk(2, dim=-1)
        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.mlp(h)
        x = x + h

        return x

# -------------------------------------
# Full DiT backbone
# -------------------------------------

class DiT(nn.Module):
    """
    Minimal Diffusion Transformer, roughly following Peebles & Xie (2023):

    - ViT-style patchify / unpatchify
    - Learned pos-emb
    - Timestep + (optional) class conditioning via AdaLayerNorm
    - Stack of DiTBlocks
    """

    def __init__(
        self,
        img_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        num_classes=None,   # set an int for class-conditional DiT
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        self.num_patches = self.patch_embed.num_patches
        self.patch_dim = self.patch_embed.patch_dim

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # Timestep MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # Optional class embedding
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, hidden_size)
        else:
            self.label_emb = None

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    cond_dim=hidden_size,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

        # Output projection (tokens → patches) + final norm
        self.norm_out = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, self.patch_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.02)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)
        if self.label_emb is not None:
            nn.init.normal_(self.label_emb.weight, std=0.02)

    # --- unpatchify is the inverse of PatchEmbed.patchify ---
    def unpatchify(self, x):
        """
        x: (B, N, patch_dim)
        returns: (B, C, H, W)
        """
        B, N, P = x.shape
        p = self.patch_size
        h = w = self.patch_embed.num_patches_per_side
        assert N == h * w

        x = x.view(B, h, w, p, p, self.in_channels)     # (B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)                 # (B, C, h, p, w, p)
        x = x.reshape(B, self.in_channels, h * p, w * p)
        return x

    def forward(self, x, t, y=None):
        """
        x: (B, C, H, W)   noisy image / latent
        t: (B,)           timesteps
        y: (B,)           class labels (if num_classes is not None)

        returns: (B, C, H, W) predicted noise ε_θ(x_t, t, y)
        """
        # Patchify to tokens
        x = self.patch_embed(x)                         # (B, N, D)
        x = x + self.pos_embed                          # add pos-emb

        # Build conditioning embedding
        t_emb = timestep_embedding(t, x.size(-1))       # (B, D)
        t_emb = self.time_mlp(t_emb)                    # (B, D)

        if self.label_emb is not None:
            assert y is not None, "Class labels must be provided when num_classes is set."
            y_emb = self.label_emb(y)                   # (B, D)
            cond = t_emb + y_emb
        else:
            cond = t_emb

        # Transformer stack
        for blk in self.blocks:
            x = blk(x, cond)

        # Tokens → patches → image
        x = self.norm_out(x)
        x = self.proj_out(x)                            # (B, N, patch_dim)
        x = self.unpatchify(x)                          # (B, C, H, W)
        return {'sample': x}


# -------------------------------------
# Tiny sanity check
# -------------------------------------
if __name__ == "__main__":
    model = DiT(
        img_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=256,
        depth=4,
        num_heads=4,
        num_classes=10,
    )
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 10, (2,))
    out = model(x, t, y)
    print("Input:", x.shape, "Output:", out.shape)
