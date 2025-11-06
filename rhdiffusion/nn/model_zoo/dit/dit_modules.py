import torch.nn as nn

# -------------------------------------
# Patch embedding (ViT-style)
# -------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.num_patches_per_side = img_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, N, D)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        assert H == self.img_size and W == self.img_size, "Input image size must match img_size."

        # (B, C, H, W) → (B, C, h, p, w, p)
        x = x.view(B, C,
                   self.num_patches_per_side, p,
                   self.num_patches_per_side, p)
        # → (B, h, w, p, p, C)
        x = x.permute(0, 2, 4, 3, 5, 1)
        # → (B, N, patch_dim)
        x = x.reshape(B, self.num_patches, self.patch_dim)

        return self.proj(x)