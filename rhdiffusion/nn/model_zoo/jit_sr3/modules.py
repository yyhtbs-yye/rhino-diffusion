import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ConvFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        # 1. Standard AdaLN modulation (keep this from your original code)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
        # 2. Project to a feature map that can be upsampled
        # We perform a small convolution here to mix spatial context *before* upsampling
        self.pre_conv = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.SiLU()
        )

        # 3. Upsampling block (PixelShuffle is standard for SR)
        # We need to turn (H/P, W/P) -> (H, W). Factor = patch_size.
        # Note: 16x upsampling in one go is aggressive. Two 4x or four 2x is smoother,
        # but one 16x PixelShuffle is strictly better than a Linear layer.
        self.upsample = nn.PixelShuffle(patch_size)
        
        # The channel dim before PixelShuffle needs to be: out_channels * (patch_size**2)
        # But wait, PixelShuffle creates artifacts too if not careful (checkerboard).
        # A safer bet for diffusion is Conv -> Upsample -> Conv.
        
        # --- Better Approach: Mini-Decoder ---
        # Assuming patch_size=16, we do two 4x upsamples or four 2x upsamples.
        # Let's do 4x -> 4x for simplicity.
        
        mid_dim = hidden_size // 4
        
        self.decoder = nn.Sequential(
            # First Upsample x4
            nn.Conv2d(hidden_size, mid_dim * 16, kernel_size=3, padding=1),
            nn.PixelShuffle(4),      # reduces dim to mid_dim, scales spatial x4
            nn.SiLU(),
            
            # Smoothing Conv
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1),
            nn.SiLU(),

            # Second Upsample x4 (Total x16)
            nn.Conv2d(mid_dim, out_channels * 16, kernel_size=3, padding=1),
            nn.PixelShuffle(4),      # reduces dim to out_channels, scales spatial x4
        )

    def forward(self, x, c):
        # x: (N, T, D)
        # c: (N, D)
        
        # 1. AdaLN Modulation
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = x * (1 + shift.unsqueeze(1)) + scale.unsqueeze(1)
        
        # 2. Reshape to latent grid
        N, T, D = x.shape
        H_grid = W_grid = int(T**0.5)
        x = x.transpose(1, 2).reshape(N, D, H_grid, W_grid) # (N, D, H/16, W/16)

        # 3. Apply Convolutional Decoding
        # This allows pixels at the border of patch [0,0] to interact with patch [0,1]
        # BEFORE they become final pixels.
        out = self.decoder(x) # (N, out_channels, H, W)
        
        return out
    
class PixelRefineHead(nn.Module):
    def __init__(self, in_ch=3, hidden=64, depth=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(hidden, in_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)  # residual refine
