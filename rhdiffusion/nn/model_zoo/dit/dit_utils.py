import math
import torch
import torch.nn as nn

def timestep_embedding(timesteps, dim, max_period=10_000):
    """
    Sinusoidal timestep embeddings, like in DDPM/DiT.
    timesteps: (B,) int/float tensor
    returns:   (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float()[:, None] * freqs[None]  # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 2*half)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb
