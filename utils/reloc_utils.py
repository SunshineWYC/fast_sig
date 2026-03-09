import math
from typing import Dict, Tuple

import torch

N_MAX = 51
_BINOM_CACHE: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_binoms(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    cached = _BINOM_CACHE.get(key)
    if cached is not None:
        return cached

    binoms = torch.zeros((N_MAX, N_MAX), device=device, dtype=dtype)
    for n in range(N_MAX):
        for k in range(n + 1):
            binoms[n, k] = math.comb(n, k)

    _BINOM_CACHE[key] = binoms
    return binoms


def compute_relocation(opacity_old: torch.Tensor, scale_old: torch.Tensor, N: torch.Tensor):
    """
    Torch implementation of Eq.(9) from 3DGS-MCMC.

    Args:
        opacity_old: (P,) tensor in [0, 1]
        scale_old: (P, 3) positive scales
        N: (P,) integer relocation multiplicity
    """
    if opacity_old.ndim != 1:
        opacity_old = opacity_old.reshape(-1)
    if scale_old.ndim != 2 or scale_old.shape[1] != 3:
        raise ValueError("scale_old must have shape [P, 3]")
    if N.ndim != 1:
        N = N.reshape(-1)

    device = opacity_old.device
    dtype = opacity_old.dtype
    eps = torch.finfo(dtype).eps

    N = N.to(device=device, dtype=torch.long).clamp_(min=1, max=N_MAX - 1)
    opacity_old = opacity_old.clamp(min=eps, max=1.0 - eps)

    opacity_new = 1.0 - torch.pow(1.0 - opacity_old, 1.0 / N.to(dtype))
    denom_sum = torch.zeros_like(opacity_old)
    binoms = _get_binoms(device, dtype)

    for i in range(1, N_MAX):
        active = N >= i
        if not torch.any(active):
            break

        alpha = opacity_new[active]
        ks = torch.arange(i, device=device, dtype=torch.long)
        sign = torch.where((ks % 2) == 0, 1.0, -1.0).to(dtype)
        coeff = binoms[i - 1, :i] * sign / torch.sqrt(ks.to(dtype) + 1.0)
        powers = alpha.unsqueeze(-1).pow((ks + 1).to(dtype))
        denom_sum[active] += (powers * coeff).sum(dim=-1)

    coeff = opacity_old / (denom_sum + eps)
    scale_new = scale_old * coeff.unsqueeze(-1)
    return opacity_new, scale_new
