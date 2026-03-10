import math

import torch
from diff_gaussian_rasterization import compute_relocation as _compute_relocation

N_max = 51
_BINOMS = {}


def _get_binoms(device: torch.device, dtype: torch.dtype = torch.float32):
    cached = _BINOMS.get(device)
    if cached is not None:
        return cached.to(dtype=dtype)

    binoms = torch.zeros((N_max, N_max), device=device, dtype=torch.float32)
    for n in range(N_max):
        for k in range(n + 1):
            binoms[n, k] = math.comb(n, k)
    _BINOMS[device] = binoms
    return binoms.to(dtype=dtype)


def compute_relocation_cuda(opacity_old, scale_old, N):
    N = N.to(device=opacity_old.device, dtype=torch.int32).contiguous()
    N.clamp_(min=1, max=N_max - 1)
    binoms = _get_binoms(opacity_old.device).contiguous()
    return _compute_relocation(opacity_old, scale_old, N, binoms, N_max)


def compute_relocation(opacity_old, scale_old, N):
    """
    Original Python/Torch implementation of Eq.(9) from 3DGS-MCMC.
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

    N = N.to(device=device, dtype=torch.long).clamp_(min=1, max=N_max - 1)
    opacity_old = opacity_old.clamp(min=eps, max=1.0 - eps)

    opacity_new = 1.0 - torch.pow(1.0 - opacity_old, 1.0 / N.to(dtype))
    denom_sum = torch.zeros_like(opacity_old)
    binoms = _get_binoms(device, dtype)

    for i in range(1, N_max):
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
