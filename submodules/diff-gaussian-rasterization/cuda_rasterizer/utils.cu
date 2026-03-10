#include "utils.h"

#include <math.h>

// Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
__global__ void compute_relocation_kernel(
    int P,
    const float* opacity_old,
    const float* scale_old,
    const int* N,
    const float* binoms,
    int n_max,
    float* opacity_new,
    float* scale_new)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= P)
        return;

    const int N_idx = N[idx];
    const float opacity_src = opacity_old[idx];
    const float opacity_dst = 1.0f - powf(1.0f - opacity_src, 1.0f / (float)N_idx);
    opacity_new[idx] = opacity_dst;

    float denom_sum = 0.0f;
    for (int i = 1; i <= N_idx; ++i)
    {
        for (int k = 0; k <= (i - 1); ++k)
        {
            const float sign = ((k & 1) == 0) ? 1.0f : -1.0f;
            const float bin_coeff = binoms[(i - 1) * n_max + k];
            const float term = sign * powf(opacity_dst, (float)(k + 1)) / sqrtf((float)(k + 1));
            denom_sum += (bin_coeff * term);
        }
    }

    const float coeff = opacity_src / (denom_sum + 1e-12f);
    const int base = idx * 3;
    scale_new[base + 0] = coeff * scale_old[base + 0];
    scale_new[base + 1] = coeff * scale_old[base + 1];
    scale_new[base + 2] = coeff * scale_old[base + 2];
}

void UTILS::ComputeRelocation(
    int P,
    const float* opacity_old,
    const float* scale_old,
    const int* N,
    const float* binoms,
    int n_max,
    float* opacity_new,
    float* scale_new)
{
    const int num_blocks = (P + 255) / 256;
    dim3 block(256, 1, 1);
    dim3 grid(num_blocks, 1, 1);
    compute_relocation_kernel<<<grid, block>>>(
        P,
        opacity_old,
        scale_old,
        N,
        binoms,
        n_max,
        opacity_new,
        scale_new);
}
