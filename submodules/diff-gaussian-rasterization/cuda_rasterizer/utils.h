#ifndef CUDA_RASTERIZER_UTILS_H_INCLUDED
#define CUDA_RASTERIZER_UTILS_H_INCLUDED

#include <cuda.h>
#include <cuda_runtime.h>

namespace UTILS
{
    void ComputeRelocation(
        int P,
        const float* opacity_old,
        const float* scale_old,
        const int* N,
        const float* binoms,
        int n_max,
        float* opacity_new,
        float* scale_new);
}

#endif
