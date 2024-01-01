#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


void init_h(
    float **batchA,
    float **batchTau,
    float **batchH1,
    float **batchH2,
    float **batchH3,
    int batchSize,
    cudaStream_t stream);