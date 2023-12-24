#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void add(int n, float *x, float *y, cudaStream_t stream);
