#include "add_kernel.h"
// Kernel function to add the elements of two arrays
__global__
void add_kernel(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}

void add(int n, float *x, float *y, cudaStream_t stream)
{
	int blockSize = 128;
	int numBlocks = (n + blockSize - 1) / blockSize;
	add_kernel<<<numBlocks, blockSize, 0, stream>>>(n, x, y);
}