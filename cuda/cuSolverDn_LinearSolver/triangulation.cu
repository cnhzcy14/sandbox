#include <array>
#include <cstdint>
#include <vector>
#include <cassert>
#include "triangulation_cu.h"


__device__ void undistortPoint(const float* src, float* dst, const float* cameraMatrix, const float* distCoeffs)
{
    dst[0] = (src[0] - cameraMatrix[0*3 + 2])/cameraMatrix[0*3 + 0];
    dst[1] = (src[1] - cameraMatrix[1*3 + 2])/cameraMatrix[1*3 + 1];
}


__global__
void init_h_kernel(
    float **batchA,
    float **batchTau,
    float **batchH1,
    float **batchH2,
    float **batchH3,
    int batchSize
)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i = index; i < batchSize; i += stride)
	{	
		float *A = batchA[i];
		float *Tau = batchTau[i];
		float *H1 = batchH1[i];
		float *H2 = batchH2[i];
		float *H3 = batchH3[i];

		// I-tau*v*v'
		H1[0] = 1-Tau[0]; 		H1[4] =  -Tau[0]*A[1]; 		H1[8]  =  -Tau[0]*A[2];			H1[12]  =  -Tau[0]*A[3];
		H1[1] =  -Tau[0]*A[1]; 	H1[5] = 1-Tau[0]*A[1]*A[1];	H1[9]  =  -Tau[0]*A[1]*A[2];	H1[13]  =  -Tau[0]*A[1]*A[3];
		H1[2] =  -Tau[0]*A[2]; 	H1[6] =  -Tau[0]*A[2]*A[1];	H1[10] = 1-Tau[0]*A[2]*A[2];	H1[14]  =  -Tau[0]*A[2]*A[3];
		H1[3] =  -Tau[0]*A[3]; 	H1[7] =  -Tau[0]*A[3]*A[1];	H1[11] =  -Tau[0]*A[3]*A[2];	H1[15]  = 1-Tau[0]*A[3]*A[3];

		H2[0] = 1;	H2[4] = 0; 				H2[8]  =  0;					H2[12] =  0;
		H2[1] = 0; 	H2[5] = 1-Tau[1];		H2[9]  =  -Tau[1]*A[6];			H2[13] =  -Tau[1]*A[7];
		H2[2] = 0; 	H2[6] =  -Tau[1]*A[6];	H2[10] = 1-Tau[1]*A[6]*A[6];	H2[14] =  -Tau[1]*A[6]*A[7];
		H2[3] = 0; 	H2[7] =  -Tau[1]*A[7];	H2[11] =  -Tau[1]*A[7]*A[6];	H2[15] = 1-Tau[1]*A[7]*A[7];

		H3[0] = 1; 	H3[4] = 0; 	H3[8]  = 0;					H3[12] = 0;
		H3[1] = 0; 	H3[5] = 1;	H3[9]  = 0;					H3[13] = 0;
		H3[2] = 0; 	H3[6] = 0;	H3[10] = 1-Tau[2];			H3[14] =  -Tau[2]*A[11];
		H3[3] = 0; 	H3[7] = 0;	H3[11] =  -Tau[2]*A[11];	H3[15] = 1-Tau[2]*A[11]*A[11];
	}
}



void init_h(
    float **batchA,
    float **batchTau,
    float **batchH1,
    float **batchH2,
    float **batchH3,
    int batchSize,
    cudaStream_t stream)
{
	int blockSize = 128;
	int numBlocks = (batchSize + blockSize - 1) / blockSize;
	init_h_kernel<<<numBlocks, blockSize, 0, stream>>>(
		batchA,
		batchTau,
		batchH1,
		batchH2,
		batchH3,
		batchSize
		);
}