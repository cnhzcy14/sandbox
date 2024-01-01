/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include ormqr_example.cpp 
 *   nvcc -o -fopenmp a.out ormqr_example.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "triangulation_cu.h"

void printMatrix(int m, int n, const float *A, int lda, const char *name)
{
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col++)
        {
            float Areg = A[row + col * lda];
            printf("%f  ", Areg);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    cudaStream_t stream = nullptr;
    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    const int m = 4;
    const int n = 3;
    const int lda = m;
    const int ldb = m;
    const int nrhs = 1; // number of right hand side vectors
                        /*       | 1 2 3 |
 *   A = | 4 5 6 |
 *       | 2 1 1 |
 *       | 3 2 1 |
 *
 *   x = (1 1 1)'
 *   b = (6 15 4 )'
 */

    cudaStat1 = cudaStreamCreate(&stream);
    assert(cudaSuccess == cudaStat1);

    float *Work1 = nullptr;
    float *Work2 = nullptr;

    float **batchA = nullptr;
    float **batchTau = nullptr;
    float **batchB = nullptr;
    float **batchH1 = nullptr;
    float **batchH2 = nullptr;
    float **batchH3 = nullptr;
    float **batchH1H2 = nullptr;
    float **batchQ = nullptr;
    float **batchX = nullptr;
    int *devInfo = nullptr; // info in gpu (device copy)
    const float one = 1.0;
    const float zero = 0.0;

    // step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    cusolver_status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    cublas_status = cublasSetStream(cublasH, stream);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // step 2: copy A and B to device
    cudaStat4 = cudaMallocManaged(&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat4);
    *devInfo = 0;

    cudaStat1 = cudaMallocManaged(&Work1, sizeof(float) * lda * m * 5);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMallocManaged(&Work2, sizeof(float) * lda * m * 5);
    assert(cudaSuccess == cudaStat1);


    cudaMallocManaged(&batchA, sizeof(*batchA) * 2);
    cudaMallocManaged(&batchB, sizeof(*batchB) * 2);
    batchA[0] = &Work1[0];
    batchA[1] = &Work2[0];
    batchB[0] = &Work1[12];
    batchB[1] = &Work2[12];


    batchA[0][0] = 1.0;
    batchA[0][1] = 4.0;
    batchA[0][2] = 2.0;
    batchA[0][3] = 3.0;
    batchA[0][4] = 2.0;
    batchA[0][5] = 5.0;
    batchA[0][6] = 1.0;
    batchA[0][7] = 2.0;
    batchA[0][8] = 3.0;
    batchA[0][9] = 6.0;
    batchA[0][10] = 1.0;
    batchA[0][11] = 1.0;
    batchB[0][0] = 9.0;
    batchB[0][1] = 21.0;
    batchB[0][2] = 5.0;
    batchB[0][3] = 7.0;
    batchA[1][0] = 1.0;
    batchA[1][1] = 4.0;
    batchA[1][2] = 2.0;
    batchA[1][3] = 3.0;
    batchA[1][4] = 2.0;
    batchA[1][5] = 5.0;
    batchA[1][6] = 1.0;
    batchA[1][7] = 2.0;
    batchA[1][8] = 3.0;
    batchA[1][9] = 6.0;
    batchA[1][10] = 3.0;
    batchA[1][11] = 1.0;
    batchB[1][0] = 0.0;
    batchB[1][1] = 3.0;
    batchB[1][2] = 0.0;
    batchB[1][3] = 4.0;

    cudaMallocManaged(&batchTau, sizeof(*batchTau) * 2);
    cudaMallocManaged(&batchH1, sizeof(*batchH1) * 2);
    cudaMallocManaged(&batchH2, sizeof(*batchH2) * 2);
    cudaMallocManaged(&batchH3, sizeof(*batchH3) * 2);
    cudaMallocManaged(&batchH1H2, sizeof(*batchH1H2) * 2);
    cudaMallocManaged(&batchQ, sizeof(*batchQ) * 2);
    cudaMallocManaged(&batchX, sizeof(*batchX) * 2);
    batchTau[0] = &Work1[64];
    batchTau[1] = &Work2[64];
    batchH1[0] = &Work1[16];
    batchH1[1] = &Work2[16];
    batchH2[0] = &Work1[32];
    batchH2[1] = &Work2[32];
    batchH3[0] = &Work1[48];
    batchH3[1] = &Work2[48];
    batchH1H2[0] = &Work1[64];
    batchH1H2[1] = &Work2[64];
    batchQ[0] = &Work1[16];
    batchQ[1] = &Work2[16];
    batchX[0] = &Work1[32];
    batchX[1] = &Work2[32];

    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);

    // step 3: compute QR of A
    cublas_status = cublasSgeqrfBatched(
        cublasH,
        m,
        n,
        batchA,
        lda,
        batchTau,
        devInfo,
        2);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    cudaStat1 = cudaStreamSynchronize(stream);
    assert(cudaSuccess == cudaStat1);
    printf("after geqrf: info_gpu = %d\n", *devInfo);
    assert(0 == *devInfo);


    // step 4: compute Q^T*B
    init_h(
        batchA,
        batchTau,
        batchH1,
        batchH2,
        batchH3,
        2,
        stream);

    cublas_status = cublasSgemmBatched(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &one, batchH1, lda, batchH2, lda, 
        &zero, batchH1H2, lda, 2);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cudaStat1 = cudaStreamSynchronize(stream);;

    cublas_status = cublasSgemmBatched(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &one, batchH1H2, lda, batchH3, lda, 
        &zero, batchQ, lda, 2);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cublas_status = cublasSgemmBatched(
        cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, nrhs, m, &one, batchQ, lda, batchB, ldb, 
        &zero, batchX, lda, 2);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cudaStat1 = cudaStreamSynchronize(stream);;
    assert(cudaSuccess == cudaStat1);
    printf("BB1 = \n");
    printMatrix(3, 1, batchX[0], 3, "BB1");
    printf("BB2 = \n");
    printMatrix(3, 1, batchX[1], 3, "BB2");
    printf("===============\n");


    // step 5: compute x = R \ Q^T*B
    cublas_status = cublasStrsmBatched(
        cublasH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        n,
        nrhs,
        &one,
        batchA,
        lda,
        batchX,
        ldb,
        2);
    cudaStat1 = cudaStreamSynchronize(stream);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    assert(cudaSuccess == cudaStat1);

    printf("X1 = \n");
    printMatrix(n, nrhs, batchX[0], ldb, "X1");
    printf("X2 = \n");
    printMatrix(n, nrhs, batchX[1], ldb, "X2");
    printf("====================\n");


    if (Work1)
        cudaFree(Work1);
    if (Work2)
        cudaFree(Work2);
    if (devInfo)
        cudaFree(devInfo);

    if (batchA)
        cudaFree(batchA);
    if (batchTau)
        cudaFree(batchTau);
    if (batchB)
        cudaFree(batchB);
    if (batchH1)
        cudaFree(batchH1);
    if (batchH2)
        cudaFree(batchH2);
    if (batchH3)
        cudaFree(batchH3);
    if (batchH1H2)
        cudaFree(batchH1H2);
    if (batchQ)
        cudaFree(batchQ);

    if (cublasH)
        cublasDestroy(cublasH);
    if (cusolverH)
        cusolverDnDestroy(cusolverH);

    cudaDeviceReset();

    return 0;
}
