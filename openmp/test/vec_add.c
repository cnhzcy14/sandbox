#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
// #include <chrono>
// #include <iostream>

#define N (100000)
int main(int argc, char *argv[])
{
    int nthreads, tid, idx;
    float a[N], b[N], c[N];
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);

    // auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (idx = 0; idx < N; ++idx)
    {
        a[idx] = b[idx] = 1.0;
    }
#pragma omp parallel for
    for (idx = 0; idx < N; ++idx)
    {
        c[idx] = a[idx] + b[idx];
        tid = omp_get_thread_num();
        printf("Thread %d: c[%d]=%f\n", tid, idx, c[idx]);
    }

    // auto t1 = std::chrono::high_resolution_clock::now();
    // auto dt = 1.e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    // std::cout << "Display Time : " << 1000.0f / dt << "fps " << dt << "ms " << std::endl;
}