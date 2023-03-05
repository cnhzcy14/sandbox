#include <stdio.h>
#include <omp.h>

void threadSafeFunction()
{
    #pragma omp critical
    {
        // Code that needs to be thread safe goes here
        printf("Thread %d is executing the critical section\n", omp_get_thread_num());
    }
}

int main()
{
    // Enable OpenMP parallelism
    #pragma omp parallel
    {
        // Call the thread safe function from each thread
        threadSafeFunction();
    }
    return 0;
}
