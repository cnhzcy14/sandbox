#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp parallel
    {
        printf("outer: omp_get_place_num = %d omp_get_level = %d\n", omp_get_place_num(), omp_get_level());
#pragma omp parallel
        {
            printf("inner: omp_get_place_num = %d omp_get_level = %d\n", omp_get_place_num(), omp_get_level());
        }
    }
}