add_executable ( matrix_mult_openmp_simd matrix_mult_openmp_simd.c )
target_link_libraries 	( matrix_mult_openmp_simd  OpenMP::OpenMP_C )

add_executable ( example_SIMD example_SIMD.c )
target_link_libraries 	( example_SIMD  OpenMP::OpenMP_C )

install(TARGETS matrix_mult_openmp_simd example_SIMD
        RUNTIME DESTINATION bin)