add_library(add_kernel add_kernel.cu)
set_target_properties(add_kernel PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)

add_executable			( cuda_test cuda_test.cpp )
target_link_libraries 	( cuda_test  add_kernel)
