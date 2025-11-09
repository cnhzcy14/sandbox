

add_library(add_kernel add_kernel.cu)
set_target_properties(add_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
)

add_executable(cuda_test cuda_test.cpp)
set_target_properties(cuda_test PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
)
target_link_libraries(cuda_test PRIVATE add_kernel)
if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    target_include_directories(cuda_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
