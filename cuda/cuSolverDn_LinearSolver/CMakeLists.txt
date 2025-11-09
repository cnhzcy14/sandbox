

add_library				( triangulation_cu triangulation.cu)
set_target_properties	( triangulation_cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties	( triangulation_cu PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}" )

add_executable			( batchQR_example batchQR_example.cpp )
target_link_libraries 	( batchQR_example cuda cublas cusolver triangulation_cu)
set_target_properties	( batchQR_example PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}" )
if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    target_include_directories(batchQR_example PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
