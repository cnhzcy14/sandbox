if(CMAKE_CUDA_COMPILER)
message(STATUS "  CUDA:                 ${CUDAToolkit_FOUND}")
    
add_subdirectory 		(test)
# add_subdirectory 		(ratio_test)
# add_subdirectory 		(batchCUBLAS)
add_subdirectory 		(cuSolverDn_LinearSolver)
endif()

