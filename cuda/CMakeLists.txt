if(CUDA_FOUND)
message(STATUS "  CUDA:                 ${CUDA_FOUND}"                                  )
message(STATUS "  CUDA:                 ${ARCH_FLAGS}"                                  )

add_subdirectory 		(test)
# add_subdirectory 		(ratio_test)
# add_subdirectory 		(batchCUBLAS)
add_subdirectory 		(cuSolverDn_LinearSolver)
endif()

