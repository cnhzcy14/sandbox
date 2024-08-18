

add_library				( triangulation_cu triangulation.cu)
set_target_properties	( triangulation_cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties	( triangulation_cu PROPERTIES CUDA_ARCHITECTURES "86" )

add_executable			( batchQR_example batchQR_example.cpp )
target_link_libraries 	( batchQR_example cuda cublas cusolver triangulation_cu)
set_target_properties	( batchQR_example PROPERTIES CUDA_ARCHITECTURES "86" )

# add_executable			( cuSolverDn_LinearSolver cuSolverDn_LinearSolver.cpp mmio_wrapper.cpp mmio.c )
# target_link_libraries 	( cuSolverDn_LinearSolver ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES}  ${CUDA_cusparse_LIBRARY} cuda cusolver)

# add_executable			( gesvdjbatch_example gesvdjbatch_example.cpp )
# target_link_libraries 	( gesvdjbatch_example ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES}  ${CUDA_cusparse_LIBRARY} cuda cusolver)

# add_executable			( ormqr_example ormqr_example.cpp )
# target_link_libraries 	( ormqr_example ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES}  ${CUDA_cusparse_LIBRARY} cuda cusolver)