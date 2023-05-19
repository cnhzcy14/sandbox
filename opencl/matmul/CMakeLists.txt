add_executable		( matmul matmul.c matrix_lib.c common/wtime.c common/device_info.c )
target_link_libraries 	( matmul PRIVATE OpenCL::OpenCL )

install(TARGETS matmul
        RUNTIME DESTINATION bin)

        
