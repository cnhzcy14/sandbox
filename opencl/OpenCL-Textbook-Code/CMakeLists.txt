add_executable		( histogram histogram.c Utils/bmp-utils.c Utils/gold.c Utils/utils.c )
target_link_libraries 	( histogram PRIVATE OpenCL::OpenCL )

install(TARGETS histogram
        RUNTIME DESTINATION bin)

        
