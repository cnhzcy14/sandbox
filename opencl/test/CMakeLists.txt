
add_executable( cl_test cl_test.c )
target_link_libraries( cl_test PRIVATE OpenCL::OpenCL )

install(TARGETS cl_test
  RUNTIME DESTINATION bin)