
add_executable(cv_test cv_test.cpp)
target_link_libraries(cv_test ${OpenCV_LIBS})

# add_executable		( cv_para_test cv_para_test.cpp)
# target_link_libraries 	( cv_para_test ${LIBS} )

# add_executable		( cv_ocl_test cv_ocl_test.cpp)
# target_link_libraries 	( cv_ocl_test ${LIBS} )

# add_executable		( cv_nv12_test cv_nv12_test.cpp)
# target_link_libraries 	( cv_nv12_test ${LIBS} )

# add_executable		( optical_flow optical_flow.cpp)
# target_link_libraries 	( optical_flow ${LIBS} )

add_executable		( opencl-opencv-interop opencl-opencv-interop.cpp)
target_link_libraries 	( opencl-opencv-interop ${OpenCV_LIBS} OpenCL::OpenCL)

add_executable(cv_yuv cv_yuv.cpp)
target_link_libraries(cv_yuv ${OpenCV_LIBS})

install(TARGETS cv_test cv_yuv opencl-opencv-interop
  RUNTIME DESTINATION bin)