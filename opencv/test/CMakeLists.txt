
add_executable          (cv_test cv_test.cpp)
target_link_libraries   (cv_test ${OpenCV_LIBS})

# add_executable		( cv_para_test cv_para_test.cpp)
# target_link_libraries 	( cv_para_test ${LIBS} )

# add_executable		( cv_ocl_test cv_ocl_test.cpp)
# target_link_libraries 	( cv_ocl_test ${LIBS} )

# add_executable		( cv_nv12_test cv_nv12_test.cpp)
# target_link_libraries 	( cv_nv12_test ${LIBS} )

# add_executable		( optical_flow optical_flow.cpp)
# target_link_libraries 	( optical_flow ${LIBS} )

add_executable		    ( opencl-opencv-interop opencl-opencv-interop.cpp)
target_link_libraries   ( opencl-opencv-interop ${OpenCV_LIBS} OpenCL::OpenCL)

add_executable		    ( opencl-gaussian opencl-gaussian.cpp)
target_link_libraries 	( opencl-gaussian ${OpenCV_LIBS} OpenCL::OpenCL)

add_executable		    ( opencl-gaussian-buf opencl-gaussian-buf.cpp)
target_link_libraries 	( opencl-gaussian-buf ${OpenCV_LIBS} OpenCL::OpenCL)

add_executable		    ( opencl-threshold opencl-threshold.cpp)
target_link_libraries 	( opencl-threshold ${OpenCV_LIBS} OpenCL::OpenCL)

add_executable		    ( opencl-histogram opencl-histogram.cpp)
target_link_libraries 	( opencl-histogram ${OpenCV_LIBS} OpenCL::OpenCL)

add_executable		    ( opencl-maxloc opencl-maxloc.cpp)
target_link_libraries 	( opencl-maxloc ${OpenCV_LIBS} OpenCL::OpenCL)

add_executable		    ( opencl-maxloc-vec opencl-maxloc-vec.cpp)
target_link_libraries 	( opencl-maxloc-vec ${OpenCV_LIBS} OpenCL::OpenCL)

add_executable          (cv_yuv cv_yuv.cpp)
target_link_libraries   (cv_yuv ${OpenCV_LIBS})

install(
    TARGETS 
        cv_test 
        cv_yuv 
        opencl-opencv-interop 
        opencl-gaussian
        opencl-gaussian-buf 
        opencl-threshold
        opencl-histogram
        opencl-maxloc
        opencl-maxloc-vec
    RUNTIME DESTINATION bin)