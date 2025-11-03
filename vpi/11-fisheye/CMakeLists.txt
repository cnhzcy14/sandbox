add_executable(fisheye main.cpp)
target_link_libraries(fisheye 
    vpi 
    opencv_core
    opencv_imgproc 
    opencv_calib3d 
    opencv_imgcodecs
    OpenMP::OpenMP_CXX )


add_executable(undistort undistort.cpp)
target_link_libraries(undistort 
	vpi 
	${OpenCV_LIBS}
    OpenMP::OpenMP_CXX )