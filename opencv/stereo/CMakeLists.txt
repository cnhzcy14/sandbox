add_executable          (disparity_filtering disparity_filtering.cpp)
target_link_libraries   (disparity_filtering ${OpenCV_LIBS})

add_executable          (stereo stereo.cpp)
target_link_libraries   (stereo opencv::opencv OpenMP::OpenMP_CXX)
