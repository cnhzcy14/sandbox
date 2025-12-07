add_executable          (disparity_filtering disparity_filtering.cpp)
target_link_libraries   (disparity_filtering ${OpenCV_LIBS})
 
add_executable          (stereo stereo.cpp)
target_link_libraries   (stereo ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

add_executable          (kitti_stereo_stereo kitti_stereo_stereo.cpp)
target_link_libraries   (kitti_stereo_stereo ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

