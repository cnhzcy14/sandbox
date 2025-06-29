if(OpenCV_FOUND)

add_executable          (disparity_filtering disparity_filtering.cpp)
target_link_libraries   (disparity_filtering ${OpenCV_LIBS})

endif()