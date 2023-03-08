add_executable(canny canny.cpp)
target_link_libraries(canny ${OPENVX_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow ${OPENVX_LIBRARIES} ${OpenCV_LIBRARIES})

install(TARGETS canny optical_flow
    RUNTIME DESTINATION bin)
