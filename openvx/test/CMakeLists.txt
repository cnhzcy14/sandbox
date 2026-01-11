add_executable(canny canny.cpp)
target_include_directories(canny PRIVATE ${OPENVX_INCLUDE_DIR})
target_link_libraries(canny ${OPENVX_LIBRARY} ${OpenCV_LIBRARIES})

add_executable(optical_flow optical_flow.cpp)
target_include_directories(optical_flow PRIVATE ${OPENVX_INCLUDE_DIR})
target_link_libraries(optical_flow ${OPENVX_LIBRARY} ${OpenCV_LIBRARIES})

install(TARGETS canny optical_flow
    RUNTIME DESTINATION bin)
