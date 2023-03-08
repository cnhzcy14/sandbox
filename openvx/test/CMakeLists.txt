add_executable(canny canny.cpp)
target_link_libraries(canny ${OPENVX_LIBRARIES} ${OpenCV_LIBRARIES})
install(TARGETS canny 
    RUNTIME DESTINATION bin)
