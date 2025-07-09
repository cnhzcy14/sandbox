if(apriltag_FOUND)
message(STATUS "  apriltag:             ${apriltag_FOUND}"                              )

add_executable(apriltag_demo apriltag_demo.c)
target_link_libraries(apriltag_demo apriltag::apriltag)


add_executable(opencv_demo opencv_demo.cc)
target_link_libraries(opencv_demo 
  ${OpenCV_LIBS} 
  apriltag::apriltag
  )

install(TARGETS opencv_demo apriltag_demo
  RUNTIME DESTINATION bin)

endif()