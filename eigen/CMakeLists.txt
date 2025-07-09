if(Eigen3_FOUND)
message(STATUS "  Eigen3:               ${Eigen3_FOUND}"                                )

add_executable(eigen_test eigen_test.cpp )
target_link_libraries(eigen_test Eigen3::Eigen)

install(TARGETS eigen_test
  RUNTIME DESTINATION bin)

endif()