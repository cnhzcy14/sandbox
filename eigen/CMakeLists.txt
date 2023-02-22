add_executable(eigen_test eigen_test.cpp )
target_link_libraries(eigen_test Eigen3::Eigen)

install(TARGETS eigen_test
  RUNTIME DESTINATION bin)