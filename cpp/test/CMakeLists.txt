add_executable			( cpp_test cpp_test.cpp )
add_executable			( event_test event_test.cpp )
target_link_libraries 	( event_test  Threads::Threads )
add_executable			( modern_time modern_time.cpp )
add_executable			( modern_file modern_file.cpp )

install(
  TARGETS 
  cpp_test 
  event_test
  modern_time
  modern_file
  RUNTIME DESTINATION bin)