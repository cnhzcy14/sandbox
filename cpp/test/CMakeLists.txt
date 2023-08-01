add_executable			( cpp_test cpp_test.cpp )
add_executable			( modern_event modern_event.cpp )
target_link_libraries 	( modern_event  Threads::Threads )
add_executable			( modern_time modern_time.cpp )
add_executable			( modern_folder modern_folder.cpp )
add_executable			( modern_file modern_file.cpp )

install(
  TARGETS 
  cpp_test 
  modern_event
  modern_time
  modern_file
  RUNTIME DESTINATION bin)