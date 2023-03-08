add_executable			( cpp_test cpp_test.cpp )
add_executable			( event_test event_test.cpp )
target_link_libraries 	( event_test  Threads::Threads )

install(TARGETS cpp_test event_test
  RUNTIME DESTINATION bin)