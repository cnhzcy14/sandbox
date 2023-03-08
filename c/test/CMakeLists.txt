add_executable			( c11_test c11_test.c )
target_link_libraries 	( c11_test  Threads::Threads )

install(TARGETS c11_test
  RUNTIME DESTINATION bin)
