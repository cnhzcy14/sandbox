if(OpenMP_FOUND)
message(STATUS "  OpenMP:               ${OpenMP_FOUND}"                                )
add_subdirectory 		(fast_software_implementation)
add_subdirectory 		(test)
endif()

