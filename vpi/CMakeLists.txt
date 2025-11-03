if(vpi_FOUND)
message(STATUS "  vpi:                  ${vpi_FOUND}"                                   )
add_subdirectory 		(samples)
add_subdirectory 		(11-fisheye)
endif()

