add_executable			( marching_squares marching_squares.c ${OCL_SRC_LIST} )
target_link_libraries 	( marching_squares ${LIBS} m)

add_executable			( hp5 hp5.cpp )
target_link_libraries 	( hp5 ${LIBS} m)