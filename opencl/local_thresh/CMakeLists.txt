# add_executable			( local_thresh local_thresh.c ${OCL_SRC_LIST} )
# target_link_libraries 	( local_thresh ${LIBS} m)

find_package( PNG REQUIRED )
include_directories( ${PNG_INCLUDE_DIRS} )
set(LIBS ${LIBS} ${PNG_LIBRARIES} )

add_executable			( interp interp.c ${OCL_SRC_LIST} )
target_link_libraries 	( interp ${LIBS} m)

add_executable			( simple_image simple_image.c ${OCL_SRC_LIST} )
target_link_libraries 	( simple_image ${LIBS} m)

add_executable			( local_thresh local_thresh.cpp ${OCL_SRC_LIST} )
target_link_libraries 	( local_thresh ${LIBS} m)