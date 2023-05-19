
add_executable			( android_test android_test.c args.c bitmap.c gaussian.c )
target_link_libraries 	( android_test ${LIBS} m)
