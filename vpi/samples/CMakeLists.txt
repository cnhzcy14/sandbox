

# Creates the blur executable target
add_executable(vpi_blur vpi_blur.cpp)

# It uses vpi and opencv. CMake will automatically
# set up the correct header and library directories,
# and make hello_work link to these libraries.
target_link_libraries(vpi_blur 
	vpi 
	opencv_core 
	opencv_imgproc 
	opencv_imgcodecs)


add_executable(optflow_dense optflow_dense.cpp)
target_link_libraries(optflow_dense 
	vpi 
	opencv_core 
	opencv_imgproc 
	opencv_imgcodecs 
	opencv_videoio)

install(TARGETS vpi_blur optflow_dense
  RUNTIME DESTINATION bin)