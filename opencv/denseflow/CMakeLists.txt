add_executable		( farneback_flow farneback_flow.cpp flow_functions.cpp)
target_link_libraries 	( farneback_flow ${OpenCV_LIBS} )

add_executable		( blockmatch_flow blockmatch_flow.cpp flow_functions.cpp)
target_link_libraries 	( blockmatch_flow ${OpenCV_LIBS} )

add_executable          ( brox_flow brox_flow.cpp flow_functions.cpp)
target_link_libraries 	( brox_flow ${OpenCV_LIBS} )

# add_executable          ( blockmatch_px4_flow blockmatch_px4_flow.cpp flow_functions.cpp px4flow.cpp)
# target_link_libraries 	( blockmatch_px4_flow ${OpenCV_LIBS} )
# install(TARGETS blockmatch_px4_flow
#         RUNTIME DESTINATION /usr/local/bin)