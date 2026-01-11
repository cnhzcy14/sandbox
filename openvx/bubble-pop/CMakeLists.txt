list(APPEND SOURCES
    src/AMD_VX_Pop_Bubble.cpp
	src/AMD_VX_Pop_Donut.cpp
	src/internal_dataTranslator.cpp
	src/internal_publishKernels.cpp
	src/internal_vxNodes.cpp
)

add_library(vx_pop SHARED ${SOURCES})
target_include_directories(vx_pop PRIVATE ${OPENVX_INCLUDE_DIR})
target_link_libraries(vx_pop ${OPENVX_LIBRARY} ${OpenCV_LIBRARIES})
target_include_directories(vx_pop PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")

add_executable(vxPop apps/AMD_app.cpp)
target_include_directories(vxPop PRIVATE ${OPENVX_INCLUDE_DIR})
target_link_libraries(vxPop ${OPENVX_LIBRARY} ${OpenCV_LIBRARIES} vx_pop)


install(TARGETS vx_pop vxPop
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
)
