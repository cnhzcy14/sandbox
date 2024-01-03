
add_executable(deepstream-test1
    deepstream_test1_app.c
)
target_include_directories(deepstream-test1 PRIVATE
    ${NVDS_INCLUDE_DIRS}
)

target_link_libraries(deepstream-test1
    PkgConfig::GSTREAMER
    PkgConfig::GST_RTSP
    PkgConfig::GLIB
    PkgConfig::GLIB_JSON
    ${YAML_CPP_LIBRARIES}
    ${NVDS_LIBRARIES}
    ${CUDA_LIBRARIES}
    X11::X11
    m
)

add_executable(deepstream-test1-cam
    deepstream_test1_app_cam.c
)
target_include_directories(deepstream-test1-cam PRIVATE
    ${NVDS_INCLUDE_DIRS}
)

target_link_libraries(deepstream-test1-cam
    PkgConfig::GSTREAMER
    PkgConfig::GST_RTSP
    PkgConfig::GLIB
    PkgConfig::GLIB_JSON
    ${YAML_CPP_LIBRARIES}
    ${NVDS_LIBRARIES}
    ${CUDA_LIBRARIES}
    X11::X11
    m
)