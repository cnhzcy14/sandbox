







# include_directories(
#     ${GSTREAMER_INCLUDE_DIRS}
#     ${GSTREAMER_VIDEO_INCLUDE_DIRS}
#     ${GSTREAMER_RTSPSERVER_INCLUDE_DIRS}
#     ${GLIB_INCLUDE_DIRS}
#     ${JSON-GLIB_INCLUDE_DIRS}
#     ${DEEPSTREAM_APP_COMMON_INCLUDE_DIRS}
#     ${NVDS_INCLUDE_DIRS}
#     ${CUDA_INCLUDE_DIRS}
#     ${X11_INCLUDE_DIR}
#     ${YAML_CPP_INCLUDE_DIRS}
#     src
# )


add_executable(deepstream-test1
    deepstream_test1_app.c
)
target_include_directories(deepstream-test1 PRIVATE
    ${NVDS_INCLUDE_DIRS}
)

target_link_libraries(deepstream-test1
    PkgConfig::GSTREAMER
    # ${GSTREAMER_VIDEO_LIBRARIES}
    PkgConfig::GST_RTSP
    ${YAML_CPP_LIBRARIES}
    PkgConfig::GLIB
    PkgConfig::GLIB_JSON
    ${NVDS_LIBRARIES}
    ${CUDA_LIBRARIES}
    # ${CMAKE_DL_LIBS}
    X11::X11
    m
)

