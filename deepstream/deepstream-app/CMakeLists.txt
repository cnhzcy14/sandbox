add_executable(deepstream-app
    deepstream_app_config_parser.c
    deepstream_app_main.c
    deepstream_app.c
    deepstream_app_config_parser_yaml.cpp
    ${DEEPSTREAM_APP_COMMON_SRCS}
)
target_include_directories(deepstream-app PRIVATE
    ${NVDS_INCLUDE_DIRS}
    ${DEEPSTREAM_APP_COMMON_INCLUDE_DIRS}
)

target_link_libraries(deepstream-app
    PkgConfig::GSTREAMER
    PkgConfig::GST_RTSP
    PkgConfig::GLIB
    PkgConfig::GLIB_JSON
    PkgConfig::GST_VIDEO
    ${YAML_CPP_LIBRARIES}
    ${NVDS_LIBRARIES}
    ${CUDA_LIBRARIES}
    X11::X11
    m
)

