
# 现代 CMake 方式：使用 CUDA::cudart 目标
add_executable(deepstream-test1
    deepstream_test1_app.c
)

# 不需要手动包含 CUDA 头文件路径，CUDA::cudart 会自动处理
target_include_directories(deepstream-test1 PRIVATE
    ${NVDS_INCLUDE_DIRS}
)

set_target_properties(deepstream-test1 PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
)

# 现代 CMake 方式：使用 CUDA::cudart 目标链接，不需要硬编码路径
target_link_libraries(deepstream-test1
    PkgConfig::GSTREAMER
    PkgConfig::GST_RTSP
    PkgConfig::GLIB
    PkgConfig::GLIB_JSON
    ${YAML_CPP_LIBRARIES}
    ${NVDS_LIBRARIES}
    $<$<TARGET_EXISTS:CUDA::cudart>:CUDA::cudart>  # 条件链接，更robust
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
    $<$<TARGET_EXISTS:CUDA::cudart>:CUDA::cudart>
    X11::X11
    m
)