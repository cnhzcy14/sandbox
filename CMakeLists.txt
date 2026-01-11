cmake_minimum_required(VERSION 3.15)
get_filename_component 	(PROJECT_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)

project(${PROJECT_NAME}
  DESCRIPTION "My test code"
  LANGUAGES CXX C)

# 现代CMake方式：先检查CUDA语言支持
include(CheckLanguage)
check_language(CUDA)

# 检查CUDA是否可用，如果可用则启用CUDA支持
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)  # 现代 CMake 方式
    set(CUDA_ARCHITECTURES "86;89;90")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin)


message(STATUS)
message(STATUS "======================================================================" )
message(STATUS "Author:                 Simon CHENG Ye"                                 )  
message(STATUS "Project Name:           ${PROJECT_NAME}"                                )

find_package(OpenMP QUIET)
find_package(Threads QUIET)
message(STATUS "  Threads:              ${Threads_FOUND}"                               )


if(UNIX AND NOT APPLE)
  find_package(X11 QUIET)
  find_package(PkgConfig QUIET) # for some lib without cmakeconfig
  pkg_check_modules(DRM QUIET IMPORTED_TARGET libdrm)
  pkg_check_modules(V4L2 QUIET IMPORTED_TARGET libv4l2)
  pkg_check_modules(GBM QUIET IMPORTED_TARGET gbm)
  pkg_check_modules(GLIB QUIET IMPORTED_TARGET glib-2.0)
  pkg_check_modules(GLIB_JSON QUIET IMPORTED_TARGET json-glib-1.0)
  pkg_check_modules(GSTREAMER QUIET IMPORTED_TARGET gstreamer-1.0)
  pkg_check_modules(GST_RTSP QUIET IMPORTED_TARGET gstreamer-rtsp-server-1.0)
  pkg_check_modules(GST_VIDEO QUIET IMPORTED_TARGET gstreamer-video-1.0)
  pkg_check_modules(ZMQ QUIET IMPORTED_TARGET libzmq)
  pkg_check_modules(JSONCPP QUIET IMPORTED_TARGET jsoncpp)
  pkg_check_modules(GLOG QUIET IMPORTED_TARGET libglog)
  pkg_check_modules(BLAS QUIET IMPORTED_TARGET blas64)
  pkg_check_modules(LAPACK QUIET IMPORTED_TARGET lapack64)
endif()

find_package(OpenCL QUIET)
find_package(OpenCV QUIET)
find_package(yaml-cpp QUIET)
find_package(Eigen3 QUIET NO_MODULE)
find_package(apriltag QUIET)
find_package(vpi QUIET)
find_package(OpenGL QUIET)
find_library(MATH_LIBRARY m) # special for finding libm (math.h)
find_package(TensorRT QUIET)
find_package(DeepstreamAppCommon QUIET)
find_package(NVDS QUIET)
find_package(ZLIB QUIET)
find_package(Ceres QUIET)
find_package(OpenSSL QUIET)
find_package(CUDAToolkit QUIET)
find_package(MIVISIONX QUIET)

# link_directories(/usr/lib/aarch64-linux-gnu)
# set(LIBS ${LIBS} Mali)

# set(LIBS ${LIBS} PkgConfig::BLAS)
# set(LIBS ${LIBS} PkgConfig::LAPACK)
# set(LIBS ${LIBS} PkgConfig::GLOG)
# set(LIBS ${LIBS} PkgConfig::JSONCPP)
# set(LIBS ${LIBS} PkgConfig::ZMQ)

add_subdirectory 		(opencl)
add_subdirectory 		(openvx)
add_subdirectory 		(opencv)
add_subdirectory 		(apriltag)
add_subdirectory 		(openmp)
# add_subdirectory 		(blas/blas_test)
# add_subdirectory 		(ceres)
# add_subdirectory 		(zmq)
add_subdirectory 		(eigen)
add_subdirectory 		(gstreamer)
add_subdirectory 		(cuda)
add_subdirectory 		(cpp)
add_subdirectory 		(c)

# dma-buf
# add_subdirectory 		(dmabuf/dma-buf-texture-sharing)
# add_subdirectory 		(dmabuf/drm-v4l2-test)
# add_subdirectory 		(dmabuf/shctx)

# drm
# add_subdirectory 		(drm/drmtoy)
# add_subdirectory 		(drm/gbm)
# add_subdirectory 		(drm/test)
# add_subdirectory 		(drm/drm-howto)

# add_subdirectory 		(ft_test)
# add_subdirectory 		(glfw_test)
# add_subdirectory 		(gst_test)
# add_subdirectory 		(json)
# add_subdirectory 		(msgpack)

add_subdirectory 		(vpi)
add_subdirectory 		(deepstream)
add_subdirectory 		(zlib)
add_subdirectory 		(openssl)


message(STATUS "  DRM:                  ${DRM_FOUND}"                                   )
message(STATUS "  V4L2:                 ${V4L2_FOUND}"                                  )
message(STATUS "  GBM:                  ${GBM_FOUND}"                                   )
message(STATUS "  GLIB:                 ${GLIB_FOUND}"                                  )
message(STATUS "  GLIB_JSON:            ${GLIB_JSON_FOUND}"                             )
message(STATUS "  GST_RTSP:             ${GST_RTSP_FOUND}"                              )
message(STATUS "  GST_VIDEO:            ${GST_VIDEO_FOUND}"                             )
message(STATUS "  YAML_CPP:             ${YAML_CPP_LIBRARIES}"                          )
message(STATUS "  X11:                  ${X11_FOUND}"                                   )
message(STATUS "  OpenGL:               ${OpenGL_FOUND}"                                )
message(STATUS "  MATH_LIBRARY:         ${MATH_LIBRARY}"                                )
message(STATUS "  TensorRT:             ${TensorRT_FOUND}"                              )
message(STATUS "  NVDS:                 ${NVDS_FOUND}"                                  )
message(STATUS "  ------"                                                               )
message(STATUS "  C/C++ Standard:       ${CMAKE_C_STANDARD}/${CMAKE_CXX_STANDARD}"      )
message(STATUS "  C/C++ Compiler:       ${CMAKE_CXX_COMPILER_VERSION}"                  )
message(STATUS "  CMake Version:        ${CMAKE_VERSION}"                               )
message(STATUS "  Default Install:      ${CMAKE_INSTALL_PREFIX}"                        )
message(STATUS "======================================================================" )
message(STATUS)
