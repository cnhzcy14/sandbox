cmake_minimum_required(VERSION 3.15)
get_filename_component 	(PROJECT_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
project(${PROJECT_NAME} 
  DESCRIPTION "My test code" 
  LANGUAGES CXX C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin)

find_package(CUDA)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
  cuda_select_nvcc_arch_flags(ARCH_FLAGS)
  set(OpenCL_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
endif()
find_package(OpenCL QUIET)

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
endif()

find_package(OpenCV QUIET)
find_package(yaml-cpp)

find_package(Eigen3 QUIET NO_MODULE)
find_package(apriltag QUIET)
find_package(vpi QUIET)
find_package(OpenMP QUIET)
find_package(Threads QUIET)
find_package(OpenGL QUIET)
find_library(MATH_LIBRARY m) # special for finding libm (math.h)
find_package(TensorRT QUIET)
find_package(DeepstreamAppCommon QUIET)
find_package(NVDS QUIET)




# link_directories(/usr/lib/aarch64-linux-gnu)
# set(LIBS ${LIBS} Mali)

# if(DEFINED OPENVX_INCLUDES)
#     message(STATUS "OPENVX_INCLUDES: \t${OPENVX_INCLUDES}")
#     include_directories(${OPENVX_INCLUDES})
# else()
#     message(STATUS "OPENVX_INCLUDES: \t<empty>")
# endif()

# if(DEFINED OPENVX_LIBRARIES)
#     message(STATUS "OPENVX_LIBRARIES: \t${OPENVX_LIBRARIES}")
# else()
#     message(STATUS "OPENVX_LIBRARIES: \t<empty>")
# endif()

# pkg_check_modules(BLAS REQUIRED IMPORTED_TARGET blas64)
# set(LIBS ${LIBS} PkgConfig::BLAS)

# pkg_check_modules(LAPACK REQUIRED IMPORTED_TARGET lapack64)
# set(LIBS ${LIBS} PkgConfig::LAPACK)


# pkg_check_modules(GLOG REQUIRED IMPORTED_TARGET libglog)
# set(LIBS ${LIBS} PkgConfig::GLOG)

# find_package(Ceres REQUIRED)
# set(LIBS ${LIBS} ceres )


# jsoncpp
# pkg_check_modules(JSONCPP REQUIRED IMPORTED_TARGET jsoncpp)
# set(LIBS ${LIBS} PkgConfig::JSONCPP)

# 0mq
# pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)
# set(LIBS ${LIBS} PkgConfig::ZMQ)

add_subdirectory 		(opencl/test)
# add_subdirectory 		(opencl/Exercises/Exercise01/C)
# add_subdirectory 		(opencl/Solutions/Exercise04/C)
# add_subdirectory 		(opencl/Solutions/Exercise05/C)
# add_subdirectory 		(opencl/Solutions/Exercise08/C)
# add_subdirectory 		(opencl/Solutions/Exercise09/C)
# add_subdirectory 		(opencl/Ch11/bsort)
# add_subdirectory 		(opencl/Ch11/string_search)
# # add_subdirectory 		(opencl/Ch7/atomic)
# add_subdirectory 		(opencl/Ch14/fft)
# add_subdirectory 		(opencl/android-opencl-blur-release)
add_subdirectory 		(opencl/matmul)
# add_subdirectory 		(opencl/marching_squares)
# add_subdirectory 		(opencl/local_thresh)
# add_subdirectory 		(opencl/OpenCL-Textbook-Code)

# openvx
# add_subdirectory 		(openvx/test)
# add_subdirectory 		(openvx/bubble-pop)

add_subdirectory 		(opencv/test)
add_subdirectory 		(opencv/ocl_interop)
# add_subdirectory 		(opencv/dist)
# add_subdirectory 		(opencv/warpCamera)
# add_subdirectory 		(opencv/thresholding)
# add_subdirectory 		(opencv/contours)
# add_subdirectory 		(opencv/dahua_camera)
# add_subdirectory 		(opencv/camera)
# add_subdirectory 		(opencv/calibration)
# add_subdirectory 		(opencv/stitch_test)
# add_subdirectory 		(opencv/denseflow)
# add_subdirectory 		(opencv/feature_detect)
# add_subdirectory 		(opencv/frozenGraph)
# add_subdirectory 		(opencv/dnn_test)
# add_subdirectory 		(opencv/aruco)
# add_subdirectory 		(opencv/maskImage)

# add_subdirectory 		(apriltag)

# OpenMP
add_subdirectory 		(openmp/fast_software_implementation)
add_subdirectory 		(openmp/test)


# # blas, lapack
# add_subdirectory 		(blas/blas_test)

# # ceres
# add_subdirectory 		(ceres/examples)

# 0mq
# add_subdirectory 		(zmq_code/pair)
# add_subdirectory 		(zmq_code/publisher0)
# add_subdirectory 		(zmq_code/subscriber0)
# add_subdirectory 		(zmq_code/json_msg)

# eigen
add_subdirectory 		(eigen)

# gstreamer
# add_subdirectory 		(gstreamer/controller_test)
# add_subdirectory 		(gstreamer/gst_test)
# add_subdirectory 		(gstreamer/gobject_test)
# add_subdirectory 		(gstreamer/gobject_tutorial/05_public_methods)
# add_subdirectory 		(gstreamer/gobject_tutorial/06_inheritence)
# add_subdirectory 		(gstreamer/gobject_tutorial/07_virtual_methods)
add_subdirectory 		(gstreamer/tutorial)

add_subdirectory 		(cuda/test)
# add_subdirectory 		(cuda/ratio_test)
# add_subdirectory 		(cuda/batchCUBLAS)
add_subdirectory 		(cuda/cuSolverDn_LinearSolver)

# cpp
add_subdirectory 		(cpp/test)
add_subdirectory 		(cpp/threads_affinity)

# c
add_subdirectory 		(c/test)

# dma-buf
add_subdirectory 		(dmabuf/dma-buf-texture-sharing)
add_subdirectory 		(dmabuf/drm-v4l2-test)
add_subdirectory 		(dmabuf/shctx)

# drm
add_subdirectory 		(drm/drmtoy)
add_subdirectory 		(drm/gbm)
add_subdirectory 		(drm/test)
add_subdirectory 		(drm/drm-howto)

# add_subdirectory 		(ft_test)
# add_subdirectory 		(glfw_test)
# add_subdirectory 		(gst_test)
# add_subdirectory 		(json)
# add_subdirectory 		(msgpack)

# VPI
add_subdirectory 		(vpi/samples)

# deepstream
add_subdirectory 		(deepstream/deepstream-test1)
# add_subdirectory 		(deepstream/deepstream-app)
# add_subdirectory 		(deepstream/nvdsinfer_custom_impl_Yolo)


message(STATUS)
message(STATUS "======================================================================" )
message(STATUS "Author:                 Simon CHENG Ye"                                 )  
message(STATUS "Project Name:           ${PROJECT_NAME}"                                )
message(STATUS "  ------ Find by pkgconfig: "                                           )
message(STATUS "  DRM:                  ${DRM_FOUND}"                                   )
message(STATUS "  V4L2:                 ${V4L2_FOUND}"                                  )
message(STATUS "  GBM:                  ${GBM_FOUND}"                                   )
message(STATUS "  GLIB:                 ${GLIB_FOUND}"                                  )
message(STATUS "  GLIB_JSON:            ${GLIB_JSON_FOUND}"                             )
message(STATUS "  GSTREAMER:            ${GSTREAMER_FOUND}"                             )
message(STATUS "  GST_RTSP:             ${GST_RTSP_FOUND}"                              )
message(STATUS "  GST_VIDEO:            ${GST_VIDEO_FOUND}"                             )
message(STATUS "  ------ Find by cmake: "                                               )
message(STATUS "  OpenCV:               ${OpenCV_FOUND}"                                )
message(STATUS "  YAML_CPP:             ${YAML_CPP_LIBRARIES}"                          )
message(STATUS "  ------ Find by modern cmake: "                                        )
message(STATUS "  X11:                  ${X11_FOUND}"                                   )
message(STATUS "  Eigen3:               ${Eigen3_FOUND}"                                )
message(STATUS "  apriltag:             ${apriltag_FOUND}"                              )
message(STATUS "  vpi:                  ${vpi_FOUND}"                                   )
message(STATUS "  OpenMP:               ${OpenMP_FOUND}"                                )
message(STATUS "  Threads:              ${Threads_FOUND}"                               )
message(STATUS "  OpenGL:               ${OpenGL_FOUND}"                                )
message(STATUS "  MATH_LIBRARY:         ${MATH_LIBRARY}"                                )
message(STATUS "  CUDA:                 ${CUDA_FOUND}"                                  )
message(STATUS "  CUDA:                 ${ARCH_FLAGS}"                                  )
message(STATUS "  OpenCL:               ${OpenCL_FOUND}"                                )
message(STATUS "  TensorRT:             ${TensorRT_FOUND}"                              )
message(STATUS "  DeepstreamAppCommon:  ${DeepstreamAppCommon_FOUND}"                   )
message(STATUS "  NVDS:                 ${NVDS_FOUND}"                                  )
message(STATUS "  ------"                                                               )
message(STATUS "  C/C++ Standard:       ${CMAKE_C_STANDARD}/${CMAKE_CXX_STANDARD}"      )
message(STATUS "  CMake Version:        ${CMAKE_VERSION}"                               )
message(STATUS "  Default Install:      ${CMAKE_INSTALL_PREFIX}"                        )
message(STATUS "======================================================================" )
message(STATUS)
