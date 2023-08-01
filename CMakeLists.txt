cmake_minimum_required(VERSION 3.15)
get_filename_component 	(PROJECT_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
project(${PROJECT_NAME} 
  DESCRIPTION "My test code" 
  LANGUAGES CXX C)

# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# option(MY_OPTION "This is settable from the command line" OFF)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
# set(CMAKE_INSTALL_PREFIX "/usr/lib/aarch64-linux-gnu")

# find_package(CUDA  REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})
# set(LIBS ${LIBS} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_npp_LIBRARY} ${CUDA_nppi_LIBRARY} ${CUDA_cusparse_LIBRARY} cuda cusolver )

# pkg_check_modules(GLIB_JSON REQUIRED IMPORTED_TARGET json-glib-1.0)
# set(LIBS ${LIBS} PkgConfig::GLIB_JSON)

find_package(PkgConfig REQUIRED) # for some lib without cmakeconfig
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
# find_package(apriltag REQUIRED)
# find_package(vpi REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)
find_library(MATH_LIBRARY m) # special for finding libm (math.h)
pkg_check_modules(DRM REQUIRED IMPORTED_TARGET libdrm)
pkg_check_modules(V4L2 REQUIRED IMPORTED_TARGET libv4l2)
pkg_check_modules(GBM REQUIRED IMPORTED_TARGET gbm)
find_package(CUDA)
if(${CUDA_FOUND})
  set(OpenCL_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
  find_package(OpenCL REQUIRED)
else()
  find_package(OpenCL REQUIRED)
endif()

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

# opencl
# include_directories(~/work/project/test_code/opencl/common)
# set(OCL_SRC_LIST ${OCL_SRC_LIST} 
#                 ~/work/project/test_code/opencl/common/device_info.c 
#                 ~/work/project/test_code/opencl/common/wtime.c)

# jsoncpp
# pkg_check_modules(JSONCPP REQUIRED IMPORTED_TARGET jsoncpp)
# set(LIBS ${LIBS} PkgConfig::JSONCPP)

# 0mq
# pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)
# set(LIBS ${LIBS} PkgConfig::ZMQ)

# gstreamer
# pkg_check_modules(GST_RTSP REQUIRED IMPORTED_TARGET gstreamer-rtsp-server-1.0)
# set(LIBS ${LIBS} PkgConfig::GST_RTSP)


# find_package(GStreamer REQUIRED)
# include_directories( ${GSTREAMER_INCLUDE_DIRS} )
# include_directories( /usr/include/gstreamer-1.0 )
# set(LIBS ${LIBS} ${GSTREAMER_LIBRARIES} ${GSTREAMER_APP_LIBRARIES} ${GSTREAMER_BASE_LIBRARIES} gstallocators-1.0 gstcontroller-1.0 gstaudio-1.0)


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

# opencv
add_subdirectory 		(opencv/test)
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
# add_subdirectory 		(gstreamer/tutorial)

# add_subdirectory 		(cuda/test)
# add_subdirectory 		(cuda/ratio_test)
# add_subdirectory 		(cuda/batchCUBLAS)
# add_subdirectory 		(cuda/cuSolverDn_LinearSolver)

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
# add_subdirectory 		(vpi/samples)



message(STATUS)
message(STATUS "======================================================================")
message(STATUS "General configuration for: ${PROJECT_NAME}")
message(STATUS "  C/C++ Standard:       ${CMAKE_C_STANDARD}/${CMAKE_CXX_STANDARD}")
message(STATUS "  CMake Version:        ${CMAKE_VERSION}")
message(STATUS "  Default Install:      ${CMAKE_INSTALL_PREFIX}")
message(STATUS "======================================================================")
message(STATUS)
