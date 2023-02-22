cmake_minimum_required(VERSION 3.15)
get_filename_component 	(PROJECT_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
project(${PROJECT_NAME} 
  DESCRIPTION "My test code" 
  LANGUAGES CXX C)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

# option(MY_OPTION "This is settable from the command line" OFF)
# set(CMAKE_C_FLAGS "-O3  -D_GNU_SOURCE -fPIC -fopenmp")
# set(CMAKE_CXX_FLAGS "-O3  -fPIC -fopenmp")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
# set(CMAKE_INSTALL_PREFIX "/usr/lib/aarch64-linux-gnu")

# find_package (Threads)
# set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT} )

# find_package( OpenCL REQUIRED )
# include_directories(${OpenCL_INCLUDE_DIR})
# link_directories(/usr/lib/aarch64-linux-gnu)
# set(LIBS ${LIBS} Mali)
# include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)
# include_directories(/opt/rocm/opencl/include)
# link_directories(/opt/rocm/opencl/lib/x86_64)
# include_directories (/opt/intel/opencl/include)
# link_directories (/opt/intel/opencl/)
# set(LIBS ${LIBS} ${OpenCL_LIBRARY})
# set(LIBS ${LIBS} OpenCL)

# acl
# link_directories(/opt/acl/lib)
# include_directories(/opt/acl/include)
# set(LIBS ${LIBS} arm_compute arm_compute_core arm_compute_graph)

# find_package(CUDA  REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})
# set(LIBS ${LIBS} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_npp_LIBRARY} ${CUDA_nppi_LIBRARY} ${CUDA_cusparse_LIBRARY} cuda cusolver )

# include_directories(/opt/DahuaTech/MVviewer/include)
# link_directories(/opt/DahuaTech/MVviewer/lib)
# set(LIBS ${LIBS} MVSDK ImageConvert )


# pkg_check_modules(GLIB_JSON REQUIRED IMPORTED_TARGET json-glib-1.0)
# set(LIBS ${LIBS} PkgConfig::GLIB_JSON)

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED NO_MODULE)

# pkg_check_modules(APRIL REQUIRED IMPORTED_TARGET apriltag)
# set(LIBS ${LIBS} PkgConfig::APRIL)

# pkg_check_modules(BLAS REQUIRED IMPORTED_TARGET blas64)
# set(LIBS ${LIBS} PkgConfig::BLAS)

# pkg_check_modules(LAPACK REQUIRED IMPORTED_TARGET lapack64)
# set(LIBS ${LIBS} PkgConfig::LAPACK)


# pkg_check_modules(GLOG REQUIRED IMPORTED_TARGET libglog)
# set(LIBS ${LIBS} PkgConfig::GLOG)

# pkg_check_modules(MSGPACK REQUIRED IMPORTED_TARGET msgpack)
# set(LIBS ${LIBS} PkgConfig::MSGPACK)

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


# add_subdirectory 		(opencl/test)
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
# add_subdirectory 		(opencl/clinfo)
# add_subdirectory 		(opencl/matmul)
# add_subdirectory 		(opencl/marching_squares)
# add_subdirectory 		(opencl/local_thresh)
# add_subdirectory 		(opencl/clpeak)

# add_subdirectory 		(cpp_test/simple_test)
# add_subdirectory 		(cpp_test/threads_affinity)

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
# add_subdirectory 		(opencv/apriltag)
# add_subdirectory 		(opencv/maskImage)

# OpenMP
# add_subdirectory 		(openmp/vector)
# add_subdirectory 		(openmp/omp_test)

# acl
# add_subdirectory 		(acl/examples)

# # blas, lapack
# add_subdirectory 		(blas/blas_test)

# # ceres
# add_subdirectory 		(ceres/examples)

# add_subdirectory 		(misc/csv)

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
# add_subdirectory 		(cpp_test)
# add_subdirectory 		(c11_test)
# add_subdirectory 		(v4l2_test)
# add_subdirectory 		(v4l2_test_gst)
# add_subdirectory 		(eigen_test)
# add_subdirectory 		(ft_test)
# add_subdirectory 		(glfw_test)
# add_subdirectory 		(gst_test)
#add_subdirectory 		(convolution)
# add_subdirectory 		(json)
# add_subdirectory 		(msgpack)

message(STATUS)
message(STATUS "General configuration for ${PROJECT_NAME}")
message(STATUS "======================================================================")
message(STATUS "Platform:")
message(STATUS "  Arch / OS:            ${CMAKE_HOST_SYSTEM_PROCESSOR} / ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION}")
message(STATUS "  C/C++ Compiler:       ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C/C++ Standard:       ${CMAKE_C_STANDARD} ${CMAKE_CXX_STANDARD}")
message(STATUS "  CMake Version:        ${CMAKE_VERSION}")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "======================================================================")
message(STATUS)