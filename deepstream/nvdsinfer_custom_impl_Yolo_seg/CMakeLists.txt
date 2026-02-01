add_library(nvdsinfer_custom_impl_Yolo_seg SHARED
    nvdsparseseg_Yolo.cpp
    
    trt_plugins/common.cpp
    trt_plugins/efficientNMSPlugin/efficientNMSPlugin.cpp
    trt_plugins/efficientNMSPlugin/efficientNMSInference.cu
    trt_plugins/roiAlignPlugin/roiAlignPlugin.cpp
    trt_plugins/roiAlignPlugin/roiAlignKernel.cu
)

target_include_directories(nvdsinfer_custom_impl_Yolo_seg PRIVATE
    ${NVDS_INCLUDE_DIRS}
    ${DEEPSTREAM_APP_COMMON_INCLUDE_DIRS}
)

set_target_properties(nvdsinfer_custom_impl_Yolo_seg PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86;89;90"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 抑制 TensorRT IPluginV2DynamicExt 弃用警告
# 这些插件基于 NVIDIA 官方实现，等待官方升级到 IPluginV3
target_compile_options(nvdsinfer_custom_impl_Yolo_seg PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
)

target_link_libraries(nvdsinfer_custom_impl_Yolo_seg
    ${OpenCV_LIBS}
    trt::nvinfer
    stdc++fs
    $<$<TARGET_EXISTS:CUDA::cudart>:CUDA::cudart>
    ${NVDS_LIBRARIES}
    ${CUDA_LIBRARIES}
)
