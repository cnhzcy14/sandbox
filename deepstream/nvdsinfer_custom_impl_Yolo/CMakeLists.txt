add_library(nvdsinfer_custom_impl_Yolo SHARED
    layers/activation_layer.cpp
    layers/batchnorm_layer.cpp
    layers/channels_layer.cpp
    layers/convolutional_layer.cpp
    layers/deconvolutional_layer.cpp
    layers/implicit_layer.cpp
    layers/pooling_layer.cpp
    layers/reorg_layer.cpp
    layers/route_layer.cpp
    layers/sam_layer.cpp
    layers/shortcut_layer.cpp
    layers/slice_layer.cpp
    layers/upsample_layer.cpp

    calibrator.cpp
    nvdsinfer_yolo_engine.cpp
    nvdsinitinputlayers_Yolo.cpp
    nvdsparsebbox_Yolo.cpp 
    nvdsparsebbox_Yolo_cuda.cu
    utils.cpp
    yolo.cpp

    yoloForward.cu
    yoloForward_nc.cu
    yoloForward_v2.cu
    yoloPlugins.cpp
)
target_include_directories(nvdsinfer_custom_impl_Yolo PRIVATE
    ${NVDS_INCLUDE_DIRS}
    ${DEEPSTREAM_APP_COMMON_INCLUDE_DIRS}
)
set_target_properties(nvdsinfer_custom_impl_Yolo PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON
                            CUDA_ARCHITECTURES "86;89;90")
target_link_libraries(nvdsinfer_custom_impl_Yolo
    ${OpenCV_LIBS}
    trt::nvonnxparser
    stdc++fs
    $<$<TARGET_EXISTS:CUDA::cudart>:CUDA::cudart>
    ${NVDS_LIBRARIES}
    ${CUDA_LIBRARIES}
)