# Add test executable
add_executable(onnx_test
    onnx_test.cpp
)

target_link_libraries(onnx_test
    PRIVATE
    PkgConfig::ONNXRUNTIME
)