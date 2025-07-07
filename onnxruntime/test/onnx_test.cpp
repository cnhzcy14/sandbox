#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main()
{

    // Initialize ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_test");

    // Print ONNX Runtime version
    std::cout << "ONNX Runtime version: "
              << ORT_API_VERSION << std::endl;

    // Create session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    std::cout << "ONNX Runtime initialization successful" << std::endl;
    return 0;
}