if(DeepstreamAppCommon_FOUND)
message(STATUS "  DeepstreamAppCommon:  ${DeepstreamAppCommon_FOUND}"                   )
add_subdirectory 		(deepstream-test1)
add_subdirectory 		(deepstream-app)
add_subdirectory 		(nvdsinfer_custom_impl_Yolo)
add_subdirectory 		(nvdsinfer_custom_impl_Yolo_seg)
endif()

