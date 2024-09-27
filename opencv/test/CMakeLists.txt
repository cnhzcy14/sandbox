if(OpenCV_FOUND)
add_executable          (cv_test cv_test.cpp)
target_link_libraries   (cv_test ${OpenCV_LIBS})

add_executable          (cv_test_ppm cv_test_ppm.cpp)
target_link_libraries   (cv_test_ppm ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# add_executable		( cv_para_test cv_para_test.cpp)
# target_link_libraries 	( cv_para_test ${LIBS} )

# add_executable		( cv_nv12_test cv_nv12_test.cpp)
# target_link_libraries 	( cv_nv12_test ${LIBS} )

add_executable          (cv_yuv cv_yuv.cpp)
target_link_libraries   (cv_yuv ${OpenCV_LIBS})

add_executable          (bg_sub bg_sub.cpp)
target_link_libraries   (bg_sub ${OpenCV_LIBS})

add_executable          (object_detection object_detection.cpp)
target_link_libraries   (object_detection ${OpenCV_LIBS})

add_executable          (find_obj find_obj.cpp)
target_link_libraries   (find_obj ${OpenCV_LIBS})

add_executable          (find_obj_queue find_obj_queue.cpp)
target_link_libraries   (find_obj_queue ${OpenCV_LIBS})

add_executable          (cv_scale cv_scale.cpp)
target_link_libraries   (cv_scale ${OpenCV_LIBS})

add_executable          (cv_video cv_video.cpp)
target_link_libraries   (cv_video ${OpenCV_LIBS})

install(
    TARGETS 
        cv_test 
        cv_yuv 
    RUNTIME DESTINATION bin)
endif()