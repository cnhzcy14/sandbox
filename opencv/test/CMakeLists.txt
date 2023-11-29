if(OpenCV_FOUND)
add_executable          (cv_test cv_test.cpp)
target_link_libraries   (cv_test ${OpenCV_LIBS})

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


install(
    TARGETS 
        cv_test 
        cv_yuv 
    RUNTIME DESTINATION bin)
endif()