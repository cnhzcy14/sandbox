if(OpenCV_FOUND)
add_executable          (cv_test cv_test.cpp)
target_link_libraries   (cv_test ${OpenCV_LIBS})

# add_executable		( cv_para_test cv_para_test.cpp)
# target_link_libraries 	( cv_para_test ${LIBS} )

# add_executable		( cv_nv12_test cv_nv12_test.cpp)
# target_link_libraries 	( cv_nv12_test ${LIBS} )

add_executable          (cv_yuv cv_yuv.cpp)
target_link_libraries   (cv_yuv ${OpenCV_LIBS})

install(
    TARGETS 
        cv_test 
        cv_yuv 
    RUNTIME DESTINATION bin)
endif()