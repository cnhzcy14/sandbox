add_executable		( omp_test omp_test.c  )
target_link_libraries 	( omp_test  OpenMP::OpenMP_C )

add_executable		( thread_safe_test thread_safe_test.c  )
target_link_libraries 	( thread_safe_test  OpenMP::OpenMP_C )

add_executable		( vec_add vec_add.c  )
target_link_libraries 	( vec_add  OpenMP::OpenMP_C )

install(TARGETS omp_test thread_safe_test vec_add
  RUNTIME DESTINATION bin)