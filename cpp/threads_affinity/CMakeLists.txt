add_executable			( launch_thread_per_cpu launch_thread_per_cpu.cpp )
target_link_libraries 	( launch_thread_per_cpu  Threads::Threads )

add_executable			( launch_threads_report_cpu launch_threads_report_cpu.cpp )
target_link_libraries 	( launch_threads_report_cpu  Threads::Threads )

add_executable			( thread_id_native_handle thread_id_native_handle.cpp )
target_link_libraries 	( thread_id_native_handle  Threads::Threads )

add_executable			( set_affinity set_affinity.cpp )
target_link_libraries 	( set_affinity  Threads::Threads )

add_executable			( thread_workload_benchmark thread_workload_benchmark.cpp )
target_link_libraries 	( thread_workload_benchmark  Threads::Threads )

install(TARGETS launch_thread_per_cpu launch_threads_report_cpu thread_id_native_handle set_affinity thread_workload_benchmark
  RUNTIME DESTINATION bin)
