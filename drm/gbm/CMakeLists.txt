add_executable			( render-nodes-minimal main.c )
target_link_libraries 	( render-nodes-minimal OpenGL::EGL PkgConfig::GBM)

install(TARGETS render-nodes-minimal
  RUNTIME DESTINATION bin)
