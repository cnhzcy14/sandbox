add_executable			( modeset-single-buffer modeset-single-buffer.c )
target_link_libraries 	( modeset-single-buffer  PkgConfig::DRM )


install(TARGETS modeset-single-buffer
  RUNTIME DESTINATION bin)
