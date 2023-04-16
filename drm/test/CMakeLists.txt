add_executable			  ( modeset-single-buffer modeset-single-buffer.c )
target_link_libraries ( modeset-single-buffer PkgConfig::DRM )

add_executable			  ( planetest planetest.c dev.c bo.c modeset.c)
target_link_libraries ( planetest PkgConfig::DRM )

install(TARGETS 
  modeset-single-buffer 
  planetest
  RUNTIME DESTINATION bin)
