add_executable			  ( modeset-single-buffer modeset-single-buffer.c )
target_link_libraries ( modeset-single-buffer PkgConfig::DRM )

add_executable			  ( planetest planetest.c dev.c bo.c modeset.c)
target_link_libraries ( planetest PkgConfig::DRM )

add_executable			  ( modeset-plane-test modeset-plane-test.c)
target_link_libraries ( modeset-plane-test PkgConfig::DRM )

add_executable			  ( modeset-atomic-plane modeset-atomic-plane.c)
target_link_libraries ( modeset-atomic-plane PkgConfig::DRM )

install(TARGETS 
  modeset-single-buffer 
  planetest
  modeset-plane-test
  modeset-atomic-plane
  RUNTIME DESTINATION bin)
