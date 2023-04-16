add_executable			  ( modeset modeset.c )
target_link_libraries ( modeset PkgConfig::DRM )

add_executable			  ( modeset-double-buffered modeset-double-buffered.c )
target_link_libraries ( modeset-double-buffered PkgConfig::DRM )

add_executable			  ( modeset-vsync modeset-vsync.c )
target_link_libraries ( modeset-vsync PkgConfig::DRM )

add_executable			  ( modeset-atomic modeset-atomic.c)
target_link_libraries ( modeset-atomic PkgConfig::DRM )

install(TARGETS 
  modeset 
  modeset-double-buffered 
  modeset-vsync 
  modeset-atomic
  RUNTIME DESTINATION bin)
