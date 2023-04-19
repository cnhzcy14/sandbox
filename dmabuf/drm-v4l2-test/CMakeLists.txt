add_executable			  ( dmabuf-sharing dmabuf-sharing.c )
target_link_libraries ( dmabuf-sharing PkgConfig::DRM PkgConfig::V4L2 )


install(TARGETS 
  dmabuf-sharing 
  RUNTIME DESTINATION bin)
