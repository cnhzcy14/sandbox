add_executable			( drm-enum drm-enum.c )
target_link_libraries 	( drm-enum  PkgConfig::DRM )

add_executable			( drmsend drmsend.c )
target_link_libraries 	( drmsend  PkgConfig::DRM )

install(TARGETS drm-enum drmsend
  RUNTIME DESTINATION bin)
