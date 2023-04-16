add_executable			( enum enum.c )
target_link_libraries 	( enum  PkgConfig::DRM )

add_executable			( drmsend drmsend.c )
target_link_libraries 	( drmsend  PkgConfig::DRM )

install(TARGETS enum drmsend
  RUNTIME DESTINATION bin)
