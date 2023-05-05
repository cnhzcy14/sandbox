add_executable			( shctx main.c sdr.c )
target_link_libraries 	( shctx  GL OpenGL::EGL X11::X11 ${CMAKE_DL_LIBS})
# OpenGL::GL doesn't import libGL, why?

install(TARGETS shctx
  RUNTIME DESTINATION bin)
