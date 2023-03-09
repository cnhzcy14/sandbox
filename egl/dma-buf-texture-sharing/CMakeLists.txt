add_executable			( dmabufshare main.c socket.h window.h render.h )
target_link_libraries 	( dmabufshare  GL OpenGL::EGL X11::X11 ${MATH_LIBRARY})
# OpenGL::GL doesn't import libGL, why?

install(TARGETS dmabufshare
  RUNTIME DESTINATION bin)
