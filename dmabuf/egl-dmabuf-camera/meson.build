dep_common = [dep_egl, dep_gles2, dep_x11]

executable(
    'egl-dmabuf-camera',
    files(
      'main.c',
      'egl_x11.c',
      'engine.c'
    ),
    dependencies: dep_common,
    install: true,
)