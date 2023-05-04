project('sandbox', 'c', 'cpp',
  version: '1.0',
  default_options: ['c_std=c11', 'cpp_std=c++14','buildtype=release']
)

cc = meson.get_compiler('c')
dep_m = cc.find_library('m', required : false)
dep_ocv = dependency('opencv4')
dep_eigen = dependency('eigen3')
dep_omp = dependency('openmp')
dep_threads = dependency('threads')
dep_egl = dependency('egl')
dep_gles2 = dependency('glesv2')
with_gles3 = cc.check_header('GLES3/gl3.h', dependencies: dep_gles2)
dep_x11 = dependency('x11')
dep_dl = dependency('dl')
dep_drm = dependency('libdrm')
dep_v4l2 = dependency('libv4l2')
dep_gbm = dependency('gbm')


# drm
subdir('drm/gbm')
subdir('drm/drmtoy')
