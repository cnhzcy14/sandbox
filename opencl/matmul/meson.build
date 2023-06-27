dep_common = [dep_ocl]

executable(
    'matmul',
    files(
      'matmul.c',
      'matrix_lib.c',
      'common/wtime.c',
      'common/device_info.c',
    ),
    dependencies: dep_common,
    install: true,
)
