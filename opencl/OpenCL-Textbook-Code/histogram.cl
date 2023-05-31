#define HIST_BINS 256

__kernel void histogram(__global uchar *data, int numData,
                        __global int *histogram) {
  __local int localHistogram[HIST_BINS];
  int lid = get_local_id(0);
  int gid = get_global_id(0);

  /* Initialize local histogram to zero */
  for (int i = lid; i < HIST_BINS; i += get_local_size(0)) {
    localHistogram[i] = 0;
  }
  /* Wait until all work-items within
   * the work-group have completed */
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Compute local histogram */
  for (int i = gid; i < numData; i += get_global_size(0)) {
    atomic_add(&localHistogram[data[i]], 1);
  }

  /* Wait until all work-items within
   * the work-group have completed */
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Write the local histogram out to
   * the global histogram */
  for (int i = lid; i < HIST_BINS; i += get_local_size(0)) {
    atomic_add(&histogram[i], localHistogram[i]);
  }
}


__kernel void bitwise_inv_buf_8uC1(__global uchar *pSrcDst, int srcDstStep,
                                   int rows, int cols) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  // int idx = mad24(y, srcDstStep, x);
  int idx = y*srcDstStep +x;
  pSrcDst[idx] = ~pSrcDst[idx];
}

__kernel void bitwise_inv_img_8uC1(read_only image2d_t srcImg,
                                   write_only image2d_t dstImg) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int2 coord = (int2)(x, y);
  uint4 val = read_imageui(srcImg, coord);
  val.x = (~val.x) & 0x000000FF;
  write_imageui(dstImg, coord, val);
}

// __kernel void bitwise_inv_img_8uC1_ip(read_write image2d_t srcImg) {
//   int x = get_global_id(0);
//   int y = get_global_id(1);
//   int2 coord = (int2)(x, y);
//   uint4 val = read_imageui(srcImg, coord);
//   val.x = (~val.x) & 0x000000FF;
//   write_imageui(srcImg, coord, val);
// }

__kernel void threshold(__global uchar *srcptr, int srcDstStep, int rows,
                        int cols, uchar thresh, uchar max_val) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int idx = mad24(y, srcDstStep, x);

  srcptr[idx] = srcptr[idx] > thresh ? max_val : 0;
}

__kernel void maxLoc(__global uchar *srcptr, int srcDstStep, int rows, int cols,
                     __global uchar *maxVal, __global int *maxLoc,
                     __global int *maxCount) {
  int col = get_global_id(0);

  // find minimum for the kernel
  for (int row = 0; row < rows; row++) {
    int idx = mad24(row, srcDstStep, col);
    if (srcptr[idx] > maxVal[col]) {
      maxVal[col] = srcptr[idx];
      maxCount[col] = 0;
      int r = mad24(maxCount[col]++, srcDstStep, col);
      maxLoc[r] = row;
    } else if (srcptr[idx] == maxVal[col]) {
      int r = mad24(maxCount[col]++, srcDstStep, col);
      maxLoc[r] = row;
    }
  }
}