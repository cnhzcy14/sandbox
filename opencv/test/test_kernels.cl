__kernel void bitwise_inv_buf_8uC1(__global uchar *pSrcDst, int srcDstStep,
                                   int rows, int cols) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int idx = mad24(y, srcDstStep, x);
  pSrcDst[idx] = ~pSrcDst[idx];
}

__kernel void bitwise_inv_img_8uC1(read_only image2d_t srcImg,
                                   read_write image2d_t dstImg) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int2 coord = (int2)(x, y);
  uint4 val = read_imageui(srcImg, coord);
  val.x = (~val.x) & 0x000000FF;
  write_imageui(dstImg, coord, val);
}

__kernel void bitwise_inv_img_8uC1_ip(read_write image2d_t srcImg) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int2 coord = (int2)(x, y);
  uint4 val = read_imageui(srcImg, coord);
  val.x = (~val.x) & 0x000000FF;
  write_imageui(srcImg, coord, val);
}

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