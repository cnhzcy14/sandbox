#define HIST_BINS 256

__kernel void histogram(__global uchar *data, int numData,
                        __global int *histogram) {
  __local int localHistogram[HIST_BINS];
  int lid = get_local_id(0);
  int gid = get_global_id(0);

  for (int i = lid; i < HIST_BINS; i += get_local_size(0)) {
    localHistogram[i] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = gid; i < numData; i += get_global_size(0)) {
    atomic_add(&localHistogram[data[i]], 1);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = lid; i < HIST_BINS; i += get_local_size(0)) {
    atomic_add(&histogram[i], localHistogram[i]);
  }
}

__kernel void bitwise_inv_buf_8uC1(__global uchar *pSrcDst, int srcDstStep,
                                   int rows, int cols) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int idx = mad24(y, srcDstStep, x);
  pSrcDst[idx] = ~pSrcDst[idx];
}

__kernel void bitwise_inv_img_8uC1(__read_only image2d_t srcImg,
                                   __write_only image2d_t dstImg) {
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

  // uchar data = srcptr[idx];
  // uchar t = thresh;
  // uchar m = max_val;
  // data = data > thresh ? max_val : 0;
  // srcptr[idx] = data;
  srcptr[idx] = srcptr[idx] > thresh ? max_val : 0;
}

__kernel void maxloc(__global uchar *srcptr, int srcDstStep, int rows, int cols,
                     __global uchar *maxVal, __global int *maxLoc,
                     __global int *maxCount) {
  int col = get_global_id(0);

  // find maximum for the kernel
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

__kernel void gaussian(__read_only image2d_t srcImg,
                          __write_only image2d_t dstImg,
                          __constant float *filter, int filterWidth,
                          int filterHeight, sampler_t sampler) {
  /* Store each work-item’s unique row and column */
  int column = get_global_id(0);
  int row = get_global_id(1);

  /* Half the width of the filter is needed for indexing
   * memory later */
  int halfWidth = (int)(filterWidth / 2);
  int halfHeight = (int)(filterHeight / 2);

  /* All accesses to images return data as four-element vector
   * (i.e., float4), although only the ’x’ component will contain
   * meaningful data in this code */
  float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};

  /* Iterator for the filter */
  int filterIdx = 0;

  /* Each work-item iterates around its local area based on the
   * size of the filter */
  int2 coords; // Coordinates for accessing the image

  /* Iterate the filter rows */
  for (int i = -halfHeight; i <= halfHeight; i++) {
    coords.y = row + i;
    /* Iterate over the filter columns */
    for (int j = -halfWidth; j <= halfWidth; j++) {
      coords.x = column + j;

      /* Read a pixel from the image. A single channel image
       * stores the pixel in the ’x’ coordinate of the returned
       * vector. */
      float4 pixel;
      pixel = read_imagef(srcImg, sampler, coords);
      sum.x += pixel.x * filter[filterIdx++];
    }
  }

  /* Copy the data to the output image */
  coords.x = column;
  coords.y = row;
  write_imagef(dstImg, coords, sum);
}