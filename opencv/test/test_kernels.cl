#define HIST_BINS 256
#define HALF_FILTER_SIZE 2

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

__kernel void bitwise_inv_buf_8uC1(__global uchar16 *pSrcDst, int srcDstStep,
                                   int rows, int cols) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int idx = mad24(y, srcDstStep, x);
  // uchar4 a = ~pSrcDst[idx];
  pSrcDst[idx] = pSrcDst[idx];
  // pSrcDst[idx] = (uchar16)(0, 20, 60, 100, 140, 180, 220, 255, 0, 20, 60, 100,
  //                          140, 180, 220, 255);
  // pSrcDst[idx].s0 = 100;
  // pSrcDst[idx].s1 = 140;
  // pSrcDst[idx].s2 = 180;
  // pSrcDst[idx].s3 = 220;
  // pSrcDst[idx].s4 = 100;
  // pSrcDst[idx].s5 = 140;
  // pSrcDst[idx].s6 = 180;
  // pSrcDst[idx].s7 = 220;
  // pSrcDst[idx].s8 = 100;
  // pSrcDst[idx].s9 = 140;
  // pSrcDst[idx].sA = 180;
  // pSrcDst[idx].sB = 220;
  // pSrcDst[idx].sC = 100;
  // pSrcDst[idx].sD = 140;
  // pSrcDst[idx].sE = 180;
  // pSrcDst[idx].sF = 220;
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

__kernel void threshold(__global uchar4 *srcptr, int srcDstStep, uchar thresh,
                        uchar max_val) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int idx = mad24(y, srcDstStep, x);
  uchar4 t = (uchar4)(thresh);
  uchar4 h = (uchar4)(max_val);
  uchar4 l = (uchar4)(0);
  // char4 c = srcptr[idx] > t;

  // srcptr[idx] = srcptr[idx] > thresh ? max_val : 0;
  // srcptr[idx] = ~srcptr[idx];
  srcptr[idx] = select(l, h, srcptr[idx] > t);
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

__kernel void maxlocvec(__global uchar4 *srcptr, int srcDstStep, int rows,
                        int cols, __global uchar4 *maxVal, __global int4 *maxY,
                        __global uchar4 *dstptr) {
  int col = get_global_id(0);

  uchar4 max_val = (uchar4)(50);
  uchar4 max_cnt = (uchar4)(0);
  const uchar4 one = (uchar4)(1);
  const uchar4 zero = (uchar4)(0);
  char4 res;
  char4 tmp_res = (char4)(0);
  char4 last_res = (char4)(0);
  int4 minus = (int4)(-1);
  int idx;
  // uchar4 = dst;

  // find maximum for the kernel
  for (int row = 0; row < rows; row++) {
    idx = mad24(row, srcDstStep, col);
    res = srcptr[idx] > max_val;
    max_val = select(max_val, srcptr[idx], res);
  }

  for (int row = rows - 1; row >= 0; row--) {
    idx = mad24(row, srcDstStep, col);
    res = (srcptr[idx] == max_val) | last_res;
    tmp_res = res - last_res;
    // max_cnt += select(zero, one, res);
    if (any(tmp_res)) {
      dstptr[idx] = convert_uchar4(tmp_res);
      maxY[col] = select(minus, (int4)(row), convert_int4(tmp_res));
    }
    last_res = last_res | res;
  }

  // barrier(CLK_LOCAL_MEM_FENCE);

  // maxVal[col] = max_val;
  // maxCount[col] = max_cnt;
}

__kernel void isolate(__global uchar *srcptr, __global uchar *dstptr,
                      __global int *maxYsrc, __global int *maxYdst, int step,
                      int radius) {
  int x = get_global_id(0);
  int y = maxYsrc[x];
  int n = 0;
  int id;

  for (int xx = x - radius; xx <= x + radius; xx++) {
    for (int yy = y - radius; yy <= y + radius; yy++) {
      id = mad24(yy, step, xx);
      if (srcptr[id])
        n++;
    }
  }
  if (n >= radius + 1) {
    maxYdst[x] = y;
    id = mad24(y, step, x);
    dstptr[id] = 0xff;
  }
}

// __kernel void __attribute__((reqd_work_group_size(32, 30, 1)))
__kernel void gaussian(__read_only image2d_t srcImg,
                       __write_only image2d_t dstImg, __constant float *filter,
                       //  __constant int2 *size, sampler_t sampler) {
                       int hW, int hH, sampler_t sampler) {
  int column = get_global_id(0);
  int row = get_global_id(1);

  // int hW = halfWidth;
  // int hH = halfHeight;
  // const int2 half_size = size[0];

  float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
  int filterIdx = 0;
  int2 coords;

  for (int i = -hH; i <= hH; i++) {
    coords.y = row + i;
    for (int j = -hW; j <= hW; j++) {
      coords.x = column + j;

      float4 pixel;
      pixel = read_imagef(srcImg, sampler, coords);
      sum.x += pixel.x * filter[filterIdx++];
    }
  }

  coords.x = column;
  coords.y = row;
  write_imagef(dstImg, coords, sum);
}

__kernel void gaussian51(__read_only image2d_t srcImg,
                         __write_only image2d_t dstImg, sampler_t sampler) {
  int column = get_global_id(0);
  int row = get_global_id(1);

  const float filter[5] = {0.054488685, 0.24420135, 0.40261996, 0.24420135,
                           0.054488685};

  float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
  int filterIdx = 0;
  int2 coords;

  coords.x = column;
  for (int i = -2; i <= 2; i++) {
    coords.y = row + i;

    float4 pixel;
    pixel = read_imagef(srcImg, sampler, coords);
    sum.x += pixel.x * filter[filterIdx++];
  }

  coords.x = column;
  coords.y = row;
  write_imagef(dstImg, coords, sum);
}

__kernel void gaussian51buf(__global uchar4 *pSrc, __global uchar4 *pDst,
                            int srcDstStep, int rows, int cols) {
  // int x = get_global_id(0);
  // int y = get_global_id(1);
  // int idx = mad24(y, srcDstStep, x);
  // pDst[idx] = ~pSrc[idx];

  int column = get_global_id(0);
  int row = get_global_id(1);
  int idx = mad24(row, srcDstStep, column);

  const float filter[5] = {0.054488685f, 0.24420135f, 0.40261996f, 0.24420135f,
                           0.054488685f};

  float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
  int filterIdx = 0;
  int2 coords;

  coords.x = column;
  // #pragma unroll
  for (int i = -2; i <= 2; i++) {
    coords.y = row + i;

    // float4 pixel;
    // pixel = read_imagef(srcImg, sampler, coords);
    int iner_id = mad24(coords.y, srcDstStep, coords.x);
    sum += convert_float4(pSrc[iner_id]) * filter[filterIdx++];
  }

  // coords.x = column;
  // coords.y = row;
  pDst[idx] = convert_uchar4(sum);
}