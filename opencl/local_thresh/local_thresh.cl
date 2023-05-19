#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE
   | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void lt_mean(
    read_only image2d_t src_image,
    write_only image2d_t dst_image)
{

}

