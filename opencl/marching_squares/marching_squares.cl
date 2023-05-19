#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define HP5_WARP_SIZE 32

__kernel void ms_trace(
    __global float4 *g_seg,
    __global int *g_id_head,
    __global int *g_id_tail,
    // __local float4 *l_seg,
    __global int *g_res,
    uint total)
{
    int i, j, h, t;
    const uint l_id = get_local_id(0);
    const uint g_id = get_group_id(0);
    const uint l_size = get_local_size(0);
    const uint id = l_id + g_id * l_size;
    h = -1;
    t = -1;

    __local float4 l_seg[1024 + 160*2];

    if (id < total)
    {
        for (i = 1; i <= 160; i++)
        {
            j = id + i;
            if (j < total)
            {
                if (all(g_seg[id].s23 == g_seg[j].s01))
                {
                    t = j;
                    break;
                }
            }
            j = id - i;
            if (j >= 0)
            {
                if (all(g_seg[id].s23 == g_seg[j].s01))
                {
                    t = j;
                    break;
                }
            }
        }
        g_id_head[t] = id;
        g_id_tail[id] = t;
    }
}

__kernel void ms_find_head(
    __global int *g_id_head,
    __global int *g_id_tail,
    __global int *g_line_len,
    __global int *g_line_ptr,
    __global int *g_line_id,
    __global int *g_res,
    uint total)
{
    uint i, j, k, val, t_id;
    val = 0;

    // const uint g_size = get_global_size(0);
    // const uint g_id = get_global_id(0);
    const uint l_size = get_local_size(0);
    const uint l_id = get_local_id(0);
    const uint g_id = get_group_id(0);
    // i = 0;
    // while(1)
    // {
    //     id = g_id+g_size*i;
    const uint id = l_id + l_size * g_id;
    if (id >= total)
        return;
    // if(id>=total) break;
    if (g_id_head[id] == -1)
    {
        t_id = id;
        for (k = 0; g_id_tail[t_id] != -1; t_id = g_id_tail[t_id])
        {
            k++;
            g_id_head[t_id] = -2;
        }
        g_id_head[t_id] = -2;
        // Points are one more than polyline segments.
        k += 2;
        j = atomic_inc(&g_res[2]);
        // while(g_line_ptr[j] == -1)
        // {
        //     val++;
        // }
        // g_line_ptr[j+1] = g_line_ptr[j] + k;
        g_line_id[j] = id;
        g_line_len[j] = k;
    }
    //     i++;
    // }
}

__kernel void ms_find_circle(
    __global int *g_id_head,
    __global int *g_id_tail,
    __global int *g_line_len,
    __global int *g_line_ptr,
    __global int *g_line_id,
    __global int *g_res,
    uint total)
{
    uint i, j, k, val, t_id, h_id;
    val = 0;

    // const uint g_size = get_global_size(0);
    // const uint g_id = get_global_id(0);
    const uint l_size = get_local_size(0);
    const uint l_id = get_local_id(0);
    const uint g_id = get_group_id(0);
    // i = 0;
    // while(1)
    // {
    //     id = g_id+g_size*i;
    const uint id = l_id + l_size * g_id;
    if (id >= total)
        return;
    // if(id>=total) break;
    if (g_id_head[id] != -2)
    {
        t_id = id;
        h_id = id;
        k = 0;
        while (g_id_tail[t_id] > id && g_id_head[h_id] > id)
        {
            k++;
            t_id = g_id_tail[t_id];
            h_id = g_id_head[h_id];
        }

        if (g_id_tail[t_id] == id)
        {
            // Points are one more than polyline segments.
            k += 2;
            j = atomic_inc(&g_res[2]);
            // while(g_line_ptr[j] == -1)
            // {
            //     val++;
            // }
            // g_line_ptr[j+1] = g_line_ptr[j] + k;
            g_line_id[j] = id;
            g_line_len[j] = k;
        }
    }
    //     i++;
    // }
}

// __kernel void ms_sum(
//     __local int *l_line_len,
//     __local int *l_line_ptr,
//     __global int* g_line_len,
//     __global int* g_line_ptr,
//     __global int *g_res)
// {
//     uint t_id, h_id, l_id, g_id, id, l_size, g_size;
//     int i, j, k, val;
//     g_id = get_global_id(0);
//     val = g_res[2];
//     if(g_id < val)
//         l_line_len[g_id] = g_line_len[g_id];
//     barrier(CLK_GLOBAL_MEM_FENCE);

//     if(g_id == 0)
//     {
//         l_line_ptr[0] = 0;
//         for(i=1; i<val; i++)
//         {
//             l_line_ptr[i] = l_line_len[i-1] + l_line_ptr[i-1];
//         }
//     }
//     barrier(CLK_GLOBAL_MEM_FENCE);
//     if(g_id < val)
//         g_line_ptr[g_id] = l_line_ptr[g_id];
// }

// __kernel void ms_restore(
//     __global float4* restrict g_seg,
//     __global float2* restrict g_polyline,
//     __global int* restrict g_index_t,
//     __global int* g_line_len,
//     __global int* g_line_ptr,
//     __global int* g_line_id,
//     __global int *g_res)
// {
//     uint t_id, h_id, l_id, g_id, id, l_size, g_size;
//     int i, ptr, len, total;
//     g_id = get_global_id(0);
//     total = g_res[2];

//     if(g_id < total)
//     {
//         id = g_line_id[g_id];
//         ptr = g_line_ptr[g_id];
//         len = g_line_len[g_id];
//         g_polyline[ptr] = g_seg[id].lo;
//         for(i=1; i<len; i++)
//         {
//             // g_polyline[ptr+i] = g_seg[id].hi;
//             id = g_index_t[id];
//         }
//     }
// }

__kernel void reduceApex(
    const uint i,
    const uint s,
    const uint L,
    __global const uint *levels,
    __global const uint *offsets,
    __global uint4 *restrict hp_g,
    __global uint *restrict sb_g,
    __global int *g_res)
{
    // total address space:
    // 0 : sum + 3 padding
    // 1 : 1 uvec4 of level 0.
    // 2 : 5 values of level 0 (top)
    // 7 : 25 values of level 1
    // 32: total sum.
    uint sb0_d, n0;
    sb0_d = offsets[(L - 1) ? (L + 1 + (s ? 0 : 1)) : 0];
    n0 = levels[L - 1];
    const uint threadIdx = get_local_id(0);
    const uint tid5 = 5 * threadIdx;

    // Fetch up to 125 elements from sb0_d.
    // a = threadIdx < n0 ? sb_g[sb0_d + threadIdx] : 0;

    __local uint sb[125 + 25];
    sb[threadIdx] = threadIdx < n0 ? sb_g[sb0_d + threadIdx] : 0;

    // Store 5x5 uint4's at uint4 offset 0 (25x4=100 elements, corresponding to 125 inputs).
    barrier(CLK_LOCAL_MEM_FENCE);
    if (threadIdx < 25)
    {
        uint e0 = sb[tid5 + 0];
        uint e1 = sb[tid5 + 1];
        uint e2 = sb[tid5 + 2];
        uint e3 = sb[tid5 + 3];
        uint e4 = sb[tid5 + 4];
        hp_g[7 + threadIdx] = (uint4)(e0,
                                      e0 + e1,
                                      e0 + e1 + e2,
                                      e0 + e1 + e2 + e3);

        sb[125 + threadIdx] = e0 + e1 + e2 + e3 + e4;
    }

    // Store 5 uint4's at uint4 offset 25 (5x4=20 elements, corresponding to 25 inputs).
    barrier(CLK_LOCAL_MEM_FENCE);
    if (threadIdx < 5)
    {
        uint e0 = sb[125 + tid5 + 0];
        uint e1 = sb[125 + tid5 + 1];
        uint e2 = sb[125 + tid5 + 2];
        uint e3 = sb[125 + tid5 + 3];
        uint e4 = sb[125 + tid5 + 4];
        hp_g[2 + threadIdx] = (uint4)(e0,
                                      e0 + e1,
                                      e0 + e1 + e2,
                                      e0 + e1 + e2 + e3);

        sb[threadIdx] = e0 + e1 + e2 + e3 + e4;
    }

    // Store 1 uint4 at uint4 offset 30 (1x4=4 elements, corresponding to 5 inputs)
    // Store total at uint4 offset 31
    barrier(CLK_LOCAL_MEM_FENCE);
    if (threadIdx < 1)
    {
        uint e0 = sb[0];
        uint e1 = sb[1];
        uint e2 = sb[2];
        uint e3 = sb[3];
        uint e4 = sb[4];
        hp_g[1 + threadIdx] = (uint4)(e0,
                                      e0 + e1,
                                      e0 + e1 + e2,
                                      e0 + e1 + e2 + e3);
        uint s = e0 + e1 + e2 + e3 + e4;
        hp_g[0] = (uint4)(s, 0, 0, 0);
        g_res[5] = s;
    }
}

__kernel void reduce1(
    const uint i,
    const uint s,
    const uint L,
    __global const uint *levels,
    __global const uint *offsets,
    __global uint4 *restrict hp_g,
    __global uint *restrict sb_g,
    __global int *g_res)
{
    uint hp1_d, sb1_d, n1, sb0_d, n0;
    const uint threadIdx = get_local_id(0);
    const uint blockIdx = get_group_id(0);
    hp1_d = offsets[i];
    sb1_d = offsets[L + 1 + (s ? 1 : 0)];
    n1 = levels[i];
    sb0_d = offsets[(i - 1) ? (L + 1 + (s ? 0 : 1)) : 0];
    n0 = levels[i - 1];
    hp1_d /= 4;

    __local uint sb0[5 * HP5_WARP_SIZE];
    const uint offset0 = 5 * HP5_WARP_SIZE * blockIdx + threadIdx;
    const uint offset1 = HP5_WARP_SIZE * blockIdx + threadIdx;
    const uint tid5 = 5 * threadIdx;

    sb0[threadIdx] = offset0 < n0 ? sb_g[sb0_d + offset0] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (threadIdx < HP5_WARP_SIZE && offset1 < n1)
    {
        uint4 hp = (uint4)(sb0[tid5 + 0], sb0[tid5 + 1], sb0[tid5 + 2], sb0[tid5 + 3]);
        hp_g[hp1_d + offset1] = hp;
        sb_g[sb1_d + offset1] = hp.x + hp.y + hp.z + hp.w + sb0[tid5 + 4];
    }
}

__kernel void reduce2(
    const uint i,
    const uint s,
    const uint L,
    __global const uint *levels,
    __global const uint *offsets,
    __global uint4 *restrict hp_g,
    __global uint *restrict sb_g,
    __global int *g_res)
{
    uint hp2_d, sb2_d, n2, hp1_d, n1, sb0_d, n0;
    const uint threadIdx = get_local_id(0);
    const uint blockIdx = get_group_id(0);
    hp2_d = offsets[i + 1];
    sb2_d = offsets[L + 1 + (s ? 1 : 0)];
    n2 = levels[i + 1];
    hp1_d = offsets[i];
    n1 = levels[i];
    sb0_d = offsets[(i - 1) ? (L + 1 + (s ? 0 : 1)) : 0];
    n0 = levels[i - 1];
    hp2_d /= 4;
    hp1_d /= 4;

    __local uint sb0[5 * 5 * HP5_WARP_SIZE];
    __local uint sb1[5 * HP5_WARP_SIZE];
    const uint offset0 = 5 * 5 * HP5_WARP_SIZE * blockIdx;
    const uint offset1 = 5 * HP5_WARP_SIZE * blockIdx + threadIdx;
    const uint offset2 = HP5_WARP_SIZE * blockIdx + threadIdx;
    const uint tid5 = 5 * threadIdx;

    uint o = threadIdx;
    for (uint i = 0; i < 5; i++)
    {
        uint q = offset0 + o;
        sb0[o] = q < n0 ? sb_g[sb0_d + q] : 0;
        o += 5 * HP5_WARP_SIZE;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint4 hp = (uint4)(sb0[tid5 + 0], sb0[tid5 + 1], sb0[tid5 + 2], sb0[tid5 + 3]);
    if (offset1 < n1)
    {
        hp_g[hp1_d + offset1] = hp;
    }
    sb1[threadIdx] = hp.x + hp.y + hp.z + hp.w + sb0[tid5 + 4];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (threadIdx < HP5_WARP_SIZE && offset2 < n2)
    {
        uint4 hp = (uint4)(sb1[tid5 + 0], sb1[tid5 + 1], sb1[tid5 + 2], sb1[tid5 + 3]);
        hp_g[hp2_d + offset2] = hp;
        sb_g[sb2_d + offset2] = hp.x + hp.y + hp.z + hp.w + sb1[tid5 + 4];
    }
}

__kernel void reduce3(
    const uint i,
    const uint s,
    const uint L,
    __global const uint *levels,
    __global const uint *offsets,
    __global uint4 *restrict hp_g,
    __global uint *restrict sb_g,
    __global int *g_res)
{
    uint hp3_d, sb3_d, n3, hp2_d, n2, hp1_d, n1, sb0_d, n0;
    const uint threadIdx = get_local_id(0);
    const uint blockIdx = get_group_id(0);
    hp3_d = offsets[i + 2];
    sb3_d = offsets[L + 1 + (s ? 1 : 0)];
    n3 = levels[i + 2];
    hp2_d = offsets[i + 1];
    n2 = levels[i + 1];
    hp1_d = offsets[i];
    n1 = levels[i];
    sb0_d = offsets[(i - 1) ? (L + 1 + (s ? 0 : 1)) : 0];
    n0 = levels[i - 1];
    hp3_d /= 4;
    hp2_d /= 4;
    hp1_d /= 4;

    __local uint sb0[5 * 5 * 5 * HP5_WARP_SIZE];
    __local uint sb1[5 * 5 * HP5_WARP_SIZE];
    __local uint sb2[5 * HP5_WARP_SIZE];
    uint offset0 = 5 * 5 * 5 * HP5_WARP_SIZE * blockIdx;
    uint offset1 = 5 * 5 * HP5_WARP_SIZE * blockIdx;
    uint offset2 = 5 * HP5_WARP_SIZE * blockIdx + threadIdx;
    uint offset3 = HP5_WARP_SIZE * blockIdx + threadIdx;
    const uint tid5 = 5 * threadIdx;

    uint o1 = threadIdx;
    for (uint k = 0; k < 5; k++)
    {
        uint o0 = threadIdx;
        for (uint i = 0; i < 5; i++)
        {
            uint q = offset0 + o0;
            sb0[o0] = q < n0 ? sb_g[sb0_d + q] : 0;
            o0 += 5 * HP5_WARP_SIZE;
        }
        offset0 += 5 * 5 * HP5_WARP_SIZE;
        barrier(CLK_LOCAL_MEM_FENCE);

        uint4 hp = (uint4)(sb0[tid5 + 0], sb0[tid5 + 1], sb0[tid5 + 2], sb0[tid5 + 3]);
        uint q1 = offset1 + o1;
        if (q1 < n1)
        {
            hp_g[hp1_d + q1] = hp;
        }
        sb1[o1] = hp.x + hp.y + hp.z + hp.w + sb0[tid5 + 4];

        o1 += 5 * HP5_WARP_SIZE;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write 5 x 32 x 4 uints to hp2, 5 x 32 uints to sb3
    uint4 hp = (uint4)(sb1[tid5 + 0], sb1[tid5 + 1], sb1[tid5 + 2], sb1[tid5 + 3]);
    if (offset2 < n2)
    {
        hp_g[hp2_d + offset2] = hp;
    }
    sb2[threadIdx] = hp.x + hp.y + hp.z + hp.w + sb1[tid5 + 4];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (threadIdx < HP5_WARP_SIZE && offset3 < n3)
    {
        uint4 hp = (uint4)(sb2[tid5 + 0], sb2[tid5 + 1], sb2[tid5 + 2], sb2[tid5 + 3]);
        hp_g[hp3_d + offset3] = hp;
        sb_g[sb3_d + offset3] = hp.x + hp.y + hp.z + hp.w + sb2[tid5 + 4];
    }
}

inline uint processHistoElement(uint *key, uint offset, const uint4 element)
{
    if (key[0] < element.x)
    {
    }
    else if (key[0] < element.y)
    {
        key[0] -= element.x;
        offset += 1;
    }
    else if (key[0] < element.z)
    {
        key[0] -= element.y;
        offset += 2;
    }
    else if (key[0] < element.w)
    {
        key[0] -= element.z;
        offset += 3;
    }
    else
    {
        key[0] -= element.w;
        offset += 4;
    }
    return offset;
}

inline uint processDataElement(uint *key, uint offset, const uint4 element)
{
    if (element.x <= key[0])
    {
        key[0] -= element.x;
        offset++;
        if (element.y <= key[0])
        {
            key[0] -= element.y;
            offset++;
            if (element.z <= key[0])
            {
                key[0] -= element.z;
                offset++;
                if (element.w <= key[0])
                {
                    key[0] -= element.w;
                    offset++;
                }
            }
        }
    }
    return offset;
}


__kernel void extract(
    const uint L,
    __global const uint *offsets,
    __global uint4 *restrict hp_g,
    __global int *g_line_ptr,
    __global int *g_line_id,
    __global float2 *restrict g_line,
    __global float4 *restrict g_seg,
    __global int *restrict g_id_tail,
    __global int *g_res)
{
    const uint blockDim = get_local_size(0);
    const uint threadIdx = get_local_id(0);
    const uint blockIdx = get_group_id(0);
    const uint index = blockDim * blockIdx + threadIdx;

    uint N = hp_g[0].x;

    if (index < N)
    {
        uint offset = 0;
        uint key = index;
        offset = processHistoElement(&key, 5 * offset, hp_g[1 + offset]);
        offset = processHistoElement(&key, 5 * offset, hp_g[2 + offset]);
        offset = processHistoElement(&key, 5 * offset, hp_g[7 + offset]);
        for (uint i = L; 1 < i; i--)
        {
            offset = processDataElement(&key, 5 * offset, hp_g[offsets[i - 1] / 4 + offset]);
        }


        int id = g_line_id[offset];
        if(key == 0)
        {
            g_line_ptr[offset] = index;
            g_line[index] = g_seg[id].lo;
        }
        else
        {
            uint i = 1;
            while(i<key)
            {
                id = g_id_tail[id];
                i++;
            }
            g_line[index] = g_seg[id].hi;
        }
    }
}