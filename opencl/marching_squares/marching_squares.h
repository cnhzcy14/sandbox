#pragma once
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif


//------------------------------------------------------------------------------
//  functions from ../Common
//------------------------------------------------------------------------------
extern int    output_device_info(cl_device_id );
extern double wtime();   // returns time since some fixed past point (wtime.c)

//------------------------------------------------------------------------------
//  File and kernels
//------------------------------------------------------------------------------
#define PROGRAM_FILE "/home/cnhzcy14/work/project/test_code/opencl/marching_squares/marching_squares.cl"
#define MS_TRACE "ms_trace"

//------------------------------------------------------------------------------
//  Constants
//------------------------------------------------------------------------------
#define NUM         700         // Total number of polylines
#define SIZE        640*480*4   // Total number of line segments
#define MAX_LEVEL   12
