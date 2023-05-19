#include "local_thresh.h"
#include "err_code.h"
#include "device_picker.h"

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

double wtime()
{
#ifdef _OPENMP
    /* Use omp_get_wtime() if we can */
    return omp_get_wtime();
#else
    /* Use a generic timer */
    static int sec = -1;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    if (sec < 0)
        sec = tv.tv_sec;
    return (tv.tv_sec - sec) + 1.0e-6 * tv.tv_usec;
#endif
}

char *getKernelSource(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error: Could not open kernel source file\n");
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    int len = ftell(file) + 1;
    rewind(file);

    char *source = (char *)calloc(sizeof(char), len);
    if (!source)
    {
        fprintf(stderr, "Error: Could not allocate memory for source string\n");
        exit(EXIT_FAILURE);
    }
    fread(source, sizeof(char), len, file);
    fclose(file);
    return source;
}

/* Create program from a file and compile it */
cl_program buildProgram(cl_context ctx, cl_device_id dev, const char *filename)
{
    cl_int err;
    cl_program program;
    char *kernelsource;
    kernelsource = getKernelSource(filename);
    // Create the comput program from the source buffer
    program = clCreateProgramWithSource(ctx, 1, (const char **)&kernelsource, NULL, &err);
    checkError(err, "Creating program with C_elem.cl");
    free(kernelsource);
    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    return program;
}

unsigned int roundUpper(unsigned int dividend, unsigned int divisor)
{
    return (dividend + (divisor - 1)) / divisor;
}

int main(int argc, char *argv[])
{
    /* OpenCL data structures */
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel k_lt_mean;
    cl_int err;
    size_t global_size, local_size, max_local_size, cu_num, local_seg_size, group_size, group_num;
    cl_ulong local_mem;

    double start_time; // Starting time
    double run_time;   // timing data

    /* Data and buffers */
    cv::namedWindow("window", cv::WINDOW_NORMAL);
    cv::Mat h_src, h_dst;
    cl_mem d_src, d_dst;
    cl_image_format img_format;
    cl_image_desc img_desc;

    /* Initialize data */
    h_src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (argc != 2 || !h_src.data)
    {
        printf("No image data \n");
        return -1;
    }
    h_dst = cv::Mat(h_src.rows, h_src.cols, CV_8UC1, cv::Scalar::all(0));
    img_format.image_channel_order = CL_LUMINANCE;
    img_format.image_channel_data_type = CL_UNSIGNED_INT8;
    img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    img_desc.image_width = h_src.cols;
    img_desc.image_height = h_src.rows;
    img_desc.image_depth = 0;
    img_desc.image_array_size = 0;
    img_desc.image_row_pitch = h_src.step[0];
    img_desc.image_slice_pitch = 0;
    img_desc.num_mip_levels = 0;
    img_desc.num_samples = 0;
    img_desc.buffer = 0;

    //--------------------------------------------------------------------------------
    // Create a context, queue and device.
    //--------------------------------------------------------------------------------
    cl_uint deviceIndex = 0;
    parseArguments(argc, argv, &deviceIndex);

    // Get list of devices
    cl_device_id devices[MAX_DEVICES];
    unsigned numDevices = getDeviceList(devices);

    // Check device index in range
    if (deviceIndex >= numDevices)
    {
        printf("Invalid device index (try '--list')\n");
        return EXIT_FAILURE;
    }

    device = devices[deviceIndex];

    char name[MAX_INFO_STRING];
    getDeviceName(device, name);
    printf("\nUsing OpenCL device: %s\n", name);

    // Create a compute context
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Creating command queue");

    /* Determine maximum work-group size */
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu_num), &cu_num, NULL);
    err |= clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_local_size), &max_local_size, NULL);
    err |= clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, NULL);
    checkError(err, "Obtain device information");

    // local_size = max_local_size;
    local_size = 1024;
    global_size = h_src.cols * h_src.rows;
    printf("--------%zd, %zd\n", local_size, global_size);

    /* Create buffer */
    d_src = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          &img_format, &img_desc, h_src.ptr(), &err);
    checkError(err, "Creating buffer d_src");

    d_dst = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_format, &img_desc, h_dst.ptr(), &err);
    checkError(err, "Creating buffer d_dst");

    //--------------------------------------------------------------------------------
    // GPU implementation test
    //--------------------------------------------------------------------------------
    /* Build the program */
    program = buildProgram(context, device, PROGRAM_FILE);
    /* Create kernels */
    k_lt_mean = clCreateKernel(program, "lt_mean", &err);
    checkError(err, "Creating kernel with lt_mean");

    /* Set kernel argument */
    err = clSetKernelArg(k_lt_mean, 0, sizeof(cl_mem), &d_src);
    err |= clSetKernelArg(k_lt_mean, 1, sizeof(cl_mem), &d_dst);
    checkError(err, "Setting k_lt_mean args");

    /* Copy data to buffer */
    size_t Origin[] = {0, 0, 0};
    size_t Region[] = {(size_t)h_src.cols, (size_t)h_src.rows, 1};
    err = clEnqueueWriteImage(queue, d_src, CL_TRUE, Origin, Region, 0, 0, h_src.ptr(), 0, NULL, NULL);
    checkError(err, "Write data to buffer");

    /* Enqueue kernel */
    start_time = wtime();
    err = clEnqueueNDRangeKernel(queue, k_lt_mean, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    checkError(err, "Enqueueing k_ms_trace");

    err = clFinish(queue);
    checkError(err, "Waiting for kernel to finish");

    run_time = wtime() - start_time;
    printf("Total GPU impl time: %.6f ms \n", run_time * 1000);

    /* Read the result */
    err |= clEnqueueReadImage(queue, d_dst, CL_TRUE, Origin, Region, 0, 0, h_dst.ptr(), 0, NULL, NULL);
    checkError(err, "Read data from buffer");

    printf("===========================================\n");
    cv::imshow("window", h_dst);
    cv::waitKey(0);

    //--------------------------------------------------------------------------------
    // Clean up
    //--------------------------------------------------------------------------------
    clReleaseMemObject(d_src);
    clReleaseMemObject(d_dst);

    clReleaseKernel(k_lt_mean);
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}