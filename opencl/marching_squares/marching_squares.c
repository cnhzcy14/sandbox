#include "marching_squares.h"
#include "err_code.h"
#include "device_picker.h"

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

char *getKernelSource(char *filename)
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
cl_program buildProgram(cl_context ctx, cl_device_id dev, char *filename)
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

uint32_t scratchLayout(uint32_t *levels, uint32_t *offsets, uint32_t N)
{
  uint32_t level_size = 0;

  // if (125 < N)
  // {
  //   levels[0] = (N + 4) / 5;
  //   level_size++;

  //   for (int i = 0; i < MAX_LEVEL; i++)
  //   {
  //     if (125 > levels[i])
  //       break;
  //     levels[i + 1] = (levels[i] + 4) / 5;
  //     level_size++;
  //   }
  // }
  levels[0] = N;
  level_size++;
  if (125 < N)
  {
    for (int i = 0; i < MAX_LEVEL; i++)
    {
      if (125 > levels[i])
        break;
      levels[i + 1] = (levels[i] + 4) / 5;
      level_size++;
    }
  }

  uint32_t o = 0;
  offsets[level_size] = o; // Apex
  o += 32 * 4;

  if (125 < N)
  {
    for (int i = level_size - 1; 0 < i; i--)
    {
      offsets[i] = o; // HP level i
      o += 4 * levels[i];
    }
  }

  // level zero
  offsets[0] = o;
  o += (levels[0] + 3) & ~3;

  offsets[level_size + 1] = o; // Large sideband buffer
  o += (level_size == 0) ? 0 : (levels[1] + 3) & ~3;

  offsets[level_size + 2] = o; // Small sideband buffer
  o += level_size < 2 ? 0 : (levels[2] + 3) & ~3;

  offsets[level_size + 3] = o; // Final size
  return level_size;
}

void compact(uint32_t L, uint32_t *levels, uint32_t *offsets)
{
  printf("level_size: %d\n", L);
  for (int i = 0; i < L; i++)
    printf("levels[%d]: %d\n", i, levels[i]);
  for (int i = 0; i < L + 4; i++)
    printf("offsets[%d]: %d\n", i, offsets[i]);

  uint32_t sb = 0;
  uint32_t i = 1;
  // uint32_t stage = 0;

  for (; i + 2 < L; i += 3)
  {
    printf("reduce3<<<%d, 160>>>\n", (levels[i + 2] + 31) / 32);
    printf("uint4 hp3: offsets[%d]\n", i + 2);
    printf("      sb3: offsets[%d]\n", L + 1 + (sb ? 1 : 0));
    printf("       n3: L[%d]\n", i + 2);
    printf("uint4 hp2: offsets[%d]\n", i + 1);
    printf("       n2: L[%d]\n", i + 1);
    printf("uint4 hp1: offsets[%d]\n", i);
    printf("       n1: L[%d]\n", i);
    printf("      sb0: offsets[%d]\n", (i - 1) ? (L + 1 + (sb ? 0 : 1)) : 0);
    printf("       n0: L[%d]\n", i - 1);

    sb = ~sb;
  }

  for (; i + 1 < L; i += 2)
  {
    printf("reduce2<<<%d, 160>>>\n", (levels[i + 1] + 31) / 32);
    printf("uint4 hp2: offsets[%d]\n", i + 1);
    printf("      sb2: offsets[%d]\n", L + 1 + (sb ? 1 : 0));
    printf("       n2: L[%d]\n", i + 1);
    printf("uint4 hp1: offsets[%d]\n", i);
    printf("       n1: L[%d]\n", i);
    printf("      sb0: offsets[%d]\n", (i - 1) ? (L + 1 + (sb ? 0 : 1)) : 0);
    printf("       n0: L[%d]\n", i - 1);

    sb = ~sb;
  }

  for (; i < L; i++)
  {
    printf("reduce1<<<%d, 160>>>\n", (levels[i] + 31) / 32);
    printf("uint4 hp1: offsets[%d]\n", i);
    printf("      sb1: offsets[%d]\n", L + 1 + (sb ? 1 : 0));
    printf("       n1: L[%d]\n", i);
    printf("      sb0: offsets[%d]\n", (i - 1) ? (L + 1 + (sb ? 0 : 1)) : 0);
    printf("       n0: L[%d]\n", i - 1);

    sb = ~sb;
  }

  printf("reduceApex<<<1, 128>>>\n");
  printf("uint4 apex: offsets[%d]\n", L);
  printf("       sum: sum_d\n");
  printf("        in: offsets[%d]\n", (L - 1) ? (L + 1 + (sb ? 0 : 1)) : 0);
  printf("         N: L[%d]\n", L - 1);

  return;
}

int main(int argc, char *argv[])
{
  /* OpenCL data structures */
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel k_ms_trace, k_ms_find_head, k_ms_find_circle;
  // cl_kernel k_ms_sum, k_ms_restore;
  cl_kernel k_reduceApex, k_reduce1, k_reduce2, k_reduce3;
  cl_kernel k_extract;
  cl_int err;
  size_t global_size, local_size, max_local_size, cu_num, local_seg_size, group_size, group_num;
  cl_ulong local_mem;

  double start_time; // Starting time
  double run_time;   // timing data

  /* Data and buffers */
  float *h_seg, *h_line;
  int *h_ptr, *h_id, *h_head, *h_tail;
  uint32_t *h_hp5;
  int h_res[8] = {0};
  cl_mem d_len, d_seg, d_line, d_res, d_head, d_tail, d_ptr, d_id, d_hp5, d_levels, d_offsets;
  cl_uint total = 0; // Total line segments, 4 floats each line seg

  /* Initialize data */
  size_t size = SIZE * sizeof(float);
  h_seg = (float *)malloc(size);
  h_line = (float *)malloc(size);
  h_head = (int *)malloc(size / 4);
  h_tail = (int *)malloc(size / 4);
  // h_len = (int *)malloc(NUM * 4);
  h_ptr = (int *)malloc(NUM * 4);
  h_id = (int *)malloc(NUM * 4);

  for (int i = 0; i < size / 16; i++)
  {
    h_head[i] = -1;
    h_tail[i] = -1;
  }

  for (int i = 0; i < NUM; i++)
  {
    // h_len[i] = 0;
    h_ptr[i] = -1;
    h_id[i] = -1;
  }
  // h_len[0] = 0;
  h_ptr[0] = 0;

  uint32_t levels[MAX_LEVEL];
  uint32_t offsets[MAX_LEVEL + 4];
  uint32_t level_size;

  level_size = scratchLayout(levels, offsets, NUM);
  compact(level_size, levels, offsets);
  h_hp5 = (uint32_t *)malloc(offsets[level_size + 3] * 4);
  for (int i = 0; i < offsets[level_size + 3]; i++)
  {
    h_hp5[i] = 0;
  }

  // Read line segments from file
  FILE *fptr;
  fptr = fopen("lines_2.txt", "r");
  if (fptr == NULL)
  {
    printf("Error!! Cannot open file \n");
    return 1;
  }
  while (feof(fptr) == 0)
  {
    fscanf(fptr, "%f\n", &h_seg[total]);
    total++;
  }
  fclose(fptr);
  total /= 4;

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

  local_size = max_local_size;
  global_size = local_size * roundUpper(total, local_size);
  printf("--------%zd, %zd\n", local_size, group_num);

  /* Create buffer */
  d_seg = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, h_seg, &err);
  checkError(err, "Creating buffer d_seg");
  d_line = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, h_line, &err);
  checkError(err, "Creating buffer d_line");
  d_head = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size / 4, h_head, &err);
  checkError(err, "Creating buffer d_head");
  d_tail = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size / 4, h_tail, &err);
  checkError(err, "Creating buffer d_tail");
  d_res = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 8 * sizeof(int), h_res, &err);
  checkError(err, "Creating buffer d_res");

  d_ptr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NUM * sizeof(int), h_ptr, &err);
  checkError(err, "Creating buffer d_ptr");
  d_id = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NUM * sizeof(int), h_id, &err);
  checkError(err, "Creating buffer d_id");
  d_hp5 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, offsets[level_size + 3] * sizeof(int), h_hp5, &err);
  checkError(err, "Creating buffer d_hp5");
  d_levels = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, MAX_LEVEL * sizeof(int), levels, &err);
  checkError(err, "Creating buffer d_levels");
  d_offsets = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (MAX_LEVEL + 4) * sizeof(int), offsets, &err);
  checkError(err, "Creating buffer d_offsets");
  cl_buffer_region region;
  region.origin = offsets[0] * sizeof(int);
  region.size = levels[0] * sizeof(int);
  d_len = clCreateSubBuffer(d_hp5, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
  checkError(err, "Creating sub buffer d_len");

  //--------------------------------------------------------------------------------
  // GPU implementation test
  //--------------------------------------------------------------------------------
  /* Build the program */
  program = buildProgram(context, device, PROGRAM_FILE);
  /* Create kernels */
  k_ms_trace = clCreateKernel(program, "ms_trace", &err);
  checkError(err, "Creating kernel with ms_trace");
  k_ms_find_head = clCreateKernel(program, "ms_find_head", &err);
  checkError(err, "Creating kernel with ms_find_head");
  k_ms_find_circle = clCreateKernel(program, "ms_find_circle", &err);
  checkError(err, "Creating kernel with ms_find_circle");
  // k_ms_sum = clCreateKernel(program, "ms_sum", &err);
  // checkError(err, "Creating kernel with ms_sum");
  // k_ms_restore = clCreateKernel(program, "ms_restore", &err);
  // checkError(err, "Creating kernel with ms_restore");

  k_reduceApex = clCreateKernel(program, "reduceApex", &err);
  checkError(err, "Creating kernel with reduceApex");
  k_reduce1 = clCreateKernel(program, "reduce1", &err);
  checkError(err, "Creating kernel with reduce1");
  k_reduce2 = clCreateKernel(program, "reduce2", &err);
  checkError(err, "Creating kernel with reduce2");
  k_reduce3 = clCreateKernel(program, "reduce3", &err);
  checkError(err, "Creating kernel with reduce3");

  k_extract = clCreateKernel(program, "extract", &err);
  checkError(err, "Creating kernel with extract");

  /* Set kernel argument */
  err = clSetKernelArg(k_ms_trace, 0, sizeof(cl_mem), &d_seg);
  err |= clSetKernelArg(k_ms_trace, 1, sizeof(cl_mem), &d_head);
  err |= clSetKernelArg(k_ms_trace, 2, sizeof(cl_mem), &d_tail);
  err |= clSetKernelArg(k_ms_trace, 3, sizeof(cl_mem), &d_res);
  err |= clSetKernelArg(k_ms_trace, 4, sizeof(int), &total);
  checkError(err, "Setting k_ms_trace args");

  err = clSetKernelArg(k_ms_find_head, 0, sizeof(cl_mem), &d_head);
  err |= clSetKernelArg(k_ms_find_head, 1, sizeof(cl_mem), &d_tail);
  err |= clSetKernelArg(k_ms_find_head, 2, sizeof(cl_mem), &d_len);
  err |= clSetKernelArg(k_ms_find_head, 3, sizeof(cl_mem), &d_ptr);
  err |= clSetKernelArg(k_ms_find_head, 4, sizeof(cl_mem), &d_id);
  err |= clSetKernelArg(k_ms_find_head, 5, sizeof(cl_mem), &d_res);
  err |= clSetKernelArg(k_ms_find_head, 6, sizeof(int), &total);
  checkError(err, "Setting k_ms_find_head args");

  err = clSetKernelArg(k_ms_find_circle, 0, sizeof(cl_mem), &d_head);
  err |= clSetKernelArg(k_ms_find_circle, 1, sizeof(cl_mem), &d_tail);
  err |= clSetKernelArg(k_ms_find_circle, 2, sizeof(cl_mem), &d_len);
  err |= clSetKernelArg(k_ms_find_circle, 3, sizeof(cl_mem), &d_ptr);
  err |= clSetKernelArg(k_ms_find_circle, 4, sizeof(cl_mem), &d_id);
  err |= clSetKernelArg(k_ms_find_circle, 5, sizeof(cl_mem), &d_res);
  err |= clSetKernelArg(k_ms_find_circle, 6, sizeof(int), &total);
  checkError(err, "Setting k_ms_find_circle args");

  // err = clSetKernelArg(k_ms_sum, 0, NUM * sizeof(int), NULL);
  // err |= clSetKernelArg(k_ms_sum, 1, NUM * sizeof(int), NULL);
  // err |= clSetKernelArg(k_ms_sum, 2, sizeof(cl_mem), &d_len);
  // err |= clSetKernelArg(k_ms_sum, 3, sizeof(cl_mem), &d_ptr);
  // err |= clSetKernelArg(k_ms_sum, 4, sizeof(cl_mem), &d_res);
  // checkError(err, "Setting k_ms_sum args");

  // err = clSetKernelArg(k_ms_restore, 0, sizeof(cl_mem), &d_seg);
  // err |= clSetKernelArg(k_ms_restore, 1, sizeof(cl_mem), &d_line);
  // err |= clSetKernelArg(k_ms_restore, 2, sizeof(cl_mem), &d_tail);
  // err |= clSetKernelArg(k_ms_restore, 3, sizeof(cl_mem), &d_len);
  // err |= clSetKernelArg(k_ms_restore, 4, sizeof(cl_mem), &d_ptr);
  // err |= clSetKernelArg(k_ms_restore, 5, sizeof(cl_mem), &d_id);
  // err |= clSetKernelArg(k_ms_restore, 6, sizeof(cl_mem), &d_res);
  // checkError(err, "Setting k_ms_restore args");

  err = clSetKernelArg(k_reduceApex, 2, sizeof(int), &level_size);
  err |= clSetKernelArg(k_reduceApex, 3, sizeof(cl_mem), &d_levels);
  err |= clSetKernelArg(k_reduceApex, 4, sizeof(cl_mem), &d_offsets);
  err |= clSetKernelArg(k_reduceApex, 5, sizeof(cl_mem), &d_hp5);
  err |= clSetKernelArg(k_reduceApex, 6, sizeof(cl_mem), &d_hp5);
  err |= clSetKernelArg(k_reduceApex, 7, sizeof(cl_mem), &d_res);

  err |= clSetKernelArg(k_reduce1, 2, sizeof(int), &level_size);
  err |= clSetKernelArg(k_reduce1, 3, sizeof(cl_mem), &d_levels);
  err |= clSetKernelArg(k_reduce1, 4, sizeof(cl_mem), &d_offsets);
  err |= clSetKernelArg(k_reduce1, 5, sizeof(cl_mem), &d_hp5);
  err |= clSetKernelArg(k_reduce1, 6, sizeof(cl_mem), &d_hp5);
  err |= clSetKernelArg(k_reduce1, 7, sizeof(cl_mem), &d_res);

  err |= clSetKernelArg(k_reduce2, 2, sizeof(int), &level_size);
  err |= clSetKernelArg(k_reduce2, 3, sizeof(cl_mem), &d_levels);
  err |= clSetKernelArg(k_reduce2, 4, sizeof(cl_mem), &d_offsets);
  err |= clSetKernelArg(k_reduce2, 5, sizeof(cl_mem), &d_hp5);
  err |= clSetKernelArg(k_reduce2, 6, sizeof(cl_mem), &d_hp5);
  err |= clSetKernelArg(k_reduce2, 7, sizeof(cl_mem), &d_res);

  err |= clSetKernelArg(k_reduce3, 2, sizeof(int), &level_size);
  err |= clSetKernelArg(k_reduce3, 3, sizeof(cl_mem), &d_levels);
  err |= clSetKernelArg(k_reduce3, 4, sizeof(cl_mem), &d_offsets);
  err |= clSetKernelArg(k_reduce3, 5, sizeof(cl_mem), &d_hp5);
  err |= clSetKernelArg(k_reduce3, 6, sizeof(cl_mem), &d_hp5);
  err |= clSetKernelArg(k_reduce3, 7, sizeof(cl_mem), &d_res);
  checkError(err, "Setting k_reduce args");

  err = clSetKernelArg(k_extract, 0, sizeof(int), &level_size);
  err |= clSetKernelArg(k_extract, 1, sizeof(cl_mem), &d_offsets);
  err |= clSetKernelArg(k_extract, 2, sizeof(cl_mem), &d_hp5);
  err |= clSetKernelArg(k_extract, 3, sizeof(cl_mem), &d_ptr);
  err |= clSetKernelArg(k_extract, 4, sizeof(cl_mem), &d_id);
  err |= clSetKernelArg(k_extract, 5, sizeof(cl_mem), &d_line);
  err |= clSetKernelArg(k_extract, 6, sizeof(cl_mem), &d_seg);
  err |= clSetKernelArg(k_extract, 7, sizeof(cl_mem), &d_tail);
  err |= clSetKernelArg(k_extract, 8, sizeof(cl_mem), &d_res);
  checkError(err, "Setting k_extract args");

  /* Copy data to buffer */
  err = clEnqueueWriteBuffer(queue, d_seg, CL_TRUE, 0, size, h_seg, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_line, CL_TRUE, 0, size, h_line, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_head, CL_TRUE, 0, size / 4, h_head, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_tail, CL_TRUE, 0, size / 4, h_tail, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_ptr, CL_TRUE, 0, NUM * sizeof(int), h_ptr, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_id, CL_TRUE, 0, NUM * sizeof(int), h_id, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_res, CL_TRUE, 0, 8 * sizeof(int), h_res, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_hp5, CL_TRUE, 0, offsets[level_size + 3] * sizeof(int), h_hp5, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_levels, CL_TRUE, 0, MAX_LEVEL * sizeof(int), levels, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_offsets, CL_TRUE, 0, (MAX_LEVEL + 4) * sizeof(int), offsets, 0, NULL, NULL);
  checkError(err, "Write data to buffer");

  /* Enqueue kernel */
  start_time = wtime();
  err = clEnqueueNDRangeKernel(queue, k_ms_trace, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  checkError(err, "Enqueueing k_ms_trace");

  // k_ms_find_head workitems should run at the same time.
  // global_size = local_size * cu_num;
  err = clEnqueueNDRangeKernel(queue, k_ms_find_head, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  checkError(err, "Enqueueing k_ms_find_head");
  err = clEnqueueNDRangeKernel(queue, k_ms_find_circle, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  checkError(err, "Enqueueing k_ms_find_circle");

  {
    uint32_t s = 0;
    uint32_t i = 1;
    local_size = 160;

    for (; i + 2 < level_size; i += 3)
    {
      global_size = local_size * ((levels[i + 2] + 31) / 32);
      err = clSetKernelArg(k_reduce3, 0, sizeof(int), &i);
      err |= clSetKernelArg(k_reduce3, 1, sizeof(int), &s);
      checkError(err, "Write stage args");

      err = clEnqueueNDRangeKernel(queue, k_reduce3, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
      checkError(err, "Enqueueing k_reduce3");

      s = ~s;
    }

    for (; i + 1 < level_size; i += 2)
    {
      global_size = local_size * ((levels[i + 1] + 31) / 32);
      err = clSetKernelArg(k_reduce2, 0, sizeof(int), &i);
      err |= clSetKernelArg(k_reduce2, 1, sizeof(int), &s);
      checkError(err, "Write stage args");

      err = clEnqueueNDRangeKernel(queue, k_reduce2, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
      checkError(err, "Enqueueing k_reduce2");

      s = ~s;
    }

    for (; i < level_size; i++)
    {
      global_size = local_size * ((levels[i] + 31) / 32);
      err = clSetKernelArg(k_reduce1, 0, sizeof(int), &i);
      err |= clSetKernelArg(k_reduce1, 1, sizeof(int), &s);
      checkError(err, "Write stage args");

      err = clEnqueueNDRangeKernel(queue, k_reduce1, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
      checkError(err, "Enqueueing k_reduce1");

      s = ~s;
    }

    local_size = 128;
    global_size = 128;
    err = clSetKernelArg(k_reduceApex, 0, sizeof(int), &i);
    err |= clSetKernelArg(k_reduceApex, 1, sizeof(int), &s);
    checkError(err, "Write stage args");

    err = clEnqueueNDRangeKernel(queue, k_reduceApex, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    checkError(err, "Enqueueing k_reduceApex");
  }

  // err = clFinish(queue);
  // checkError(err, "Waiting for kernel to finish");

  // err = clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0, 8 * sizeof(int), h_res, 0, NULL, NULL);
  // checkError(err, "Read data from buffer");

  // level_size = scratchLayout(levels, offsets, h_res[2]);

  // global_size = local_size;
  // err = clEnqueueNDRangeKernel(queue, k_ms_sum, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  // if (err < 0)
  // {
  //   perror("Couldn't enqueue k_ms_sum");
  //   exit(1);
  // }

  // global_size = local_size;
  // err = clEnqueueNDRangeKernel(queue, k_ms_restore, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  // if (err < 0)
  // {
  //   perror("Couldn't enqueue k_ms_restore");
  //   exit(1);
  // }

  // err = clFinish(queue);
  // checkError(err, "Waiting for kernel to finish");

  // err = clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0, 8 * sizeof(int), h_res, 0, NULL, NULL);

  global_size = ((SIZE + max_local_size - 1) / max_local_size) * max_local_size;
  err = clEnqueueNDRangeKernel(queue, k_extract, 1, NULL, &global_size, &max_local_size, 0, NULL, NULL);
  checkError(err, "Enqueueing k_extract");

  err = clFinish(queue);
  checkError(err, "Waiting for kernel to finish");

  run_time = wtime() - start_time;
  printf("Total GPU impl time: %.2f ms \n", run_time * 1000);

  /* Read the result */
  err |= clEnqueueReadBuffer(queue, d_line, CL_TRUE, 0, size, h_line, 0, NULL, NULL);
  err |= clEnqueueReadBuffer(queue, d_head, CL_TRUE, 0, size / 4, h_head, 0, NULL, NULL);
  err |= clEnqueueReadBuffer(queue, d_tail, CL_TRUE, 0, size / 4, h_tail, 0, NULL, NULL);
  err |= clEnqueueReadBuffer(queue, d_hp5, CL_TRUE, 0, offsets[level_size + 3] * sizeof(int), h_hp5, 0, NULL, NULL);
  err |= clEnqueueReadBuffer(queue, d_ptr, CL_TRUE, 0, NUM * sizeof(int), h_ptr, 0, NULL, NULL);
  err |= clEnqueueReadBuffer(queue, d_id, CL_TRUE, 0, NUM * sizeof(int), h_id, 0, NULL, NULL);
  err |= clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0, 8 * sizeof(int), h_res, 0, NULL, NULL);
  checkError(err, "Read data from buffer");

  printf("===========================================\n");

  printf("res[0] head finish: %d\nres[1] tail finish: %d\nres[2] head size: %d\nres[3] tail size: %d\nres[4] global size: %d\nres[5] : %d\nres[6] line num: %d\nres[7] error code: %d\n",
         h_res[0], h_res[1], h_res[2], h_res[3], h_res[4], h_res[5], h_res[6], h_res[7]);
  // for(int i=0; i<30; i++)
  // {
  //     printf("polyLine %d: %d (%.2f, %.2f), (%.2f, %.2f) %d\n", i, index_h_[i], seg[i*4], seg[i*4+1], seg[i*4+2], seg[i*4+3], index_t_[i]);
  // }

  // for(int i=17130; i<17180; i++)
  // {
  //     printf("polyLine %d: %d (%.2f, %.2f), (%.2f, %.2f) %d\n", i, index_h_[i], seg[i*4], seg[i*4+1], seg[i*4+2], seg[i*4+3], index_t_[i]);
  // }
  // for (int i = 0; i < 40; i++)
  // {
  //   printf("polyLine %d: (%.2f, %.2f)\n", i, h_line[i * 2], h_line[i * 2 + 1]);
  // }

  //--------------------------------------------------------------------------------
  // Output validation
  //--------------------------------------------------------------------------------

  for (int i = 0; i < total; i++)
  {
    // if(index_h_[i] != index_h[i] || index_t_[i] != index_t[i])
    // {
    //     printf("%d: %d, %d, %d, %d\n", i, index_h_[i], index_h[i], index_t_[i], index_t[i]);
    //     une++;
    // }
    if (h_head[i] == -1)
    {
      printf("head: %d\n", i);
    }
  }

  int lineSum = 0;
  for (int i = 0; i < h_res[2]; i++)
  {
    lineSum += h_hp5[offsets[0] + i];
    int id = h_id[i];
    int len = h_hp5[offsets[0] + i];
    int ptr = h_ptr[i];
    // printf("new head ====> %d: %d, %d, %d\n", i, id, len, ptr);
    // for(int i=0; i<len; i++)
    // {
    //     printf("(%.1f, %.1f) ", polyLine[(ptr + i) * 2], polyLine[(ptr + i) * 2 + 1]);
    // }
    // printf("\n");
  }
  printf("\n====total items: %d\n", h_res[5] - h_res[2]);

  // int ptr = 0;
  // for (int num = 0; lineLen[num] != 0; num++)
  // {
  //     if (num > 0)
  //         ptr += lineLen[num - 1];
  //     for (int i = 0; i < lineLen[num]; i++)
  //     {
  //         // printf("(%.1f, %.1f) ", polyLine[(ptr + i) * 2], polyLine[(ptr + i) * 2 + 1]);
  //     }
  //     // printf("\n====================%d: %d\n", num, lineHeadPtr[num]);
  // }
  // printf("====total: %d\n", ptr);

  //--------------------------------------------------------------------------------
  // Clean up
  //--------------------------------------------------------------------------------
  free(h_seg);
  free(h_line);
  // free(h_len);
  free(h_ptr);
  free(h_id);
  free(h_head);
  free(h_tail);
  free(h_hp5);
  clReleaseMemObject(d_len);
  clReleaseMemObject(d_seg);
  clReleaseMemObject(d_line);
  clReleaseMemObject(d_ptr);
  clReleaseMemObject(d_id);
  clReleaseMemObject(d_res);
  clReleaseMemObject(d_head);
  clReleaseMemObject(d_tail);
  clReleaseMemObject(d_hp5);
  clReleaseMemObject(d_levels);
  clReleaseMemObject(d_offsets);

  clReleaseKernel(k_ms_trace);
  clReleaseKernel(k_ms_find_head);
  clReleaseKernel(k_ms_find_circle);
  // clReleaseKernel(k_ms_sum);
  // clReleaseKernel(k_ms_restore);
  clReleaseKernel(k_extract);
  clReleaseKernel(k_reduceApex);
  clReleaseKernel(k_reduce1);
  clReleaseKernel(k_reduce2);
  clReleaseKernel(k_reduce3);
  clReleaseProgram(program);
  clReleaseContext(context);

  return 0;
}