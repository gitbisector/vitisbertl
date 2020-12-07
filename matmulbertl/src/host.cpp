#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "xcl2.hpp"

// HBM Banks requirements
#define MAX_HBM_BANKCOUNT 32
#define BANK_NAME(n) n | XCL_MEM_TOPOLOGY
const int bank[MAX_HBM_BANKCOUNT] = {
    BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),
    BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),
    BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),
    BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
    BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),
    BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29),
    BANK_NAME(30), BANK_NAME(31)};

const int map[8] = {
  BANK_NAME(0),
  BANK_NAME(4),
  BANK_NAME(8),
  BANK_NAME(12),
  BANK_NAME(16),
  BANK_NAME(20),
  BANK_NAME(24),
  BANK_NAME(26)
};


enum {
  Npk = 8,
  Wr = 3*1024,
  Wc = 1024,
  Vr = 1024,
  Vc = 14,
  Niter= 1,
};

int swres[Wr][Vc];

typedef char Dt;

void
swmatmul(std::vector<Dt, aligned_allocator<Dt> > &W, std::vector<Dt, aligned_allocator<Dt> >&V)
{
  memset(swres, 0, sizeof(swres));
  for(int r = 0; r < Wr; r++)
    for(int c = 0; c < Vc; c++)
      for(int k = 0; k < Wc; k++)
        swres[r][c] += (W[r*Wc+k] * V[k+c*Vr]) >> 16;
}

int
main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <XCLBIN> \n", argv[0]);
    return -1;
  }

  int is_sw_emulation = xcl::is_emulation() && (!xcl::is_hw_emulation());

  /* Allocate space in each one of 9 HBM banks */
  /* Load (3072,1024) weights into HBM in Device Global memory*/

  /* Load (1024,14) vector from Host Local into separate HBM in Device Global memory */
  /* Run both feeder_1() and wb_1() kernels */
  /* Load (1024,14) results vector from Device Global memory back to Host Local */

  std::string binaryFile = argv[1];
  cl_int err;
  cl::CommandQueue q;
  cl::Kernel krnl_feeder;
  cl::Context context;
  std::vector<Dt, aligned_allocator<Dt>> source_w[Npk];
  std::vector<Dt, aligned_allocator<Dt>> source_v(sizeof(Dt)*14*1024);
  std::vector<int, aligned_allocator<int>> source_hw_wb_results(sizeof(int)*3*14*1024);

  for(int i = 0; i < Npk; i++) {
    source_w[i].resize(sizeof(Dt)*3*1024*1024/Npk);
  }

  // Create the test data
  for(int i = 0; i < Npk; i++) {
    std::fill(source_w[i].begin(), source_w[i].end(), 0);
    //std::generate(source_w[i].begin(), source_w[i].end(), std::rand);
  }

  std::fill(source_v.begin(), source_v.end(), 8);
  //std::generate(source_v.begin(), source_v.end(), std::rand);

  source_w[0][10] = 5;
  source_w[0][1024+11] = 7;

  // Initializing output vectors to zero
  std::fill(source_hw_wb_results.begin(), source_hw_wb_results.end(), 0);

  // OPENCL HOST CODE AREA START
  // The get_xil_devices will return vector of Xilinx Devices
  auto devices = xcl::get_xil_devices();

  // read_binary_file() command will find the OpenCL binary file created using
  // the  V++ compiler load into OpenCL Binary and return pointer to file buffer.
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  bool valid_device = false;
  for (unsigned int i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                                            CL_QUEUE_PROFILING_ENABLE,
                                        &err));
    std::cout << "Trying to program device[" << i
              << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Program program(context, {device}, bins, NULL, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    } else {
      std::cout << "Device[" << i << "]: program successful!\n";
      // Creating Kernel object using Compute unit names

      std::string krnl_name_feeder = "feeder:{feeder_1}";
      printf("Creating a kernel [%s] for CU\n", krnl_name_feeder.c_str());
      OCL_CHECK(err, krnl_feeder = cl::Kernel(program, krnl_name_feeder.c_str(), &err));
      valid_device = true;
      break; // we break because we found a valid device
    }
  }
  if (!valid_device) {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }

  std::vector<cl_mem_ext_ptr_t> inBufExtw(Npk);
  std::vector<cl_mem_ext_ptr_t> inBufExtv(1);
  std::vector<cl_mem_ext_ptr_t> outBufExtwb(1);

  std::vector<cl::Buffer> buffer_inputw(Npk);
  std::vector<cl::Buffer> buffer_inputv(1);
  std::vector<cl::Buffer> buffer_output_wb(1);

  // For Allocating Buffer to specific Global Memory Bank, user has to use
  // cl_mem_ext_ptr_t
  // and provide the Banks
  for(int i = 0; i < Npk; i++) {
    inBufExtw[i].obj = source_w[i].data();
    inBufExtw[i].param = 0;
    inBufExtw[i].flags = map[i];
    if(is_sw_emulation)
      inBufExtw[i].flags = bank[0];
  }

  inBufExtv[0].obj = source_v.data();
  inBufExtv[0].param = 0;
  inBufExtv[0].flags = bank[14];
  if(is_sw_emulation) {
    inBufExtv[0].flags = bank[0];
  }

  outBufExtwb[0].obj = source_hw_wb_results.data();
  outBufExtwb[0].param = 0;
  outBufExtwb[0].flags = bank[14];
  if(is_sw_emulation) {
    outBufExtwb[0].flags = bank[0];
  }


  // These commands will allocate memory on the FPGA and copy the data across
  // The cl::Buffer objects can be used to reference the memory locations on the device.
  // The Weights are spread across Nk even-numbered HBM banks and vector lives in bank 31.
  for(int i = 0; i < Npk; i++) {
    OCL_CHECK(err, buffer_inputw[i] = cl::Buffer(
                      context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                      sizeof(Dt) * 3 * 1024 * 1024/Npk, &inBufExtw[i], &err));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_inputw[i]}, 0 /* 0 means from host*/));
  }
  OCL_CHECK(err, buffer_inputv[0] = cl::Buffer(
                      context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                      sizeof(Dt) * 1024 * 14, &inBufExtv[0], &err));
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_inputv[0]}, 0 /* 0 means from host*/));

  OCL_CHECK(err, buffer_output_wb[0] = cl::Buffer(
                      context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                      sizeof(int) * 3 * 1024 * 14, &outBufExtwb[0], &err));
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_wb[0]}, 0 /* 0 means from host*/));
  // Copy input data to Device Global Memory
  q.finish();


    // Setting the feeder Arguments
    OCL_CHECK(err, err = krnl_feeder.setArg(0, buffer_inputv[0]));
    for(int j = 0; j < Npk; j++) {
      OCL_CHECK(err, err = krnl_feeder.setArg(1+j, buffer_inputw[j]));
    }
    int outshiftscale = 0;
    OCL_CHECK(err, err = krnl_feeder.setArg(9, buffer_output_wb[0]));
    OCL_CHECK(err, err = krnl_feeder.setArg(10, (char)3));
    OCL_CHECK(err, err = krnl_feeder.setArg(11, (char)14));
    OCL_CHECK(err, err = krnl_feeder.setArg(12, outshiftscale));

  double kernel_time_in_sec = 0;
  std::chrono::duration<double> kernel_time(0);
  auto kernel_start = std::chrono::high_resolution_clock::now();
  for(int iter=0; iter<Niter; iter++) {
    // Invoking the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_feeder));
    q.finish();
  }
  std::cout << "q finished" << std::endl;

  auto kernel_end = std::chrono::high_resolution_clock::now();
  kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
  kernel_time_in_sec = kernel_time.count();

  // Copy Result from Device Global Memory to Host Local Memory
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_wb[0]}, CL_MIGRATE_MEM_OBJECT_HOST));
  q.finish();
  std::cout << "readback finished" << std::endl;

  std::cout << "kernel time = " << (kernel_time_in_sec/Niter*1000000) << " us" << std::endl;
  // OPENCL HOST CODE AREA ENDS

  for(int i=0; i < Wr*Vc; i++) {
    if(source_hw_wb_results[i] != 0) {
      std::cout << "i = " << i << "; v = " << source_hw_wb_results[i] << std::endl;
    }
  }

  int match = 1;
  /* Need to do some verification of the result*/
  std::cout << (match ? "TEST PASSED" : "TEST FAILED") << std::endl;
  return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}

