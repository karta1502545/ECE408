#include <wb.h>
#include <iostream>
using namespace std;

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// http://s3.amazonaws.com/files.rai-project.com/userdata/build-65160553f31975c9fdb59c23.tar.gz

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 6 // TODO: can be higher?

//@@ Define constant memory for device kernel here
// initialize constant memory for mask
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  // TODO: How to use constant memory in kernel function
  // TODO: Identify all boundaries that I may face
  //       z_size, y_size, x_size
  __shared__ float tile[TILE_WIDTH+MASK_WIDTH-1][TILE_WIDTH+MASK_WIDTH-1][TILE_WIDTH+MASK_WIDTH-1];

  int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  int hgh_o = blockIdx.z * TILE_WIDTH + tz;
  int row_o = blockIdx.y * TILE_WIDTH + ty;
  int col_o = blockIdx.x * TILE_WIDTH + tx;
  
  int offset = (MASK_WIDTH / 2);
  int hgh_i = hgh_o - offset, row_i = row_o - offset, col_i = col_o - offset;

  float pValue = 0;
  // load
  if ( ((hgh_i >= 0) && (hgh_i < z_size)) && \
       ((row_i >= 0) && (row_i < y_size)) && \
       ((col_i >= 0) && (col_i < x_size)) ) {
    tile[ty][tz][tx] = input[hgh_i * (y_size * x_size) + row_i * (x_size) + col_i];
  }
  else {
    tile[ty][tz][tx] = 0;
  }
  
  __syncthreads();

  // calcuate
  if (tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH) {
    for(int i=0; i<MASK_WIDTH; i++) {
      for(int j=0; j<MASK_WIDTH; j++) {
        for(int k=0; k<MASK_WIDTH; k++) {
          pValue += Mc[i][j][k] * tile[i+ty][j+tz][k+tx];
        }
      }
    }
    if (hgh_o < z_size && row_o < y_size && col_o < x_size)
      output[hgh_o * (y_size * x_size) + row_o * (x_size) + col_o] = pValue;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // float *hostInputClean = (float *)malloc((inputLength-3) * sizeof(float));
  // memcpy(hostInputClean, hostInput+3, sizeof(float)*(inputLength-3));
  // cout << "hostInputClean: " << endl;
  // for (int i=0; i<inputLength-3;i++)
  //   cout << hostInputClean[i] << " ";
  // cout << endl;

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int inputSize = (z_size * y_size * x_size) * sizeof(float);
  int outputSize = (z_size * y_size * x_size) * sizeof(float);
  cudaMalloc((void **) &deviceInput, inputSize);
  cudaMalloc((void **) &deviceOutput, outputSize);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  // copy input matrix to device memory
  cudaMemcpy(deviceInput, hostInput+3, inputSize, cudaMemcpyHostToDevice);
  // copy mask to constant memory
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  // TODO: dim setting might be wrong
  dim3 DimGrid(ceil(x_size/float(TILE_WIDTH)), ceil(y_size/float(TILE_WIDTH)), ceil(z_size/float(TILE_WIDTH)));
  dim3 DimBlock(TILE_WIDTH + MASK_WIDTH - 1, \
                TILE_WIDTH + MASK_WIDTH - 1, \
                TILE_WIDTH + MASK_WIDTH - 1);
  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, \
                               z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, inputSize, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // for(int i=0; i<z_size; i++) {
  //   for(int j=0; j<y_size; j++) {
  //     for(int k=0; k<x_size; k++) {
  //       cout << hostOutput[3 + i*(y_size*x_size) + j*(x_size) + key_t] << " ";
  //     }
  //     cout << endl;
  //   }
  //   cout << endl;
  // }

  // float Mc_dup[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];
  // cudaMemcpy(Mc_dup, Mc, kernelLength, cudaMemcpyDeviceToHost);
  // for(int i=0; i<3; i++) {
  //   for(int j=0; j<3; j++) {
  //     for(int k=0; k<3; k++) {
  //       cout << Mc_dup[i][j][k] << " ";
  //     }
  //     cout << endl;
  //   }
  //   cout << endl;
  // }

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  // free(hostInputClean);
  free(hostOutput);
  return 0;
}
