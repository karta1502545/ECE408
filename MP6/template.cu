// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>
#include <iostream>
using namespace std;

#define BLOCK_SIZE 512 //@@ You can change this
// IMPORTANT: size should be large enough
//            so that we can complete the preSum in aux scan using 1 block

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float *aux, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[BLOCK_SIZE*2];
  
  int bx = blockIdx.x, bWidth = blockDim.x, tx = threadIdx.x;
  int i = bx * bWidth * 2 + tx * 2; // IMPORTANT!!! block should * 2

  // load T
  for(int offset=0; offset<2; offset++) {
    if (i+offset < len) T[tx*2 + offset] = input[i+offset];
    else T[tx*2+offset] = 0.0;
  }

  // preScan
  for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
    __syncthreads();
    int index = (tx+1) * stride * 2 - 1;
    if (index < 2*BLOCK_SIZE && (index-stride) >= 0)
      T[index] += T[index - stride];
  }
  
  // postScan
  for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    int index = (tx+1) * stride * 2 - 1;
    if ((index+stride) < 2*BLOCK_SIZE)
      T[index+stride] += T[index];
  }
  
  __syncthreads();
  for(int offset=0; offset<2; offset++) {
    if (i+offset < len) output[i+offset] = T[tx*2 + offset];
    // if (tx*2 + offset == 2*BLOCK_SIZE-1) aux[bx] = T[tx*2 + offset]; // line 63 is better
  }

  if (tx == 0) aux[bx] = T[BLOCK_SIZE*2 - 1];
  // At this point, I have every block preSum calculated.
}

__global__ void add(float *input, float *aux, int len) {
  // add aux into input
  __shared__ float offset;

  int tx = threadIdx.x, bx = blockIdx.x, bWidth = blockDim.x;
  int i = bx * bWidth + tx;
  
  if (bx != 0 && tx == 0) offset = aux[bx-1];
  __syncthreads();

  if (bx != 0 && i < len) {
    input[i] += offset;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  float *dummy;

  // my code
  float *hostAux;
  float *deviceAuxInput;
  float *deviceAuxOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  hostAux = (float *)malloc(ceil(numElements / float(BLOCK_SIZE * 2)) * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceAuxInput, ceil(numElements / float(BLOCK_SIZE * 2)) * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceAuxOutput, ceil(numElements / float(BLOCK_SIZE * 2)) * sizeof(float)));
  wbCheck(cudaMalloc((void **)&dummy, 1 * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numElements / float(BLOCK_SIZE * 2)), 1, 1); // IMPORTNAT: '/2'
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  dim3 DimGrid2(1, 1, 1);
  dim3 DimBlock2(BLOCK_SIZE, 1, 1);

  dim3 DimGrid3(ceil(numElements / float(BLOCK_SIZE * 2)), 1, 1);
  dim3 DimBlock3(BLOCK_SIZE*2, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the device
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, deviceAuxInput, numElements);
  cudaDeviceSynchronize();

  // calculate aux's PreSum, assume size of aux < BLOCK_SIZE
  scan<<<DimGrid2, DimBlock2>>>(deviceAuxInput, deviceAuxOutput, dummy, ceil(numElements / float(BLOCK_SIZE * 2)));
  cudaDeviceSynchronize();

  // add aux to deviceOutput to form complete answers
  add<<<DimGrid3, DimBlock3>>>(deviceOutput, deviceAuxOutput, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbCheck(cudaMemcpy(hostAux, deviceAuxOutput, ceil(numElements / float(BLOCK_SIZE * 2)) * sizeof(float),
                     cudaMemcpyDeviceToHost));
                     
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // [DEBUG] see output elements
  // for (int i=0; i<ceil(numElements / float(BLOCK_SIZE * 2)); i++)
  //   cout << hostAux[i] << " ";
  // cout << endl << endl << endl;


  // for (int i=0; i<numElements; i++)
  //   cout << hostOutput[i] << " ";
  // cout << endl;

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxInput);
  cudaFree(deviceAuxOutput);
  cudaFree(dummy);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);
  free(hostAux);

  return 0;
}
