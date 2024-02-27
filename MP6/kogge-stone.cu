// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>
// #include <iostream>
// using namespace std;

#define BLOCK_SIZE 512 //@@ You can change this

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
  // TODO: Double buffered and aux array.
  __shared__ float XY[BLOCK_SIZE];
  int bx = blockIdx.x, bWidth = blockDim.x, tx = threadIdx.x;
  int i = bx * bWidth + tx;

  // load data
  if (i < len) XY[tx] = input[i];
  else XY[tx] = 0;
  
  for (int stride = 1; stride < bWidth; stride *= 2) {
    float temp = XY[tx]; // IMPORTANT: not 0
    __syncthreads();
    if (tx >= stride)
      temp += XY[tx - stride];
    __syncthreads();
    XY[tx] = temp;
  }
  if (i < len) // IMPORTANT
    output[i] = XY[tx];
  // At this point, I have every block preSum calculated.
  if (tx == bWidth - 1)
    aux[bx] = XY[tx];
}

__global__ void add(float *input, float *aux, int len) {
  // add aux into input
  int tx = threadIdx.x, bx = blockIdx.x, bWidth = blockDim.x;
  int i = bx * bWidth + tx;
  __shared__ float offset;
  if (bx != 0) offset = aux[bx-1];

  if (bx != 0 && i < len)
    input[i] += offset;
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
  // float *hostAux;
  float *deviceAuxInput;
  float *deviceAuxOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceAuxInput, ceil(numElements / float(BLOCK_SIZE)) * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceAuxOutput, ceil(numElements / float(BLOCK_SIZE)) * sizeof(float)));
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
  dim3 DimGrid(ceil(numElements / float(BLOCK_SIZE)), 1, 1);
  dim3 DimGrid2(1, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the device
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, deviceAuxInput, numElements);
  cudaDeviceSynchronize();

  // calculate aux's PreSum, assume size of aux < BLOCK_SIZE
  scan<<<DimGrid2, DimBlock>>>(deviceAuxInput, deviceAuxOutput, dummy, ceil(numElements / float(BLOCK_SIZE)));
  cudaDeviceSynchronize();
  add<<<DimGrid, DimBlock>>>(deviceOutput, deviceAuxOutput, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // [DEBUG] see output elements
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

  return 0;
}
