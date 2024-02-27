// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <wb.h>
#include <iostream>
using namespace std;

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
  
__global__ void total(float *input, float *output, int len) {
  // for each block, we use BLOCK_SIZE threads to
  // reduce 2 * block_size inputs to partialSum[0]
  __shared__ float partialSum[2 * BLOCK_SIZE];

  int tx = threadIdx.x;
  int bx = blockIdx.x;
  //@@ Load a segment of the input vector into shared memory
  int start = 2 * bx * BLOCK_SIZE;
  // each thread load tx and BLOCK_SIZE+tx
  if (start + tx < len)
    partialSum[tx] = input[start + tx];
  else
    partialSum[tx] = 0;
  
  if (start + BLOCK_SIZE + tx < len)
    partialSum[BLOCK_SIZE + tx] = input[start + BLOCK_SIZE + tx];
  else
    partialSum[BLOCK_SIZE + tx] = 0;
  
  // __syncthreads(); // we do not need it since we have line 45 already

  //@@ Traverse the reduction tree
  // All 32 thread in a SM are working until stride is shrinked to 16.
  for (int stride = BLOCK_SIZE; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tx < stride) {
      partialSum[tx] += partialSum[tx + stride];
    }
  }
  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  if (tx == 0)
    output[bx] = partialSum[0];
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceInput, numInputElements * sizeof(float));
  cudaMalloc((void **) &deviceOutput, numOutputElements * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numInputElements / (2.0 * BLOCK_SIZE)), 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  total<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  for(int i=0; i<numOutputElements; i++) {
      cout << hostOutput[i] << " " << endl;
  }

  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. For simplicity, we do not
   * require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}

/*
Bug Fixed:
1. DimGrid should divide by 2 (data size is 2x than # of threads)
2. cudaMalloc, cudaMemcpy should add sizeof(float)
3. [Kernel function] output should add "if (tx == 0)" to avoid other thread overwrite the result of thread 0
4. [Kernel function] should add boundary when loading shared memory (< len)
*/