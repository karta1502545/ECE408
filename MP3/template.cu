#include <wb.h>
// #include <iostream>
// using namespace std;

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP

  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  // subTileM[ty][tx] = 0.0;
  // subTileN[ty][tx] = 0.0;
  // __syncthreads();
  float pValue = 0;
  int numOfq = numAColumns / TILE_WIDTH;
  if (numAColumns % TILE_WIDTH) numOfq += 1;
  // if (Row < numCRows && Col < numCColumns) { // NOTE: This is wrong!
    for (int q = 0; q < numOfq; ++q) {
      // load things from device global memory to shared memory
      if (q * TILE_WIDTH + tx < numAColumns && Row < numARows) // should not limited by "Col < numCColumns"
        subTileM[ty][tx] = A[Row * numAColumns + (q * TILE_WIDTH + tx)];
      else
        subTileM[ty][tx] = 0;
      
      if (q * TILE_WIDTH + ty < numBRows && Col < numBColumns) // should not limited by "Row < numCRows"
        subTileN[ty][tx] = B[((q * TILE_WIDTH + ty) * numBColumns) + Col];
      else
        subTileN[ty][tx] = 0;

      __syncthreads();
      // calcuate pValue
      for (int k = 0; k < TILE_WIDTH; ++k) {
        // if ((q * TILE_WIDTH + k < numAColumns)) // no need to use it
          pValue += subTileM[ty][k] * subTileN[k][tx];
      }
      __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
      C[Row * numCColumns + Col] = pValue;
  // }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int Asize = numARows * numAColumns * sizeof(float);
  int Bsize = numBRows * numBColumns * sizeof(float);
  int Csize = numCRows * numCColumns * sizeof(float);
  cudaMalloc((void **) &deviceA, Asize);
  cudaMalloc((void **) &deviceB, Bsize);
  cudaMalloc((void **) &deviceC, Csize);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, Asize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, Bsize, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // TODO: now sure if the dim is right
  // cout << ceil(numCColumns/float(TILE_WIDTH)) << ", " << ceil(numCRows/float(TILE_WIDTH)) << endl;
  dim3 DimGrid(ceil(numCColumns/float(TILE_WIDTH)), ceil(numCRows/float(TILE_WIDTH)), 1);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid,DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, Csize, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");


  // cout << "ceil = " << ceil(numCRows / float(TILE_WIDTH)) << ", " << ceil(numCColumns / float(TILE_WIDTH)) << endl;
  
  // for (int i=numCRows-10; i<numCRows; i++) {
  //   for (int j=numCColumns-10; j<numCColumns; j++) {
  //     cout << hostC[i*numCColumns+j] << " ";
  //   }
  //   cout << endl;
  // }
  // cout << endl;

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}