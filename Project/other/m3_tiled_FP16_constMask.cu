#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>
#include <iostream>
using namespace std;

// reference: https://github.com/aschuh703/ECE408/tree/main/Project#milestone-2-baseline-convolutional-kernel

__constant__ __half MASK_c[10000];
#define TILE_WIDTH 16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void my_kernel(__half *output, const __half *input, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{

    const int inSize = K + (TILE_WIDTH-1) * S;
    // const int inSize = TILE_WIDTH + K - 1;
    const int maskSize = K * K;
    const int in2dSize = inSize * inSize;

    extern __shared__ __half sharedMem[];

    __half *in2d = &sharedMem[0];

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) MASK_c[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // #define mask_2d(i1, i0) mask2d[(i1) * (K) + (i0)]
    #define in_2d(i1, i0) in2d[(i1) * (inSize) + (i0)]

    const int W_Grid = ceil(float(W_out) / TILE_WIDTH); // IMPORTANT: for grid, we see an output element as a block unit.
    int b = blockIdx.x;
    int m = blockIdx.z;
    int h = (blockIdx.y / W_Grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_Grid) * TILE_WIDTH + threadIdx.x;

    int hBase = (blockIdx.y / W_Grid) * TILE_WIDTH * S;
    int wBase = (blockIdx.y % W_Grid) * TILE_WIDTH * S;
    int ty = threadIdx.y, tx = threadIdx.x;

    // Insert your GPU convolution kernel code here
    __half pValue = __float2half(0.0f);
    for (int c=0; c<C; c++) {
        // load into shared memory
        // if (ty < K && tx < K)
            // mask_2d(ty, tx) = mask_4d(m, c, ty, tx);

        for (int i=ty; i<inSize; i+=blockDim.y) {
            for (int j=tx; j<inSize; j+=blockDim.x) {
                if (hBase+i < H && wBase+j < W)
                    in_2d(i, j) = in_4d(b, c, hBase+i, wBase+j);
                else
                    in_2d(i, j) = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        for (int kr=0; kr<K; kr++) {
            for (int kc=0; kc<K; kc++) {
                pValue = __hadd(pValue, __hmul(in_2d(ty*S+kr, tx*S+kc),  mask_4d(m, c, kr, kc)));
            }
        }
        __syncthreads();
    }

    if (h < H_out && w < W_out) {
        out_4d(b, m, h, w) = pValue;
    }


    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    __half *half_host_mask;
    half_host_mask = (__half *) malloc(M * C * K * K * sizeof(__half));
    for (int i=0; i<M * C * K * K; i++)
        half_host_mask[i] = __float2half(host_mask[i]);

    __half *half_host_input;
    half_host_input = (__half *) malloc(H * W * C * B * sizeof(__half));
    for (int i=0; i<H * W * C * B; i++)
        half_host_input[i] = __float2half(host_input[i]);

    cudaMalloc((void**) device_output_ptr, H_out * W_out * M * B * sizeof(float));
    cudaMalloc((void**) device_input_ptr, H * W * C * B * sizeof(float));
    // cudaMalloc((void**) device_mask_ptr, M * C * K * K * sizeof(float));

    // IMPORTANT: Pointer level should be right.
    cudaMemcpy(*device_input_ptr, half_host_input, H * W * C * B * sizeof(__half), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(__half), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(MASK_c, half_host_mask, M*C*K*K*sizeof(__half));

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    const int H_Grid = ceil(H_out / float(TILE_WIDTH));
    const int W_Grid = ceil(W_out / float(TILE_WIDTH));
    const int G = H_Grid*W_Grid;
    // cout << "H_out = " << H_out << ", W_out = " << W_out << endl;
    // cout << "H_Grid = " << H_Grid << ", W_Grid = " << W_Grid << endl;
    // cout << "G = " << G << endl;

    __half *half_device_output = (__half*) device_output;
    __half *half_device_input = (__half*) device_input;

    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGrid(B, G, M); // batch_size, GridSize, # of mask
    // conv_forward_kernel<<<DimGrid,DimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    const int inSize = K + (TILE_WIDTH-1) * S;
    size_t sharedMemSize = inSize * inSize * sizeof(__half);
    my_kernel<<<DimGrid,DimBlock,sharedMemSize>>>(half_device_output, half_device_input, B, M, C, H, W, K, S);
    cudaDeviceSynchronize(); // not sure if I need this.
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    __half* half_device_output = (__half*) device_output;
    __half* half_host_output = (__half*) host_output;

    // Copy the output back to host
    // cudaMemcpy(host_output, device_output, H_out * W_out * M * B * sizeof(float), cudaMemcpyDeviceToHost);

    gpuErrchk(cudaMemcpy(half_host_output, half_device_output, H_out * W_out * M * B * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int i=H_out * W_out * M * B; i>=0; i--)
        host_output[i] = __half2float(half_host_output[i]);
   
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
