#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>
#include <iostream>
using namespace std;

// reference: https://github.com/aschuh703/ECE408/tree/main/Project#milestone-2-baseline-convolutional-kernel

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define TILE_WIDTH 32

__global__ void conv_forward_kernel(__half *output, const __half *input, const __half *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int W_Grid = ceil(float(W_out) / TILE_WIDTH); // IMPORTANT: for grid, we see an output element as a block unit.
    int b = blockIdx.x;
    int m = blockIdx.z;
    int h = (blockIdx.y / W_Grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_Grid) * TILE_WIDTH + threadIdx.x;

    // Insert your GPU convolution kernel code here
    __half pValue = __float2half(0.0f);
    for (int c=0; c<C; c++) {
        for (int kr=0; kr<K; kr++) {
            for (int kc=0; kc<K; kc++) {
                // if ((h*S+kr) < H && (w*S+kc) < W) {
                    // pValue = __hadd(pValue, __hmul((in_4d(b, c, h*S+kr, w*S+kc)), (mask_4d(m, c, kr, kc))));
                    pValue = __float2half( __half2float(in_4d(b, c, h*S+kr, w*S+kc)) * __half2float(mask_4d(m, c, kr, kc)) );
                    // pValue += in_4d(b, c, h+kr, w+kc) * mask_4d(m, c, kr, kc);
                // }
            }
        }
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

    // gpuErrchk(cudaMalloc((void**) device_output_ptr, H_out * W_out * M * B * sizeof(float)));

    float* host_out_cast = (float*) host_output;

    __half *half_host_output;
    __half *half_host_input;
    __half *half_host_mask;

    __half *half_device_output;
    __half *half_device_input;
    __half *half_device_mask;

    half_host_input = (__half *) malloc(H * W * C * B * sizeof(__half));
    half_host_mask = (__half *) malloc(M * C * K * K * sizeof(__half));
    half_host_output = (__half *) malloc(H_out * W_out * M * B * sizeof(__half));

    gpuErrchk(cudaMalloc((void**) &half_device_input, H * W * C * B * sizeof(__half)));
    gpuErrchk(cudaMalloc((void**) &half_device_mask, M * C * K * K * sizeof(__half)));
    gpuErrchk(cudaMalloc((void**) &half_device_output, H_out * W_out * M * B * sizeof(__half)));

    for (int i=0; i<H * W * C * B; i++)
        half_host_input[i] = __float2half(host_input[i]);
    for (int i=0; i<M * C * K * K; i++)
        half_host_mask[i] = __float2half(host_mask[i]);

    gpuErrchk(cudaMemcpy(half_device_input, half_host_input, H * W * C * B * sizeof(__half), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(half_device_mask, half_host_mask, M * C * K * K * sizeof(__half), cudaMemcpyHostToDevice));

    const int H_Grid = ceil(H_out / float(TILE_WIDTH));
    const int W_Grid = ceil(W_out / float(TILE_WIDTH));
    const int G = H_Grid*W_Grid;
    // cout << "H_out = " << H_out << ", W_out = " << W_out << endl;
    // cout << "H_Grid = " << H_Grid << ", W_Grid = " << W_Grid << endl;
    // cout << "G = " << G << endl;

    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGrid(B, G, M); // batch_size, GridSize, # of mask
    conv_forward_kernel<<<DimGrid,DimBlock>>>(half_device_output, half_device_input, half_device_mask, B, M, C, H, W, K, S);
    gpuErrchk(cudaDeviceSynchronize()); // not sure if I need this.

    // Copy the output back to host
    gpuErrchk(cudaMemcpy(half_host_output, half_device_output, H_out * W_out * M * B * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int i=0; i<H_out * W_out * M * B; i++)
        host_out_cast[i] = __half2float(half_host_output[i]);
   
    // Free device memory
    cudaFree(half_device_output);
    cudaFree(half_device_input);
    cudaFree(half_device_mask);
    free(half_host_input);
    free(half_host_output);
    free(half_host_mask);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    return;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    return;
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
