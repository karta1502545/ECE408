
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <iostream>
using namespace std;

// reference: https://github.com/aschuh703/ECE408/tree/main/Project#milestone-2-baseline-convolutional-kernel

#define TILE_WIDTH 32
#define MUL 2
#define MUL_SQUARE MUL*MUL

/*
Test batch size: 100
Conv-GPU==
Layer Time: 7.906 ms
Op Time: 0.475706 ms
Conv-GPU==
Layer Time: 7.29744 ms
Op Time: 1.67055 ms

Test Accuracy: 0.86

Test batch size: 100
Conv-GPU==
Layer Time: 8.32807 ms
Op Time: 0.628193 ms
Conv-GPU==
Layer Time: 17.4729 ms
Op Time: 9.39184 ms

Test Accuracy: 0.86
*/

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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

    const int W_Grid = ceil(float(W_out) / (TILE_WIDTH*MUL)); // IMPORTANT: for grid, we see an output element as a block unit.
    int b = blockIdx.x;
    int m = blockIdx.z;
    // int h = (blockIdx.y / W_Grid) * TILE_WIDTH + 2*threadIdx.y;
    // int w = (blockIdx.y % W_Grid) * TILE_WIDTH + 2*threadIdx.x;
    int hBase = (blockIdx.y / W_Grid) * TILE_WIDTH * MUL;
    int wBase = (blockIdx.y % W_Grid) * TILE_WIDTH * MUL;

    // Insert your GPU convolution kernel code here
    float pValue[MUL_SQUARE];

    # pragma unroll
    for (int i = 0; i < MUL_SQUARE; i++) {
        int ty = MUL*threadIdx.y + i / MUL;
        int tx = MUL*threadIdx.x + i % MUL;
        int h = hBase + ty;
        int w = wBase + tx;
        int py = i / MUL, px = i % MUL;
        int pIdx = py*MUL + px;
        pValue[pIdx] = 0.0;
        for (int c=0; c<C; c++) {
            for (int kr=0; kr<K; kr+=4) {
                for (int kc=0; kc<K; kc+=4) {
                    if (kr < K && kc < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr, w*S+kc) * mask_4d(m, c, kr, kc);
                    if (kr < K && kc+1 < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr, w*S+kc+1) * mask_4d(m, c, kr, kc+1);
                    if (kr < K && kc+2 < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr, w*S+kc+2) * mask_4d(m, c, kr, kc+2);
                    if (kr < K && kc+3 < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr, w*S+kc+3) * mask_4d(m, c, kr, kc+3);

                    if (kr+1 < K && kc < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr+1, w*S+kc) * mask_4d(m, c, kr+1, kc);
                    if (kr+1 < K && kc+1 < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr+1, w*S+kc+1) * mask_4d(m, c, kr+1, kc+1);
                    if (kr+1 < K && kc+2 < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr+1, w*S+kc+2) * mask_4d(m, c, kr+1, kc+2);
                    if (kr+1 < K && kc+3 < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr+1, w*S+kc+3) * mask_4d(m, c, kr+1, kc+3);

                    if (kr+2 < K && kc < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr+2, w*S+kc) * mask_4d(m, c, kr+2, kc);
                    if (kr+2 < K && kc+1 < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr+2, w*S+kc+1) * mask_4d(m, c, kr+2, kc+1);
                    if (kr+2 < K && kc+2 < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr+2, w*S+kc+2) * mask_4d(m, c, kr+2, kc+2);
                    if (kr+2 < K && kc+3 < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr+2, w*S+kc+3) * mask_4d(m, c, kr+2, kc+3);

                    if (kr+3 < K && kc < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr+3, w*S+kc) * mask_4d(m, c, kr+3, kc);
                    if (kr+3 < K && kc+1 < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr+3, w*S+kc+1) * mask_4d(m, c, kr+3, kc+1);
                    if (kr+3 < K && kc+2 < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr+3, w*S+kc+2) * mask_4d(m, c, kr+3, kc+2);
                    if (kr+3 < K && kc+3 < K)
                        pValue[pIdx] += in_4d(b, c, h*S+kr+3, w*S+kc+3) * mask_4d(m, c, kr+3, kc+3);
                }
            }
        }
        // if (h < H_out && w < W_out) {
        //     out_4d(b, m, h, w) = pValue[pIdx];
        // }
    }

    // for (int ty=MUL*threadIdx.y; ty<MUL*(threadIdx.y+1); ty++) {
    //     for (int tx=MUL*threadIdx.x; tx<MUL*(threadIdx.x+1); tx++) {
    //         int h = hBase + ty;
    //         int w = wBase + tx;
    //         if (h < H_out && w < W_out) {
    //             int py = (ty-MUL*threadIdx.y), px = (tx-MUL*threadIdx.x);
    //             int pIdx = py*MUL + px;
    //             out_4d(b, m, h, w) = pValue[pIdx];
    //         }
    //     }
    // }

    # pragma unroll
    for (int i = 0; i < MUL_SQUARE; i++) {
        int ty = MUL*threadIdx.y + i / MUL;
        int tx = MUL*threadIdx.x + i % MUL;
        int h = hBase + ty;
        int w = wBase + tx;
        if (h < H_out && w < W_out) {
            out_4d(b, m, h, w) = pValue[i];
        }
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

    cudaMalloc((void**) device_output_ptr, H_out * W_out * M * B * sizeof(float));
    cudaMalloc((void**) device_input_ptr, H * W * C * B * sizeof(float));
    cudaMalloc((void**) device_mask_ptr, M * C * K * K * sizeof(float));

    // IMPORTANT: Pointer level should be right.
    cudaMemcpy(*device_input_ptr, host_input, H * W * C * B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);

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

    const int H_Grid = ceil(H_out / float(TILE_WIDTH*MUL));
    const int W_Grid = ceil(W_out / float(TILE_WIDTH*MUL));
    const int G = H_Grid*W_Grid;
    // cout << "H_out = " << H_out << ", W_out = " << W_out << endl;
    // cout << "H_Grid = " << H_Grid << ", W_Grid = " << W_Grid << endl;
    // cout << "G = " << G << endl;

    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGrid(B, G, M); // batch_size, GridSize, # of mask
    conv_forward_kernel<<<DimGrid,DimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    cudaDeviceSynchronize(); // not sure if I need this.
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // Copy the output back to host
    cudaMemcpy(host_output, device_output, H_out * W_out * M * B * sizeof(float), cudaMemcpyDeviceToHost);
   
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
