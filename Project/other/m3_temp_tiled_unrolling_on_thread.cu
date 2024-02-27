#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <iostream>
using namespace std;

// reference: https://github.com/aschuh703/ECE408/tree/main/Project#milestone-2-baseline-convolutional-kernel

__constant__ float MASK_c[10000];
#define TILE_WIDTH 16
// tileWidth=28, ((3+(28-1)*4)^2+3*3)*4 = 49320 > 49152(maxShareMemorySize per block), shared memory overflow!
// tileWidth=27, ((3+(27-1)*4)^2+3*3)*4 = 45832 > 49152(maxShareMemorySize per block), time = 10.5+60.1=70.6ms
// tileWidth=16, time = 8.6 + 43.6 = 52.2ms

// with constant memory, time = 7.83 + 37.42 = 45.25ms

/*
Conv-GPU==
Layer Time: 297.666 ms
Op Time: 6.69915 ms
Conv-GPU==
Layer Time: 246.629 ms
Op Time: 33.0789 ms

Conv-GPU==
Layer Time: 319.333 ms
Op Time: 6.62453 ms
Conv-GPU==
Layer Time: 257.743 ms
Op Time: 32.8471 ms

Conv-GPU==
Layer Time: 302.691 ms
Op Time: 6.7325 ms
Conv-GPU==
Layer Time: 252.77 ms
Op Time: 33.0872 ms
*/

__global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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

    const int inSize = K + (TILE_WIDTH-1) * S;
    // const int inSize = TILE_WIDTH + K - 1;
    const int maskSize = K * K;
    const int in2dSize = inSize * inSize;

    extern __shared__ float sharedMem[];

    float *in2d = &sharedMem[0];

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

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
    
    // int ty = threadIdx.y, tx = threadIdx.x;
    // Insert your GPU convolution kernel code here
    float pValue[4];

    for (int ty=2*threadIdx.y; ty<2*(threadIdx.y+1); ty++) {
        for (int tx=2*threadIdx.x; tx<2*(threadIdx.x+1); tx++) {
            int py = (ty-2*threadIdx.y), px = (tx-2*threadIdx.x);
            int pIdx = py*2 + px;
            pValue[pIdx] = 0.0;
            if (h+py < H_out && w+px < W_out) {
                for (int c=0; c<C; c++) {
                    // load into shared memory
                    // if (ty < K && tx < K)
                        // mask_2d(ty, tx) = mask_4d(m, c, ty, tx);

                    for (int k=ty; k<inSize; k+=blockDim.y) {
                        for (int l=tx; l<inSize; l+=blockDim.x) {
                            if (hBase+k < H && wBase+l < W)
                                in_2d(k, l) = in_4d(b, c, hBase+k, wBase+l);
                            else
                                in_2d(k, l) = 0.0;
                        }
                    }
                    
                    __syncthreads();
                    for (int kr=0; kr<K; kr+=2) {
                        for (int kc=0; kc<K; kc+=2) {
                            /* 2 by 2 */
                            if (kr < K && kc < K)
                                pValue[pIdx] += in_2d(ty*S+kr, tx*S+kc) * mask_4d(m, c, kr, kc);
                            if (kr < K && kc+1 < K)
                                pValue[pIdx] += in_2d(ty*S+kr, tx*S+kc+1) * mask_4d(m, c, kr, kc+1);
                            if (kr+1 < K && kc < K)
                                pValue[pIdx] += in_2d(ty*S+kr+1, tx*S+kc) * mask_4d(m, c, kr+1, kc);
                            if (kr+1 < K && kc+1 < K)
                                pValue[pIdx] += in_2d(ty*S+kr+1, tx*S+kc+1) * mask_4d(m, c, kr+1, kc+1);
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }

    for (int i=0; i<2; i++) 
        for (int j=0; j<2; j++)
            if (h+i < H_out && w+j < W_out)
                out_4d(b, m, h+i, w+j) = pValue[i*2+j];

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
    // cudaMalloc((void**) device_mask_ptr, M * C * K * K * sizeof(float));

    // IMPORTANT: Pointer level should be right.
    cudaMemcpy(*device_input_ptr, host_input, H * W * C * B * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(MASK_c, host_mask, M*C*K*K*sizeof(float));
    // get_device_properties();

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

    // const int H_out = (H)/S + 1;
    // const int W_out = (W)/S + 1;

    // const int H_out = H;
    // const int W_out = W;

    const int H_Grid = ceil(H_out / float(TILE_WIDTH));
    const int W_Grid = ceil(W_out / float(TILE_WIDTH));
    const int G = H_Grid*W_Grid;
    // const int G = ceil(H / float(TILE_WIDTH)) * ceil(W / float(TILE_WIDTH));
    // cout << "H_out = " << H_out << ", W_out = " << W_out << endl;
    // cout << "H_Grid = " << H_Grid << ", W_Grid = " << W_Grid << endl;
    // cout << "G = " << G << endl;

    dim3 DimBlock(TILE_WIDTH/2, TILE_WIDTH/2, 1);
    dim3 DimGrid(B, G, M); // batch_size, GridSize, # of mask
    // int inSize = TILE_WIDTH * S + K - 1;
    // const int inSize = TILE_WIDTH + K - 1;
    const int inSize = K + (TILE_WIDTH-1) * S;
    size_t sharedMemSize = inSize * inSize * sizeof(float);

    conv_forward_kernel<<<DimGrid,DimBlock,sharedMemSize>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    cudaDeviceSynchronize(); // not sure if I need this.
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // Copy the output back to host
    cudaMemcpy(host_output, device_output, H_out * W_out * M * B * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i=0; i<H_out * W_out; i++) {
    //     cout << host_output[i] << " ";
    // }
    // cout << endl;
   
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    // cudaFree(device_mask);
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
