#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <iostream>
using namespace std;

// reference: https://github.com/aschuh703/ECE408/tree/main/Project#milestone-2-baseline-convolutional-kernel
// reference: https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu

#define TILE_WIDTH 32

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

    const int W_Grid = ceil(float(W_out) / TILE_WIDTH); // IMPORTANT: for grid, we see an output element as a block unit.
    int b = blockIdx.x;
    int m = blockIdx.z;
    int h = (blockIdx.y / W_Grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_Grid) * TILE_WIDTH + threadIdx.x;

    // Insert your GPU convolution kernel code here
    float pValue = 0.0;
    for (int c=0; c<C; c++) {
        for (int kr=0; kr<K; kr++) {
            for (int kc=0; kc<K; kc++) {
                // if ((h*S+kr) < H && (w*S+kc) < W) {
                    pValue += in_4d(b, c, h*S+kr, w*S+kc) * mask_4d(m, c, kr, kc);
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

    int numOfStream;
    for (int i=20; i>=1; i--) {
        if (B / i > 0 && B % i == 0) {
            numOfStream = i;
            break;
        }
    }

    cudaStream_t stream[numOfStream];

    for (int i=0; i<numOfStream; i++)
        cudaStreamCreate(&stream[i]);

    float* host_out_cast = (float*) host_output;

    int stream_outSize = H_out * W_out * M * B / numOfStream;
    int stream_inSize = H * W * C * B / numOfStream;
    int remainingInputFloats = (H * W * C * B) % numOfStream;
    int remainingOutputFloats = (H_out * W_out * M * B) % numOfStream;

    cudaMalloc((void**) device_output_ptr, H_out * W_out * M * B * sizeof(float));
    cudaMalloc((void**) device_input_ptr, H * W * C * B * sizeof(float));
    cudaMalloc((void**) device_mask_ptr, M * C * K * K * sizeof(float));

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    const int H_Grid = ceil(H_out / float(TILE_WIDTH));
    const int W_Grid = ceil(W_out / float(TILE_WIDTH));
    const int G = H_Grid*W_Grid;
    // cout << "H_out = " << H_out << ", W_out = " << W_out << endl;
    // cout << "H_Grid = " << H_Grid << ", W_Grid = " << W_Grid << endl;
    // cout << "G = " << G << endl;

    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGrid(B/numOfStream, G, M); // batch_size, GridSize, # of mask
    cudaMemcpyAsync(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice, stream[0]);

    for (int i=0; i<numOfStream; i++) {
        int inputOffset = i * stream_inSize;
        int inputSize = (i == numOfStream - 1) ? (stream_inSize + remainingInputFloats) : stream_inSize;
        // int inputSize = stream_inSize;
        cudaMemcpyAsync((*device_input_ptr) + inputOffset, host_input + inputOffset, inputSize * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
    }
    for (int i=0; i<numOfStream; i++) {
        conv_forward_kernel<<<DimGrid,DimBlock,0,stream[i]>>>((*device_output_ptr) + i * stream_outSize, (*device_input_ptr) + i * stream_inSize, *device_mask_ptr, B, M, C, H, W, K, S);
    }
    for (int i=0; i<numOfStream; i++) {
        int outputOffset = i * stream_outSize;
        int outputSize = (i == numOfStream - 1) ? (stream_outSize + remainingOutputFloats) : stream_outSize;
        // int outputSize = stream_outSize;
        // Copy the output back to host
        cudaMemcpyAsync(host_out_cast + outputOffset, (*device_output_ptr)+ outputOffset, outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize(); // not sure if I need this.

    for (int i=0; i<numOfStream; i++)
        cudaStreamDestroy(stream[i]);
   
    // Free device memory
    cudaFree(device_output_ptr);
    cudaFree(device_input_ptr);
    cudaFree(device_mask_ptr);
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
