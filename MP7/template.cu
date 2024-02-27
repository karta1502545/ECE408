// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void floatToUnsignedChar(float *inputImage, unsigned char *ucharImage, int size)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size) {
	  ucharImage[idx] = (unsigned char) (255 * inputImage[idx]);
  }
}

__global__ void rgbToGrey(unsigned char *ucharImage, unsigned char *grayImage, int grayImageSize)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < grayImageSize) {
    unsigned char r = ucharImage[3*idx];
    unsigned char g = ucharImage[3*idx + 1];
    unsigned char b = ucharImage[3*idx + 2];
    grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void computeHistogram(unsigned int *histogram, unsigned char *grayImage, int grayImageSize)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ unsigned int subHistogram[HISTOGRAM_LENGTH];
  // I use one thread block (rather than input_size/HISTOGRAM_LENGTH) to calculate all the histogram.
  // This reduce the number of times of atomicAdd() execution with global memory to HISTOGRAM_LENGTH(256).
  int stride = blockDim.x * gridDim.x;

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    subHistogram[threadIdx.x] = 0;
  }
  __syncthreads();

  while (idx < grayImageSize) {
    atomicAdd(&(subHistogram[grayImage[idx]]), 1);
    idx += stride;
  }
  __syncthreads();

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    atomicAdd(&(histogram[threadIdx.x]), subHistogram[threadIdx.x]);
  }
}

__global__ void computeCDF(unsigned int *histogram, float *cdf, int size)
{
  /*
  cdf[0] = p(histogram[0])
  for ii from 1 to 256 do
    cdf[ii] = cdf[ii - 1] + p(histogram[ii])
  end
  */
  __shared__ float T[HISTOGRAM_LENGTH*2];
  
  int bx = blockIdx.x, bWidth = blockDim.x, tx = threadIdx.x;
  int i = bx * bWidth * 2 + tx * 2;

  // load T
  for(int offset=0; offset<2; offset++) {
    if (i+offset < HISTOGRAM_LENGTH) T[tx*2 + offset] = histogram[i+offset];
    else T[tx*2+offset] = 0.0;
  }

  // preScan
  for (int stride = 1; stride < 2*HISTOGRAM_LENGTH; stride *= 2) {
    __syncthreads();
    int index = (tx+1) * stride * 2 - 1;
    if (index < 2*HISTOGRAM_LENGTH && (index-stride) >= 0)
      T[index] += T[index - stride];
  }
  
  // postScan
  for (int stride = HISTOGRAM_LENGTH / 2; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (tx+1) * stride * 2 - 1;
    if ((index+stride) < 2*HISTOGRAM_LENGTH)
      T[index+stride] += T[index];
  }
  
  __syncthreads();
  for(int offset=0; offset<2; offset++) {
    if (i+offset < HISTOGRAM_LENGTH) cdf[i+offset] = T[tx*2 + offset] / float(size);
  }
}

__global__ void correct_color(float *cdf, float *output, unsigned char *ucharImage, int size)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < size) {
    output[idx] = (float) min(max(255*(cdf[ucharImage[idx]] - cdf[0])/(1.0 - cdf[0]), 0.0), 255.0) / 255.0;
  }

  // def correct_color(val) 
	// return clamp(255*(cdf[val] - cdfmin)/(1.0 - cdfmin), 0, 255.0)
  // end

  // def clamp(x, start, end)
  //   return min(max(x, start), end)
  // end
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  unsigned char *deviceInputImageUnsignedChar;
  unsigned char *deviceGrayImageData;
  unsigned int *deviceHistogram;
  float *deviceCdf;
  float *deviceOutput;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  int inputSize = imageWidth * imageHeight * imageChannels;
  cudaMalloc((void **) &deviceInputImageData, inputSize * sizeof(float));
  cudaMalloc((void **) &deviceInputImageUnsignedChar, inputSize * sizeof(unsigned char));
  cudaMalloc((void **) &deviceGrayImageData, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **) &deviceCdf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **) &deviceOutput, inputSize * sizeof(float));

  cudaMemset((void *) deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void *) deviceCdf, 0, HISTOGRAM_LENGTH * sizeof(float));

  cudaMemcpy(deviceInputImageData, hostInputImageData, inputSize * sizeof(float), cudaMemcpyHostToDevice);

  dim3 DimBlock(HISTOGRAM_LENGTH, 1, 1);
  dim3 DimGrid(ceil(inputSize * 1.0/HISTOGRAM_LENGTH), 1, 1);

  floatToUnsignedChar<<<DimGrid, DimBlock>>>(deviceInputImageData, deviceInputImageUnsignedChar, inputSize);
  rgbToGrey<<<DimGrid, DimBlock>>>(deviceInputImageUnsignedChar, deviceGrayImageData, imageWidth * imageHeight * 1);
  computeHistogram<<<1, DimBlock>>>(deviceHistogram, deviceGrayImageData, imageWidth * imageHeight * 1);
  computeCDF<<<DimGrid, DimBlock>>>(deviceHistogram, deviceCdf, imageWidth * imageHeight * 1);
  correct_color<<<DimGrid, DimBlock>>>(deviceCdf, deviceOutput, deviceInputImageUnsignedChar, inputSize);

  cudaMemcpy(hostOutputImageData, deviceOutput, inputSize*sizeof(float), cudaMemcpyDeviceToHost);
  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceInputImageUnsignedChar);
  cudaFree(deviceGrayImageData);
  cudaFree(deviceHistogram);
  cudaFree(deviceCdf);
  cudaFree(deviceOutput);

  return 0;
}
