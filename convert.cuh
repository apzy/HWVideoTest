#include <builtin_types.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <cuda.h>

typedef unsigned char   uint8;
typedef unsigned int    uint32;
typedef int             int32;

#define COLOR_COMPONENT_MASK            0x3FF
#define COLOR_COMPONENT_BIT_SIZE        10

#define MUL(x,y)    ((x)*(y))

namespace cuda_common
{
    __constant__ float  constHueColorSpaceMat2[9];  

    cudaError_t setColorSpace2(float hue);

    cudaError_t CUDAToBGR(uint32* dataY, uint32* dataUV, size_t pitchY, size_t pitchUV, unsigned char* d_dstRGB, int width, int height);

    cudaError_t convertInt32(unsigned char* src, uint32* dst, int width, int height);

    cudaError_t convertInt32toRgb(uint32* src, unsigned char* dst, int width, int height);

    cudaError_t resize(uint32* src, uint32* dst, int srcW, int srcH, int dstW, int dstH);

    cudaError_t rgb2nv12(unsigned char* rgb, unsigned char* nv12, int width, int height);

    cudaError_t rgb2yuv420p(unsigned char* rgb, unsigned char* yuv, int width, int height);
}