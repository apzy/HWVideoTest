#include <builtin_types.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <memory.h>
#include <math.h>

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
}