#include <builtin_types.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <cuda.h>

#include "datatypes.h"

#define DEFAULT_ALPHA 255

extern "C" {
	__constant__ float matYuv2Rgb[9];
	__constant__ float matRgb2Yuv[9];

	int yuv2rgb_cuda(const uint8_t* src[], int srcStride[], uint8_t* dst[], int dstStride[], int nWidth, int nHeight, 
		int srcFormat, int dstFormat, CUstream stream);
}

