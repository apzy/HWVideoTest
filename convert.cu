#include "convert.cuh"

#define CUDA_THREAD_NUM 1024

#define BLOCK_DIM 32
#define threadNum 1024
#define WARP_SIZE 32
#define elemsPerThread 1

#define Y_COEF_R 0.299f  
#define Y_COEF_G 0.587f  
#define Y_COEF_B 0.114f  
#define U_COEF_R -0.14713f  
#define U_COEF_G -0.28886f  
#define U_COEF_B 0.436f  
#define V_COEF_R 0.615f  
#define V_COEF_G -0.51498f  
#define V_COEF_B -0.10001f  

namespace cuda_common
{

	__device__ void YUV2RGB2(uint32* yuvi, float* red, float* green, float* blue)
	{
		float luma, chromaCb, chromaCr;

		// Prepare for hue adjustment
		luma = (float)yuvi[0];
		chromaCb = (float)((int32)yuvi[1] - 512.0f);
		chromaCr = (float)((int32)yuvi[2] - 512.0f);


		// Convert YUV To RGB with hue adjustment
		*red = MUL(luma, constHueColorSpaceMat2[0]) +
			MUL(chromaCb, constHueColorSpaceMat2[1]) +
			MUL(chromaCr, constHueColorSpaceMat2[2]);
		*green = MUL(luma, constHueColorSpaceMat2[3]) +
			MUL(chromaCb, constHueColorSpaceMat2[4]) +
			MUL(chromaCr, constHueColorSpaceMat2[5]);
		*blue = MUL(luma, constHueColorSpaceMat2[6]) +
			MUL(chromaCb, constHueColorSpaceMat2[7]) +
			MUL(chromaCr, constHueColorSpaceMat2[8]);

	}

	__device__ unsigned char clip_v(int x, int min_val, int  max_val)
	{
		if (x > max_val)
		{
			return max_val;
		}
		else if (x < min_val)
		{
			return min_val;
		}
		else
		{
			return x;
		}
	}

	// CUDA kernel for outputing the final RGB output from NV12;

	extern "C"
		__global__ void CUDAToBGR_drvapi(uint32 * dataY, uint32 * dataUV, size_t pitchY, size_t pitchUV, unsigned char* dstImage, int width, int height)
	{

		int32 x, y;

		// Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
		x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
		y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= width)
		{
			return;
		}

		if (y >= height)
		{
			return;
		}

		uint32 yuv101010Pel[2];
		uint8* srcImageU8_Y = (uint8*)dataY;
		uint8* srcImageU8_UV = (uint8*)dataUV;

		// Read 2 Luma components at a time, so we don't waste processing since CbCr are decimated this way.
		// if we move to texture we could read 4 luminance values
		yuv101010Pel[0] = (srcImageU8_Y[y * pitchY + x]) << 2;
		yuv101010Pel[1] = (srcImageU8_Y[y * pitchY + x + 1]) << 2;

		int32 y_chroma = y >> 1;

		if (y & 1)  // odd scanline ?
		{
			uint32 chromaCb;
			uint32 chromaCr;

			chromaCb = srcImageU8_UV[y_chroma * pitchUV + x];
			chromaCr = srcImageU8_UV[y_chroma * pitchUV + x + 1];

			if (y_chroma < ((height >> 1) - 1)) // interpolate chroma vertically
			{
				chromaCb = (chromaCb + srcImageU8_UV[(y_chroma + 1) * pitchUV + x] + 1) >> 1;
				chromaCr = (chromaCr + srcImageU8_UV[(y_chroma + 1) * pitchUV + x + 1] + 1) >> 1;
			}

			yuv101010Pel[0] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));
			yuv101010Pel[0] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

			yuv101010Pel[1] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));
			yuv101010Pel[1] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
		}
		else
		{
			yuv101010Pel[0] |= ((uint32)srcImageU8_UV[y_chroma * pitchUV + x] << (COLOR_COMPONENT_BIT_SIZE + 2));
			yuv101010Pel[0] |= ((uint32)srcImageU8_UV[y_chroma * pitchUV + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

			yuv101010Pel[1] |= ((uint32)srcImageU8_UV[y_chroma * pitchUV + x] << (COLOR_COMPONENT_BIT_SIZE + 2));
			yuv101010Pel[1] |= ((uint32)srcImageU8_UV[y_chroma * pitchUV + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
		}

		// this steps performs the color conversion
		uint32 yuvi[6];
		float red[2], green[2], blue[2];

		yuvi[0] = (yuv101010Pel[0] & COLOR_COMPONENT_MASK);
		yuvi[1] = ((yuv101010Pel[0] >> COLOR_COMPONENT_BIT_SIZE) & COLOR_COMPONENT_MASK);
		yuvi[2] = ((yuv101010Pel[0] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

		yuvi[3] = (yuv101010Pel[1] & COLOR_COMPONENT_MASK);
		yuvi[4] = ((yuv101010Pel[1] >> COLOR_COMPONENT_BIT_SIZE) & COLOR_COMPONENT_MASK);
		yuvi[5] = ((yuv101010Pel[1] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

		// YUV to RGB Transformation conversion
		YUV2RGB2(&yuvi[0], &red[0], &green[0], &blue[0]);
		YUV2RGB2(&yuvi[3], &red[1], &green[1], &blue[1]);


		dstImage[y * width * 3 + x * 3] = clip_v(blue[0] * 0.25, 0, 255);
		dstImage[y * width * 3 + x * 3 + 3] = clip_v(blue[1] * 0.25, 0, 255);

		dstImage[width * y * 3 + x * 3 + 1] = clip_v(green[0] * 0.25, 0, 255);
		dstImage[width * y * 3 + x * 3 + 4] = clip_v(green[1] * 0.25, 0, 255);

		dstImage[width * y * 3 + x * 3 + 2] = clip_v(red[0] * 0.25, 0, 255);
		dstImage[width * y * 3 + x * 3 + 5] = clip_v(red[1] * 0.25, 0, 255);
	}

	cudaError_t setColorSpace2(float hue)
	{

		float hueSin = sin(hue);
		float hueCos = cos(hue);

		float hueCSC[9];
		//CCIR 709
		hueCSC[0] = 1.0f;
		hueCSC[1] = hueSin * 1.57480f;
		hueCSC[2] = hueCos * 1.57480f;
		hueCSC[3] = 1.0;
		hueCSC[4] = (hueCos * -0.18732f) - (hueSin * 0.46812f);
		hueCSC[5] = (hueSin * 0.18732f) - (hueCos * 0.46812f);
		hueCSC[6] = 1.0f;
		hueCSC[7] = hueCos * 1.85560f;
		hueCSC[8] = hueSin * -1.85560f;

		cudaError_t cudaStatus = cudaMemcpyToSymbol(constHueColorSpaceMat2, hueCSC, 9 * sizeof(float), 0, cudaMemcpyHostToDevice);
		float tmpf[9];
		memset(tmpf, 0, 9 * sizeof(float));
		cudaMemcpyFromSymbol(tmpf, constHueColorSpaceMat2, 9 * sizeof(float), 0, ::cudaMemcpyDefault);
		cudaDeviceSynchronize();

		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		return cudaStatus;
	}

	cudaError_t CUDAToBGR(uint32* dataY, uint32* dataUV, size_t pitchY, size_t pitchUV, unsigned char* d_dstRGB, int width, int height)
	{
		dim3 block(32, 16, 1);
		dim3 grid((width + (2 * block.x - 1)) / (2 * block.x), (height + (block.y - 1)) / block.y, 1);
		CUDAToBGR_drvapi << < grid, block >> > (dataY, dataUV, pitchY, pitchUV, d_dstRGB, width, height);
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "NV12ToRGB_drvapi launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching NV12ToRGB_drvapi !\n", cudaStatus);
			return cudaStatus;
		}

		return cudaStatus;
	}

	extern "C" __global__ void CUDAconvertInt32(unsigned char* src, uint32 * dst, int width, int height)
	{
		int idx = blockIdx.x * CUDA_THREAD_NUM + threadIdx.x;
		int r = *(src + idx * 3);
		int g = *(src + idx * 3 + 1);
		int b = *(src + idx * 3 + 2);
		*(dst + idx) = 0xff000000 |
			((((uint32)r) << 16) & 0xff0000) |
			((((uint32)g) << 8) & 0xff00) |
			((uint32)b);
	}


	extern "C" __global__ void CUDAconvertInt32toRgb(uint32 * src, unsigned char* dst, int width, int height)
	{
		int idx = blockIdx.x * CUDA_THREAD_NUM + threadIdx.x;
		uint32 data = *(src + idx);
		int r = data & 0xff;
		int g = (data >> 8) & 0xff;
		int b = (data >> 16) & 0xff;
		*(dst + idx * 3) = b;
		*(dst + idx * 3 + 1) = g;
		*(dst + idx * 3 + 2) = r;
	}

	extern "C" __global__ void SomeKernel(uint32 * originalImage, uint32 * resizedImage, int w, int h, int w2, int h2)
	{
		__shared__ uint32 tile[1024];
		const float x_ratio = ((float)(w - 1)) / w2;
		const float y_ratio = ((float)(h - 1)) / h2;
		unsigned int threadId = blockIdx.x * threadNum * elemsPerThread + threadIdx.x * elemsPerThread;
		unsigned int shift = 0;
		while ((threadId < w2 * h2 && shift < elemsPerThread))
		{
			const uint32 i = threadId / w2;
			const uint32 j = threadId - (i * w2);
			//float x_diff, y_diff, blue, red, green;

			const uint32 x = (int)(x_ratio * j);
			const uint32 y = (int)(y_ratio * i);
			const float x_diff = (x_ratio * j) - x;
			const float y_diff = (y_ratio * i) - y;
			const uint32 index = (y * w + x);
			const uint32 a = originalImage[index];
			const uint32 b = originalImage[index + 1];
			const uint32 c = originalImage[index + w];
			const uint32 d = originalImage[index + w + 1];
			// blue element
			const float blue = (a & 0xff) * (1 - x_diff) * (1 - y_diff) + (b & 0xff) * (x_diff) * (1 - y_diff) +
				(c & 0xff) * (y_diff) * (1 - x_diff) + (d & 0xff) * (x_diff * y_diff);

			// green element
			const float green = ((a >> 8) & 0xff) * (1 - x_diff) * (1 - y_diff) + ((b >> 8) & 0xff) * (x_diff) * (1 - y_diff) +
				((c >> 8) & 0xff) * (y_diff) * (1 - x_diff) + ((d >> 8) & 0xff) * (x_diff * y_diff);

			// red element
			const float red = ((a >> 16) & 0xff) * (1 - x_diff) * (1 - y_diff) + ((b >> 16) & 0xff) * (x_diff) * (1 - y_diff) +
				((c >> 16) & 0xff) * (y_diff) * (1 - x_diff) + ((d >> 16) & 0xff) * (x_diff * y_diff);


			tile[threadIdx.x] =
				0xff000000 |
				((((uint32)red) << 16) & 0xff0000) |
				((((uint32)green) << 8) & 0xff00) |
				((uint32)blue);

			//printf("%x %d %d %d\n", a, b, c, d);

			threadId++;
			//threadId+= WARP_SIZE;
			shift++;
		}

		__syncthreads();
		threadId = blockIdx.x * threadNum * elemsPerThread + threadIdx.x * elemsPerThread;
		resizedImage[threadId] = tile[threadIdx.x];

	}

	extern "C" __global__ void rgb_to_nv12_kernel(unsigned char* rgb, unsigned char* nv12, int width, int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < width && y < height)
		{
			int rgb_index = (y * width + x) * 3;
			int y_index = y * width + x;
			int uv_index = width * height + (y / 2) * width + x;

			// 提取RGB值  
			unsigned char r = rgb[rgb_index];
			unsigned char g = rgb[rgb_index + 1];
			unsigned char b = rgb[rgb_index + 2];

			//printf("%d %d %d\n", r, g, b);

			// 转换到YUV  
			float y_val = 0.299f * r + 0.587f * g + 0.114f * b;
			float u_val = -0.169f * r - 0.331f * g + 0.5f * b + 128.0f;
			float v_val = 0.5f * r - 0.419f * g - 0.081f * b + 128.0f;

			// 存储Y分量  
			nv12[y_index] = static_cast<unsigned char>(y_val);

			// 对于UV分量，需要处理交织存储，且每两个像素共享一对UV值  
			if (y % 2 == 0 && x < width)
			{
				nv12[uv_index] = static_cast<unsigned char>(u_val);
				if (x + 1 < width)
				{ // 防止越界  
					nv12[uv_index + 1] = static_cast<unsigned char>(v_val);
				}
			}
		}
	}

	extern "C" __global__ void rgb_to_yuv420p(unsigned char* rgb, unsigned char* yuv, int width, int height)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < width && y < height)
		{
			int rgb_index = (y * width + x) * 3;
			float r = rgb[rgb_index];
			float g = rgb[rgb_index + 1];
			float b = rgb[rgb_index + 2];

			float y_val = Y_COEF_R * r + Y_COEF_G * g + Y_COEF_B * b;
			yuv[(y * width + x)] = static_cast<unsigned char>(y_val);

			if ((x % 2 == 0) && (y % 2 == 0))
			{   
				float u_val = U_COEF_R * r + U_COEF_G * g + U_COEF_B * b + 128; // 加128是为了将色度分量范围移到[0, 255]  
				float v_val = V_COEF_R * r + V_COEF_G * g + V_COEF_B * b + 128;

				int uv_width = width / 2;
				int uv_index = ((y / 2) * uv_width + (x / 2)) * 2;
				yuv[width * height + uv_index] = static_cast<unsigned char>(u_val);
				yuv[width * height + uv_index + 1] = static_cast<unsigned char>(v_val);
			}
		}
	}

	cudaError_t rgb2yuv420p(unsigned char* rgb, unsigned char* yuv, int width, int height)
	{
		dim3 blockSize(32, 32); // 可以根据GPU架构调整  
		dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
		rgb_to_yuv420p << <gridSize, blockSize >> > (rgb, yuv, width, height);

		cudaError_t cudaStatus = cudaGetLastError();
		cudaStatus = cudaDeviceSynchronize();
		return cudaStatus;
	}

	cudaError_t rgb2nv12(unsigned char* rgb, unsigned char* nv12, int width, int height)
	{
		dim3 blockSize(32,32); // 可以根据GPU架构调整  
		dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
		rgb_to_nv12_kernel << <gridSize, blockSize >> > (rgb, nv12, width, height);
		cudaError_t cudaStatus = cudaGetLastError();
		cudaStatus = cudaDeviceSynchronize();
		return cudaStatus;
	}

	cudaError_t convertInt32(unsigned char* src, uint32* dst, int width, int height)
	{
		int blockSize = width * height / CUDA_THREAD_NUM + 1;
		CUDAconvertInt32 << <blockSize, CUDA_THREAD_NUM >> > (src, dst, width, height);
		cudaError_t cudaStatus = cudaGetLastError();
		cudaStatus = cudaDeviceSynchronize();
		return cudaStatus;
	}

	cudaError_t convertInt32toRgb(uint32* src, unsigned char* dst, int width, int height)
	{
		int blockSize = width * height / CUDA_THREAD_NUM + 1;
		CUDAconvertInt32toRgb << <blockSize, CUDA_THREAD_NUM >> > (src, dst, width, height);
		cudaError_t cudaStatus = cudaGetLastError();
		cudaStatus = cudaDeviceSynchronize();
		return cudaError_t();
	}


	cudaError_t resize(uint32* src, uint32* dst, int srcW, int srcH, int dstW, int dstH)
	{
		dim3 threads = dim3(threadNum, 1, 1); //block size 32,32,x
		dim3 blocks = dim3(dstW * dstH / threadNum * elemsPerThread, 1, 1);
		SomeKernel << <blocks, threads >> > (src, dst, srcW, srcH, dstW, dstH);
		cudaDeviceSynchronize();
		return cudaError_t();
	}
}