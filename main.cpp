#include <chrono>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "convert.cuh"
#include <cuda.h>
#include <iostream>
#include <algorithm>
#include <memory>
#include "NvCodec/NvDecoder/NvDecoder.h"
#include "Utils/FFmpegDemuxer.h"
#include "Utils/ColorSpace.h"
#include "Common/AppDecUtils.h"
#include <chrono>

simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger();

/**
*   @brief  Function to copy image data from CUDA device pointer to host buffer
*   @param  dpSrc   - CUDA device pointer which holds decoded raw frame
*   @param  pDst    - Pointer to host buffer which acts as the destination for the copy
*   @param  nWidth  - Width of decoded frame
*   @param  nHeight - Height of decoded frame
*/
void GetImage(CUdeviceptr dpSrc, uint8_t* pDst, int nWidth, int nHeight)
{
	CUDA_MEMCPY2D m = { 0 };
	m.WidthInBytes = nWidth;
	m.Height = nHeight;
	m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	m.srcDevice = (CUdeviceptr)dpSrc;
	m.srcPitch = m.WidthInBytes;
	m.dstMemoryType = CU_MEMORYTYPE_HOST;
	m.dstDevice = (CUdeviceptr)(m.dstHost = pDst);
	m.dstPitch = m.WidthInBytes;
	cuMemcpy2D(&m);
}

enum OutputFormat
{
	native = 0, bgrp, rgbp, bgra, rgba, bgra64, rgba64
};

std::vector<std::string> vstrOutputFormatName =
{
	"native", "bgrp", "rgbp", "bgra", "rgba", "bgra64", "rgba64"
};

std::string GetSupportedFormats()
{
	std::ostringstream oss;
	for (auto& v : vstrOutputFormatName)
	{
		oss << " " << v;
	}

	return oss.str();
}

int main(int argc, char** argv)
{
	char szInFilePath[256] = "input.mp4", szOutFilePath[256] = "output.mp4";
	OutputFormat eOutputFormat = bgra;
	int iGpu = 0;
	bool bReturn = 1;
	CUdeviceptr pTmpImage = 0;

	try
	{
		CheckInputFile(szInFilePath);

		if (!*szOutFilePath)
		{
			sprintf(szOutFilePath, "out.%s", vstrOutputFormatName[eOutputFormat].c_str());
		}

		std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
		if (!fpOut)
		{
			std::ostringstream err;
			err << "Unable to open output file: " << szOutFilePath << std::endl;
			throw std::invalid_argument(err.str());
		}

		ck(cuInit(0));
		int nGpu = 0;
		ck(cuDeviceGetCount(&nGpu));
		if (iGpu < 0 || iGpu >= nGpu)
		{
			std::ostringstream err;
			err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]";
			throw std::invalid_argument(err.str());
		}

		CUcontext cuContext = NULL;
		createCudaContext(&cuContext, iGpu, 0);

		FFmpegDemuxer demuxer(szInFilePath);
		NvDecoder dec(cuContext, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
		int nWidth = 0, nHeight = 0, nFrameSize = 0;
		int anSize[] = { 0, 3, 3, 4, 4, 8, 8 };
		std::unique_ptr<uint8_t[]> pImage;

		int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0, iMatrix = 0;
		uint8_t* pVideo = nullptr;
		uint8_t* pFrame;

		auto beginTime = std::chrono::high_resolution_clock::now();
		cv::Mat show;
		do
		{
			demuxer.Demux(&pVideo, &nVideoBytes);
			nFrameReturned = dec.Decode(pVideo, nVideoBytes);
			if (!nFrame && nFrameReturned)
			{
				LOG(INFO) << dec.GetVideoInfo();
				// Get output frame size from decoder
				nWidth = dec.GetWidth(); nHeight = dec.GetHeight();
				nFrameSize = eOutputFormat == native ? dec.GetFrameSize() : nWidth * nHeight * anSize[eOutputFormat];
				std::unique_ptr<uint8_t[]> pTemp(new uint8_t[nFrameSize]);
				pImage = std::move(pTemp);
				cuMemAlloc(&pTmpImage, nWidth * nHeight * anSize[eOutputFormat]);
				show = cv::Mat(cv::Size(dec.GetWidth(), dec.GetHeight()), CV_8UC4);
			}

			for (int i = 0; i < nFrameReturned; i++)
			{
				iMatrix = dec.GetVideoFormatInfo().video_signal_description.matrix_coefficients;
				pFrame = dec.GetFrame();
				Nv12ToColor32<BGRA32>(pFrame, dec.GetWidth(), (uint8_t*)pTmpImage, 4 * dec.GetWidth(), dec.GetWidth(), dec.GetHeight(), iMatrix);
				GetImage(pTmpImage, show.data, 4 * dec.GetWidth(), dec.GetHeight());
				cv::imshow("frame", show);
				cv::waitKey(1);
			}
			nFrame += nFrameReturned;
			printf("%d\n", nFrame);
		}
		while (nVideoBytes);
		auto endTime = std::chrono::high_resolution_clock::now();
		auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime);
		std::cout << elapsedTime.count() << std::endl;

		if (pTmpImage)
		{
			cuMemFree(pTmpImage);
		}

		std::cout << "Total frame decoded: " << nFrame << std::endl << "Saved in file " << szOutFilePath << std::endl;
		fpOut.close();
	}
	catch (const std::exception& ex)
	{
		std::cout << ex.what();
		exit(1);
	}
	return 0;
}

