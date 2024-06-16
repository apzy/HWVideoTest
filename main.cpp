#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <string.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include "NvCodec/NvDecoder/NvDecoder.h"
#include "Utils/FFmpegDemuxer.h"
#include "Utils/ColorSpace.h"
#include "Common/AppDecUtils.h"
#include <chrono>
#include "NvCodec/NvEncoder/NvEncoderCuda.h"
#include "Utils/NvEncoderCLIOptions.h"
#include "Utils/FFmpegStreamer.h"

using NvEncCudaPtr = std::unique_ptr<NvEncoderCuda, std::function<void(NvEncoderCuda*)>>;

simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger();

auto EncodeDeleteFunc = [](NvEncoderCuda* pEnc)
	{
		if (pEnc)
		{
			pEnc->DestroyEncoder();
			delete pEnc;
		}
	};

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

int FindMin(volatile int* a, int n)
{
	int r = INT_MAX;
	for (int i = 0; i < n; i++)
	{
		if (a[i] < r)
		{
			r = a[i];
		}
	}
	return r;
}

template<class EncoderClass>
void InitializeEncoder(EncoderClass& pEnc, NvEncoderInitParam encodeCLIOptions, NV_ENC_BUFFER_FORMAT eFormat)
{
	NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
	NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };

	initializeParams.encodeConfig = &encodeConfig;
	pEnc->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
	encodeCLIOptions.SetInitParams(&initializeParams, eFormat);

	pEnc->CreateEncoder(&initializeParams);
}


void TranscodeOneToN(CUcontext cuContext, NV_ENC_BUFFER_FORMAT eFormat, NvEncoderInitParam encodeCLIOptions, std::ofstream& fpOut, NvDecoder* pDec, FFmpegDemuxer* pDemuxer, const char* szOutFileNamePrefix)
{
	const int nSrcFrame = 8;

	volatile bool bEnd = false;
	volatile int iDec = 0;

	uint8_t* apSrcFrame[nSrcFrame] = { 0 };

	std::vector<NvThread> vpth;
	int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
	uint8_t* pVideo = NULL, * pFrame = NULL;

	int nWidth, nHeight;
	CUdeviceptr pTmpImage = 0;
	cuMemAlloc(&pTmpImage, 1920 * 1080 * 4);

	uint8_t* dstFrame;
	int dstWidth = 640;
	int dstHeight = 480;
	size_t dstPitch = 0;

	cuMemAllocPitch((CUdeviceptr*)&dstFrame, &dstPitch, dstWidth, dstHeight + (dstHeight / 2), 16);
	cv::Mat show(cv::Size(dstWidth, dstHeight), CV_8UC4);


	std::unique_ptr<NvEncoderCuda> pEnc(new NvEncoderCuda(cuContext, dstWidth, dstHeight, eFormat));
	InitializeEncoder(pEnc, encodeCLIOptions, eFormat);

	do
	{
		pDemuxer->Demux(&pVideo, &nVideoBytes);
		nFrameReturned = pDec->Decode(pVideo, nVideoBytes);

		for (int i = 0; i < nFrameReturned; i++)
		{
			std::vector<std::vector<uint8_t>> vPacket;
			pFrame = pDec->GetFrame();
			int iMatrix = pDec->GetVideoFormatInfo().video_signal_description.matrix_coefficients;

			ResizeNv12(dstFrame, dstPitch, dstWidth, dstHeight, pFrame, pDec->GetDeviceFramePitch(), pDemuxer->GetWidth(), pDemuxer->GetHeight());
			Nv12ToColor32<BGRA32>(dstFrame, dstPitch, (uint8_t*)pTmpImage, 4 * dstWidth,
				dstWidth, dstHeight, iMatrix);

			const NvEncInputFrame* encoderInputFrame = pEnc->GetNextInputFrame();
			cuMemcpy((CUdeviceptr)encoderInputFrame->inputPtr, (CUdeviceptr)pTmpImage, dstWidth * dstHeight * 4);

			pEnc->EncodeFrame(vPacket);

			//GetImage(pTmpImage, show.data, 4 * dstWidth, dstHeight);
			//cv::imshow("frame", show);
			//cv::waitKey(1);
			for (std::vector<uint8_t>& packet : vPacket)
			{
				fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
			}
		}

	}
	while (nVideoBytes);
	std::vector<std::vector<uint8_t>> vPacket;
	pEnc->EndEncode(vPacket);
	for (std::vector<uint8_t>& packet : vPacket)
	{
		fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
	}
	pEnc->DestroyEncoder();
}


int main(int argc, char* argv[])
{

	NvEncoderInitParam encodeCLIOptions;

	int iGpu = 0;
	char szInFilePath[260] = "input.mp4";
	char szOutFileNamePrefix[260] = "out";
	std::vector<int2> vResolution;
	vResolution.push_back(make_int2(640, 480));
	std::vector<std::exception_ptr> vExceptionPtrs;
	try
	{
		CheckInputFile(szInFilePath);

		CUdevice cuDevice = 0;
		CUcontext cuContext = NULL;
		ck(cuInit(0));
		ck(cuDeviceGet(&cuDevice, iGpu));
		ck(cuCtxCreate(&cuContext, 0, cuDevice));

		FFmpegDemuxer demuxer(szInFilePath);
		NvDecoder dec(cuContext, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), false, true);

		std::ofstream fpOut("output.h264", std::ios::out | std::ios::binary);
		if (!fpOut)
		{
			std::ostringstream err;
			throw std::invalid_argument(err.str());
		}


		TranscodeOneToN(cuContext, NV_ENC_BUFFER_FORMAT_ARGB, encodeCLIOptions, fpOut, &dec, &demuxer, szOutFileNamePrefix);
		fpOut.close();

	}
	catch (const std::exception& ex)
	{
		std::cout << ex.what();
		exit(1);
	}
	return 0;
}

