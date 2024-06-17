#pragma once

#include <string>
#include "NvCodec/NvDecoder/NvDecoder.h"
#include "Utils/FFmpegDemuxer.h"
#include "Utils/ColorSpace.h"
#include "Common/AppDecUtils.h"
#include "NvCodec/NvEncoder/NvEncoderCuda.h"
#include "Utils/NvEncoderCLIOptions.h"
#include "Utils/FFmpegStreamer.h"
#include <cuda.h>

class VideoCapture
{
public:
	VideoCapture(const std::string& filePath);
	~VideoCapture();

	uint8_t* read();

	int get_width();

	int get_height();

	int get_pitch();

	int get_matrix();

private:
	void get_frame();

private:
	std::string m_filePath;
	CUdevice m_cuDevice = 0;
	CUcontext m_cuContext = NULL;
	FFmpegDemuxer* m_demuxer;
	NvDecoder* m_decoder;

	int m_frameIdx;

	std::list<uint8_t*> m_frames;

	uint8_t* m_video;
	int m_videoBytes;
};
