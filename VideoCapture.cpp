#include "VideoCapture.h"

VideoCapture::VideoCapture(const std::string& filePath)
{
	m_filePath = filePath;
	CheckInputFile(m_filePath.c_str());

	ck(cuInit(0));
	ck(cuDeviceGet(&m_cuDevice, 0));
	ck(cuCtxCreate(&m_cuContext, 0, m_cuDevice));
	m_demuxer = new FFmpegDemuxer(m_filePath.c_str());
	Rect cropRect = { 0, 100, 100, 200 };
	m_decoder = new NvDecoder(m_cuContext, true, FFmpeg2NvCodecId(m_demuxer->GetVideoCodec()), false, true);
	m_frameSize = 0;
	get_frame();
}


VideoCapture::~VideoCapture()
{}

uint8_t* VideoCapture::read()
{
	uint8_t* ptr = nullptr;
	get_frame();
	if (m_frames.size() > 0)
	{
		ptr = m_frames.front();
		m_frames.pop_front();
	}
	return ptr;
}

int VideoCapture::get_width()
{
	return m_demuxer->GetWidth();
}

int VideoCapture::get_height()
{
	return m_demuxer->GetHeight();
}

int VideoCapture::get_pitch()
{
	return m_decoder->GetDeviceFramePitch();
}

int VideoCapture::get_matrix()
{
	return m_decoder->GetVideoFormatInfo().video_signal_description.matrix_coefficients;
}

void VideoCapture::get_frame()
{
	do
	{
		m_demuxer->Demux(&m_video, &m_videoBytes);
		int frameReturned = m_decoder->Decode(m_video, m_videoBytes);
		if (m_videoBytes == 0)
			break;
		for (int i = 0; i < frameReturned; ++i)
		{
			if (m_frameSize == 0)
			{
				m_frameSize = m_decoder->GetDeviceFramePitch() * m_decoder->GetHeight() * 1.5;
			}
			uint8_t* frame = NULL;
			cuMemAlloc((CUdeviceptr*)&frame, m_frameSize);
			cuMemcpy((CUdeviceptr)frame, (CUdeviceptr)m_decoder->GetFrame(), m_frameSize);
			m_frames.push_back(frame);
		}
	}
	while (m_frames.size() == 0);
}
