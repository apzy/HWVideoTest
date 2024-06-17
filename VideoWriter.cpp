#include "VideoWriter.h"

template<class EncoderClass>
void InitializeEncoder(EncoderClass& m_encoder, NvEncoderInitParam encodeCLIOptions, NV_ENC_BUFFER_FORMAT eFormat)
{
	NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
	NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };

	initializeParams.encodeConfig = &encodeConfig;
	m_encoder->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
	encodeCLIOptions.SetInitParams(&initializeParams, eFormat);

	m_encoder->CreateEncoder(&initializeParams);
}

VideoWriter::VideoWriter(const std::string& filePath, const int& width, const int& height)
{
	printf("%s\n", m_encodeCLIOptions.GetHelpMessage().c_str());
	m_filePath = filePath;

	m_width = width;
	m_height = height;

	m_outFp = std::ofstream(m_filePath.c_str(), std::ios::out | std::ios::binary);
	if (!m_outFp)
	{
		std::ostringstream err;
		throw std::invalid_argument(err.str());
	}

	ck(cuInit(0));
	ck(cuDeviceGet(&m_cuDevice, 0));
	ck(cuCtxCreate(&m_cuContext, 0, m_cuDevice));

	m_encoder = new NvEncoderCuda(m_cuContext, width, height, NV_ENC_BUFFER_FORMAT_ARGB);
	InitializeEncoder(m_encoder, m_encodeCLIOptions, NV_ENC_BUFFER_FORMAT_ARGB);
}

VideoWriter::~VideoWriter()
{
	delete m_encoder;
}

void VideoWriter::write(const CUdeviceptr& frame)
{
	std::vector<std::vector<uint8_t>> vPacket;
	const NvEncInputFrame* encoderInputFrame = m_encoder->GetNextInputFrame();

	NvEncoderCuda::CopyToDeviceFrame(m_cuContext, (uint8_t*)frame,
		m_encoder->GetWidthInBytes(NV_ENC_BUFFER_FORMAT_ARGB, m_encoder->GetEncodeWidth()), (CUdeviceptr)encoderInputFrame->inputPtr,
		(int)encoderInputFrame->pitch,
		m_encoder->GetEncodeWidth(),
		m_encoder->GetEncodeHeight(),
		CU_MEMORYTYPE_HOST,
		encoderInputFrame->bufferFormat,
		encoderInputFrame->chromaOffsets,
		encoderInputFrame->numChromaPlanes);

	m_encoder->EncodeFrame(vPacket);

	for (std::vector<uint8_t>& packet : vPacket)
	{
		m_outFp.write(reinterpret_cast<char*>(packet.data()), packet.size());
	}
}

void VideoWriter::release()
{
	std::vector<std::vector<uint8_t>> vPacket;
	m_encoder->EndEncode(vPacket);
	for (std::vector<uint8_t>& packet : vPacket)
	{
		m_outFp.write(reinterpret_cast<char*>(packet.data()), packet.size());
	}
	m_encoder->DestroyEncoder();
}
