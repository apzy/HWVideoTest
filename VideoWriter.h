#pragma  once
#include <string>
#include <cuda.h>
#include "NvCodec/NvEncoder/NvEncoderCuda.h"
#include "Utils/NvEncoderCLIOptions.h"

class VideoWriter
{
public:
	VideoWriter(const std::string& filePath,const int& width,const int& height);
	~VideoWriter();

	void write(const CUdeviceptr& frame);

	void release();

private:
	std::string m_filePath;
	CUdevice m_cuDevice = 0;
	CUcontext m_cuContext = NULL;
	NvEncoderCuda* m_encoder;
	NvEncoderInitParam m_encodeCLIOptions;
	std::ofstream m_outFp;
	int m_width;
	int m_height;
};