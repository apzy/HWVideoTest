#include <chrono>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "convert.cuh"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavformat/avformat.h>
#include <libavutil/avassert.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

static AVBufferRef* hw_device_ctx = NULL;
static enum AVPixelFormat hw_pix_fmt;
static FILE* output_file = NULL;

int dstWith = 1080, dstHeight = 1920;

unsigned char* pHwRgb = nullptr;
unsigned char* pHwDstRgb = nullptr;

cv::Mat frameMat(cv::Size(1080, 1920), CV_8UC3);

uint32* pSrcInt32Data = nullptr;
uint32* pDstInt32Data = nullptr;

unsigned char* dstyuvData = nullptr;

static int hw_decoder_init(AVCodecContext* ctx, const enum AVHWDeviceType type)
{
	int err = 0;

	if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type, NULL, NULL, 0)) < 0)
	{
		fprintf(stderr, "Failed to create specified HW device.\n");
		return err;
	}
	ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

	return err;
}

static void encode(AVCodecContext* enc_ctx, AVFrame* frame, AVPacket* pkt, FILE* outfile)
{
	int ret;

	/* send the frame to the encoder */
	if (frame)
		printf("Send frame %3d\n", frame->pts);

	ret = avcodec_send_frame(enc_ctx, frame);
	if (ret < 0)
	{
		fprintf(stderr, "Error sending a frame for encoding\n");
		exit(1);
	}

	while (ret >= 0)
	{
		ret = avcodec_receive_packet(enc_ctx, pkt);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
			return;
		else if (ret < 0)
		{
			fprintf(stderr, "Error during encoding\n");
			exit(1);
		}

		printf("Write packet %3d (size=%5d)\n", pkt->pts, pkt->size);
		fwrite(pkt->data, 1, pkt->size, outfile);
		av_packet_unref(pkt);
	}
}

static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts)
{
	const enum AVPixelFormat* p;

	for (p = pix_fmts; *p != -1; p++)
	{
		if (*p == hw_pix_fmt)
			return *p;
	}

	fprintf(stderr, "Failed to get HW surface format.\n");
	return AV_PIX_FMT_NONE;
}

static int decode_write(AVCodecContext* avctx, AVPacket* packet)
{
	AVFrame* frame = NULL, * sw_frame = NULL;
	AVFrame* tmp_frame = NULL;
	uint8_t* buffer = NULL;
	int size;
	int ret = 0;

	ret = avcodec_send_packet(avctx, packet);
	if (ret < 0)
	{
		fprintf(stderr, "Error during decoding\n");
		return ret;
	}

	while (1)
	{
		auto startTime = std::chrono::high_resolution_clock::now();
		if (!(frame = av_frame_alloc()) || !(sw_frame = av_frame_alloc()))
		{
			fprintf(stderr, "Can not alloc frame\n");
			ret = AVERROR(ENOMEM);
			goto fail;
		}

		ret = avcodec_receive_frame(avctx, frame);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
		{
			av_frame_free(&frame);
			av_frame_free(&sw_frame);
			return 0;
		}
		else if (ret < 0)
		{
			fprintf(stderr, "Error while decoding\n");
			goto fail;
		}

		if (frame->format == hw_pix_fmt)
		{
			cudaError_t cudaStatus;
			if (pHwRgb == nullptr)
			{
				cudaStatus = cudaMalloc((void**)&pHwRgb, 3 * frame->width * frame->height * sizeof(unsigned char));
				cudaStatus = cudaMalloc((void**)&pHwDstRgb, 3 * dstWith * dstHeight * sizeof(unsigned char));
				cudaStatus = cudaMalloc((void**)&pSrcInt32Data, frame->width * frame->height * sizeof(uint32));
				cudaStatus = cudaMalloc((void**)&pDstInt32Data, dstWith * dstHeight * sizeof(uint32));
				cudaStatus = cudaMalloc((void**)&dstyuvData, dstWith * dstHeight * sizeof(unsigned char) * 1.5);
			}
			printf("%d %d %d\n", frame->data[1] - frame->data[0],frame->linesize[0], frame->linesize[1]);
			cudaStatus = cuda_common::CUDAToBGR((uint32*)frame->data[0], (uint32*)frame->data[1], frame->linesize[0], 
				frame->linesize[1], pHwRgb, frame->width, frame->height);
			if (cudaStatus != cudaSuccess)
			{
				std::cout << "CUDAToBGR failed !!!" << std::endl;
				return 0;
			}
			cuda_common::convertInt32(pHwRgb, pSrcInt32Data, frame->width, frame->height);
			cuda_common::resize(pSrcInt32Data, pDstInt32Data, frame->width, frame->height, dstWith, dstHeight);
			cuda_common::convertInt32toRgb(pDstInt32Data, pHwDstRgb, dstWith, dstHeight);
			cuda_common::rgb2yuv420p(pHwDstRgb, dstyuvData, dstWith, dstHeight);

			//cv::Mat resizedMat(dstHeight + dstHeight / 2, dstWith, CV_8UC1);
			//printf("%d %d\n", resizedMat.cols, resizedMat.rows);
			//cudaMemcpy(resizedMat.data, dstyuvData, 1.5 * dstWith * dstHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			//cv::Mat rgbDst;
			//cv::cvtColor(resizedMat, rgbDst, cv::COLOR_YUV2BGR_YV12,3);
			//cv::imshow("out", rgbDst);
			//cv::waitKey(1);
			//cudaStatus = cudaMemcpy(frameMat.data, pHwRgb, 3 * frame->width * frame->height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			/* retrieve data from GPU to CPU */
			//if ((ret = av_hwframe_transfer_data(sw_frame, frame, 0)) < 0)
			//{
			//	fprintf(stderr, "Error transferring the data to system memory\n");
			//	goto fail;
			//}
			//tmp_frame = sw_frame;
		}
		else
		{
			//tmp_frame = frame;
		}

		//printf("tmp frame fmt ; %s\n", av_get_pix_fmt_name((AVPixelFormat)tmp_frame->format));

		{
			auto endTime = std::chrono::high_resolution_clock::now();
			auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
			std::cout << "create Elapsed time: " << elapsedTime.count() << "mis\n";
		}

	fail:
		av_frame_free(&frame);
		av_frame_free(&sw_frame);
		av_freep(&buffer);
		if (ret < 0)
			return ret;
	}
}

int main(int argc, char* argv[])
{
	cuda_common::setColorSpace2(0);

	int video_stream, ret;
	const char* outputFile, * codecName;
	const AVCodec* encoderCodec;
	AVCodecContext* encoderCtx = NULL;
	AVPacket* encoderPkt;

	uint8_t endcode[] = { 0, 0, 1, 0xb7 };
	outputFile = "output.mp4";
	codecName = "h264_nvenc";
	encoderCodec = avcodec_find_encoder_by_name(codecName);
	if (!encoderCodec)
	{
		fprintf(stderr, "Codec '%s' not found\n", codecName);
		exit(1);
	}
	encoderCtx = avcodec_alloc_context3(encoderCodec);

	if (!encoderCtx)
	{
		fprintf(stderr, "Could not allocate video codec context\n");
		exit(1);
	}

	encoderPkt = av_packet_alloc();
	if (!encoderPkt)
		exit(1);
	encoderCtx->bit_rate = 400000;
	encoderCtx->width = dstWith;
	encoderCtx->height = dstHeight;
	encoderCtx->time_base = {1,25};
	encoderCtx->framerate = {25,1};
	encoderCtx->gop_size = 10;
	encoderCtx->max_b_frames = 1;
	encoderCtx->pix_fmt = AV_PIX_FMT_NV12;

	if (encoderCodec->id == AV_CODEC_ID_H264)
		av_opt_set(encoderCtx->priv_data, "preset", "slow", 0);

	/* open it */
	ret = avcodec_open2(encoderCtx, encoderCodec, NULL);
	if (ret < 0)
	{
		fprintf(stderr, "Could not open codec: %d\n", ret);
		exit(1);
	}

	AVFormatContext* input_ctx = NULL;
	AVStream* video = NULL;
	AVCodecContext* decoder_ctx = NULL;
	const AVCodec* decoder = NULL;
	AVPacket* packet = NULL;
	enum AVHWDeviceType type;
	int i;

	type = av_hwdevice_find_type_by_name("cuda");
	if (type == AV_HWDEVICE_TYPE_NONE)
	{
		fprintf(stderr, "Device type %s is not supported.\n", argv[1]);
		fprintf(stderr, "Available device types:");
		while ((type = av_hwdevice_iterate_types(type)) != AV_HWDEVICE_TYPE_NONE)
			fprintf(stderr, " %s", av_hwdevice_get_type_name(type));
		fprintf(stderr, "\n");
		return -1;
	}

	packet = av_packet_alloc();
	if (!packet)
	{
		fprintf(stderr, "Failed to allocate AVPacket\n");
		return -1;
	}

	/* open the input file */
	if (avformat_open_input(&input_ctx, "input.mp4", NULL, NULL) != 0)
	{
		fprintf(stderr, "Cannot open input file '%s'\n", argv[2]);
		return -1;
	}

	if (avformat_find_stream_info(input_ctx, NULL) < 0)
	{
		fprintf(stderr, "Cannot find input stream information.\n");
		return -1;
	}

	/* find the video stream information */
	ret = av_find_best_stream(input_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
	if (ret < 0)
	{
		fprintf(stderr, "Cannot find a video stream in the input file\n");
		return -1;
	}
	video_stream = ret;

	for (i = 0;; i++)
	{
		const AVCodecHWConfig* config = avcodec_get_hw_config(decoder, i);
		if (!config)
		{
			fprintf(stderr,
				"Decoder %s does not support device type %s.\n",
				decoder->name,
				av_hwdevice_get_type_name(type));
			return -1;
		}
		if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == type)
		{
			hw_pix_fmt = config->pix_fmt;
			break;
		}
	}

	if (!(decoder_ctx = avcodec_alloc_context3(decoder)))
		return AVERROR(ENOMEM);

	video = input_ctx->streams[video_stream];
	if (avcodec_parameters_to_context(decoder_ctx, video->codecpar) < 0)
		return -1;

	decoder_ctx->get_format = get_hw_format;

	if (hw_decoder_init(decoder_ctx, type) < 0)
		return -1;

	if ((ret = avcodec_open2(decoder_ctx, decoder, NULL)) < 0)
	{
		fprintf(stderr, "Failed to open codec for stream #%u\n", video_stream);
		return -1;
	}

	/* open the file to dump raw data */
	output_file = fopen("output", "w+b");

	/* actual decoding and dump the raw data */
	auto beginTime = std::chrono::high_resolution_clock::now();
	int fps = 0;
	while (ret >= 0)
	{
		++fps;
		if ((ret = av_read_frame(input_ctx, packet)) < 0)
			break;

		if (video_stream == packet->stream_index)
			ret = decode_write(decoder_ctx, packet);

		av_packet_unref(packet);

		{
			auto endTime = std::chrono::high_resolution_clock::now();
			auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime
				- beginTime);
			int sendTime = elapsedTime.count();
			if (sendTime >= 1000)
			{
				printf("fps = %d\n", fps);
				beginTime = endTime;
				fps = 0;
			}
		}
	}

	/* flush the decoder */
	ret = decode_write(decoder_ctx, NULL);

	if (output_file)
		fclose(output_file);
	av_packet_free(&packet);
	avcodec_free_context(&decoder_ctx);
	avformat_close_input(&input_ctx);
	av_buffer_unref(&hw_device_ctx);

	return 0;
}
