cmake_minimum_required(VERSION 3.5)

project(HWVideoTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR D:/source/opencv_build/opencv/build/install)
find_package(OpenCV 4 REQUIRED)

add_executable(HWVideoTest "main.cpp" 
"convert.cuh" 
"convert.cu" 
"NvCodec/NvDecoder/NvDecoder.cpp"
"NvCodec/NvEncoder/NvEncoder.cpp"
"NvCodec/NvEncoder/NvEncoderCuda.cpp"
"Utils/ColorSpace.cu"
"Utils/Resize.cu"
"VideoCapture.h"
"VideoCapture.cpp"
 "VideoWriter.h" "VideoWriter.cpp")

target_include_directories(HWVideoTest PUBLIC "E:/Video_Codec_SDK_12.2.72/Interface")
target_include_directories(HWVideoTest PUBLIC "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")
target_include_directories(HWVideoTest PUBLIC "D:/source/ffmpeg/ffmpeg-5.1.2-full_build-shared/include")

target_link_directories(HWVideoTest PUBLIC "D:/source/ffmpeg/ffmpeg-5.1.2-full_build-shared/lib")
target_link_directories(HWVideoTest PUBLIC "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64")
target_link_directories(HWVideoTest PUBLIC "E:/Video_Codec_SDK_12.2.72/Lib/x64")

target_link_libraries(HWVideoTest PUBLIC avcodec.lib avdevice.lib avfilter.lib avformat.lib avutil.lib postproc.lib swresample.lib swscale.lib)
target_link_libraries(HWVideoTest PUBLIC ${OpenCV_LIBS})
target_link_libraries(HWVideoTest PUBLIC nvcuvid.lib nvencodeapi.lib cuda.lib)

include(GNUInstallDirs)
install(TARGETS HWVideoTest
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
