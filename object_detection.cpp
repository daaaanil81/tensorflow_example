/*
 * This program is used for object detecting by using model from link:
 * https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1
 *
 * Program has class with hard code names of layers inside class.
 * So, this class can work only with that model.
 * */

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include "detection_model.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#ifdef __cplusplus
}
#endif

enum Code {
    SUCCESS_CODE = 0,
    ERROR_CODE   = -1,
};

struct SwsContext_Deleter {
    void operator() (SwsContext* ptr) {
        if (ptr != nullptr) {
            sws_freeContext(ptr);
        }
    }
};

static
int decode_packet(DetectionModel& model, AVPacket* pPacket,
        AVCodecContext *pCodecContext, AVFrame* pFrame) {
    int res = -1;
    int cvLinesizes[1] = {0};
    std::ostringstream os("frame");
    cv::Mat image;
    AVPixelFormat src_pix_fmt = AV_PIX_FMT_YUV420P;
    AVPixelFormat dst_pix_fmt = AV_PIX_FMT_RGB24;
    std::unique_ptr<SwsContext, SwsContext_Deleter> sws_ctx(nullptr, SwsContext_Deleter());

    /* Supply raw packet data as input to a decoder. */
    res = avcodec_send_packet(pCodecContext, pPacket);
    if (res != SUCCESS) {
        LOG(ERROR) << "Failed to send packet to decoder";

        return res;
    }

    /* Return decoded output data from a decoder. */
    res = avcodec_receive_frame(pCodecContext, pFrame);
    if (res == AVERROR(EAGAIN) || res == AVERROR_EOF) {
        return SUCCESS_CODE;
    } else if (res != Code::SUCCESS) {
        LOG(ERROR) << "Failed to receive frame from decoder";

        return res;
    }

    LOG(INFO) << "Frame " << pCodecContext->frame_number <<
                " (type=" << av_get_picture_type_char(pFrame->pict_type) <<
                ", size=" << pFrame->pkt_size <<
                " bytes, format=" << pFrame->format <<
                ") pts " << pFrame->pts << " " << pFrame->width <<
                " x " << pFrame->height <<
                " key_frame " << pFrame->key_frame <<
                " [DTS " << pFrame->coded_picture_number << "]";

    /* create scaling context */
    sws_ctx.reset(sws_getContext(pFrame->width, pFrame->height, src_pix_fmt,
                                 pFrame->width, pFrame->height, dst_pix_fmt,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr));
    image = cv::Mat(pFrame->height, pFrame->width, CV_8UC3);

    cvLinesizes[0] = image.step1();

    /* convert to destination format */
    sws_scale(sws_ctx.get(), pFrame->data, pFrame->linesize, 0, pFrame->height,
              &image.data, cvLinesizes);


    if (pCodecContext->frame_number % 3 == 0) {
        model.Testing(image);
    }

    os << pCodecContext->frame_number << ".jpg";
    cv::imwrite(os.str(), image);

    return res;
}

int ffmpeg_proceed(DetectionModel& mode, const std::string& filename) {
    int res = SUCCESS_CODE;
    int video_stream_index = -1;
    int response = 0;
    int count_of_packets = 8;

    AVPacket* pPacket = nullptr;
    AVFrame* pFrame = nullptr;
    AVFormatContext* pFormatContext = nullptr;
    AVCodecContext* pCodecContext = nullptr;

    AVCodec* pCodec = nullptr;
    AVCodecParameters* pCodecParameters = nullptr;

    /* Allocate memory for context
     * AVFormatContext holds the header information from the format (Container)
     * */
    pFormatContext = avformat_alloc_context();
    if (pFormatContext == nullptr) {
        LOG(ERROR) << "Failed with allocate memory for context";

        return ERROR_CODE;
    }

    /* Open an input stream and read the header. */
    res = avformat_open_input(&pFormatContext, filename.c_str(), nullptr, nullptr);
    if (res != SUCCESS_CODE) {
        LOG(ERROR) << "Failed with receiving format context of file";

        goto free_context;
    }

    LOG(INFO) << "Format: " << pFormatContext->iformat->name <<
                " Duration: " << pFormatContext->duration << " us";

    /* Read packets of a media file to get stream information. */
    res = avformat_find_stream_info(pFormatContext, nullptr);
    if (res != SUCCESS) {
        std::cout << "Failed with find stream in file" << std::endl;

        goto close_input;
    }

    LOG(INFO) << "Count of Stream: " << pFormatContext->nb_streams;

    for (int i = 0; i < pFormatContext->nb_streams; i++) {
        /* Receive Codec parameters */
        AVCodecParameters* pLocalCodecParameters = pFormatContext->streams[i]->codecpar;
        /* Receive decoder */
        AVCodec* pLocalCodec = avcodec_find_decoder(pLocalCodecParameters->codec_id);
        if (pLocalCodec == nullptr) {
            LOG(ERROR) << "Unsupported codec";

            continue;
        }

        if (pLocalCodecParameters->codec_type == AVMEDIA_TYPE_VIDEO) {
            if (video_stream_index == -1) {
                video_stream_index = i;
                pCodec = pLocalCodec;
                pCodecParameters = pLocalCodecParameters;
            }

            LOG(INFO) << "Video Codec: resolution "
                      << pLocalCodecParameters->width << " x "
                      << pLocalCodecParameters->height;
        } else if (pLocalCodecParameters->codec_type == AVMEDIA_TYPE_AUDIO) {
            LOG(INFO) << "Audio Codec: "
                       << pLocalCodecParameters->channels
                       << " channels, sample rate "
                       << pLocalCodecParameters->sample_rate;
        }

        LOG(INFO) << "Codec " << pLocalCodec->name
                  << " ID " << pLocalCodec->id
                  << " bit_rate " << pLocalCodecParameters->bit_rate;
    }

    if (video_stream_index == -1) {
        LOG(ERROR) << "File " << filename << " does not contain a video stream!";
        res = ERROR_CODE;

        goto close_input;
    }

    /* Allocate an AVCodecContext and set its fields to default values. */
    pCodecContext = avcodec_alloc_context3(pCodec);
    if (!pCodecContext) {
        LOG(ERROR) << "Failed to allocated memory for AVCodecContext";
        res = ERROR_CODE;

        goto close_input;
    }

    /* Fill the codec context based on the values from the supplied codec parameters. */
    res = avcodec_parameters_to_context(pCodecContext, pCodecParameters);
    if (res != SUCCESS_CODE) {
        LOG(ERROR) << "Failed to fill codec context";
        res = ERROR_CODE;

        goto free_codec_context;
    }

    /* Initialize the AVCodecContext to use the given AVCodec. */
    res = avcodec_open2(pCodecContext, pCodec, nullptr);
    if (res != SUCCESS_CODE) {
        LOG(ERROR) << "Failed to initialize context to use the given codec";
        res = ERROR_CODE;

        goto free_codec_context;
    }

    /* Allocate an AVPacket and set its fields to default values. */
    pPacket = av_packet_alloc();
    if (pPacket == nullptr) {
        LOG(ERROR) << "Failed to allocate memory for packet";
        res = ERROR_CODE;

        goto free_codec_context;
    }

    pFrame = av_frame_alloc();
    if (pFrame == nullptr) {
        LOG(ERROR) << "Failed to allocate memory for frame";
        res = ERROR_CODE;

        goto free_packet;
    }

    while(av_read_frame(pFormatContext, pPacket) >= 0) {
        if (pPacket->stream_index == video_stream_index) {

            res = decode_packet(model, pPacket, pCodecContext, pFrame);
            if (res != SUCCESS_CODE) {
                av_packet_unref(pPacket);
                break;
            }

            if (--count_of_packets <= 0) break;
        }

        av_packet_unref(pPacket);
    }

free_frame:
    av_frame_free(&pFrame);

free_packet:
    av_packet_free(&pPacket);

free_codec_context:
     avcodec_free_context(&pCodecContext);

close_input:
    avformat_close_input(&pFormatContext);

free_context:
    avformat_free_context(pFormatContext);

    return res;
}

int main(int argv, char** argc) {
    int res = SUCCESS_CODE;
    /* std::string path_to_image; */
    std::string path_to_model;
    std::string path_to_video;

    std::vector<Flag> flag_list = {
        /* Flag("image", &path_to_image, "path of image to be processed"), */
        Flag("model", &path_to_model, "path of model to be processed"),
        Flag("video_file", &path_to_video, "path of video to be processed"),
    };

    std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
    bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
    if (!parse_result) {
        LOG(ERROR) << usage;
        return ERROR_CODE;
    }

    /* if (path_to_image.empty()) { */
    /*     LOG(INFO) << "Path of image is empty!!!"; */
    /* } else { */
    /*     LOG(INFO) << "Path of image: " << path_to_image; */
    /* } */

    if (path_to_model.empty()) {
        LOG(ERROR) << "Path of model is empty!!!";
        return ERROR_CODE;
    }
    LOG(INFO) << "Path of model: " << path_to_model;

    if (path_to_video.empty()) {
        LOG(ERROR) << "Path of video file is empty!!!";
        return ERROR_CODE;
    }
    LOG(INFO) << "Path of video file: " << path_to_video;

    try {
        DetectionModel model(path_to_model);
        res = ffmpeg_proceed(model, path_to_video);
        if (res != SUCCESS_CODE) {
            LOG(ERROR) << "Failed with FFmpeg proceed";
            return ERROR_CODE;
        }
        /* model.Testing(path_to_image); */
    } catch (const std::exception& e) {
        LOG(ERROR) << e.what();
        return ERROR_CODE;
    }

}
