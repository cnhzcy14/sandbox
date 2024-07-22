/*
* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/OpticalFlowDense.h>

#include <iostream>
#include <sstream>

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

static void ProcessMotionVector(VPIImage mvImg, cv::Mat &outputImage)
{
    // Lock the input image to access it from CPU
    VPIImageData mvData;
    CHECK_STATUS(vpiImageLockData(mvImg, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &mvData));

    // Create a cv::Mat that points to the input image data
    cv::Mat mvImage;
    CHECK_STATUS(vpiImageDataExportOpenCVMat(mvData, &mvImage));

    // Convert S10.5 format to float
    cv::Mat flow(mvImage.size(), CV_32FC2);
    mvImage.convertTo(flow, CV_32F, 1.0f / (1 << 5));

    // Image not needed anymore, we can unlock it.
    CHECK_STATUS(vpiImageUnlock(mvImg));

    // Create an image where the motion vector angle is
    // mapped to a color hue, and intensity is proportional
    // to vector's magnitude.
    cv::Mat magnitude, angle;
    {
        cv::Mat flowChannels[2];
        split(flow, flowChannels);
        cv::cartToPolar(flowChannels[0], flowChannels[1], magnitude, angle, true);
    }

    float clip = 5;
    cv::threshold(magnitude, magnitude, clip, clip, cv::THRESH_TRUNC);

    // build hsv image
    cv::Mat _hsv[3], hsv, bgr;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude / clip; // intensity must vary from 0 to 1
    merge(_hsv, 3, hsv);

    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    bgr.convertTo(outputImage, CV_8U, 255.0);
}

int main(int argc, char *argv[])
{
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    cv::Mat cvPrevFrame, cvCurFrame;

    // VPI objects that will be used
    VPIStream stream         = NULL;
    VPIImage imgPrevFramePL  = NULL;
    VPIImage imgPrevFrameTmp = NULL;
    VPIImage imgPrevFrameBL  = NULL;
    VPIImage imgCurFramePL   = NULL;
    VPIImage imgCurFrameTmp  = NULL;
    VPIImage imgCurFrameBL   = NULL;
    VPIImage imgMotionVecBL  = NULL;
    VPIPayload payload       = NULL;

    int retval = 0;

    try
    {
        if (argc != 4)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <nvenc> <input_video> <low|medium|high>");
        }

        // Parse input parameters
        std::string strBackend    = argv[1];
        std::string strInputVideo = argv[2];
        std::string strQuality    = argv[3];

        VPIOpticalFlowQuality quality;
        if (strQuality == "low")
        {
            quality = VPI_OPTICAL_FLOW_QUALITY_LOW;
        }
        else if (strQuality == "medium")
        {
            quality = VPI_OPTICAL_FLOW_QUALITY_MEDIUM;
        }
        else if (strQuality == "high")
        {
            quality = VPI_OPTICAL_FLOW_QUALITY_HIGH;
        }
        else
        {
            throw std::runtime_error("Unknown quality provided");
        }

        VPIBackend backend;
        if (strBackend == "nvenc")
        {
            backend = VPI_BACKEND_NVENC;
        }
        else
        {
            throw std::runtime_error("Backend '" + strBackend + "' not recognized, it must be nvenc.");
        }

        // Load the input video
        cv::VideoCapture invid;
        if (!invid.open(strInputVideo))
        {
            throw std::runtime_error("Can't open '" + strInputVideo + "'");
        }

        // Create the stream where processing will happen. We'll use user-provided backend
        // for Optical Flow, and CUDA/VIC for image format conversions.
        CHECK_STATUS(vpiStreamCreate(backend | VPI_BACKEND_CUDA | VPI_BACKEND_VIC, &stream));

        // Fetch the first frame
        if (!invid.read(cvPrevFrame))
        {
            throw std::runtime_error("Cannot read frame from input video");
        }

        // Create the previous and current frame wrapper using the first frame. This wrapper will
        // be set to point to every new frame in the main loop.
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvPrevFrame, 0, &imgPrevFramePL));
        CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvPrevFrame, 0, &imgCurFramePL));

        // Define the image formats we'll use throughout this sample.
        VPIImageFormat imgFmt   = VPI_IMAGE_FORMAT_NV12_ER;
        VPIImageFormat imgFmtBL = VPI_IMAGE_FORMAT_NV12_ER_BL;

        int32_t width  = cvPrevFrame.cols;
        int32_t height = cvPrevFrame.rows;
        const int32_t gridSize = 4;
        int32_t numLevels = 1;

        // Create Dense Optical Flow payload to be executed on the given backend
        CHECK_STATUS(vpiCreateOpticalFlowDense(backend, width, height, imgFmtBL, &gridSize, numLevels, quality, &payload));

        // The Dense Optical Flow on NVENC backend expects input to be in block-linear format.
        // Since Convert Image Format algorithm doesn't currently support direct BGR
        // pitch-linear (from OpenCV) to NV12 block-linear conversion, it must be done in two
        // passes, first from BGR/PL to NV12/PL using CUDA, then from NV12/PL to NV12/BL using VIC.
        // The temporary image buffer below will store the intermediate NV12/PL representation.
        CHECK_STATUS(vpiImageCreate(width, height, imgFmt, 0, &imgPrevFrameTmp));
        CHECK_STATUS(vpiImageCreate(width, height, imgFmt, 0, &imgCurFrameTmp));

        // Now create the final block-linear buffer that'll be used as input to the
        // algorithm.
        CHECK_STATUS(vpiImageCreate(width, height, imgFmtBL, 0, &imgPrevFrameBL));
        CHECK_STATUS(vpiImageCreate(width, height, imgFmtBL, 0, &imgCurFrameBL));

        // Motion vector image width and height, align to be multiple of 4
        int32_t mvWidth  = (width + 3) / 4;
        int32_t mvHeight = (height + 3) / 4;

        // The output video will be heatmap of motion vector image
        int fourcc = cv::VideoWriter::fourcc('M', 'P', 'E', 'G');
        double fps = invid.get(cv::CAP_PROP_FPS);

        cv::VideoWriter outVideo("denseoptflow_mv_" + strBackend + ".mp4", fourcc, fps, cv::Size(mvWidth, mvHeight));
        if (!outVideo.isOpened())
        {
            throw std::runtime_error("Can't create output video");
        }

        // Create the output motion vector buffer
        CHECK_STATUS(vpiImageCreate(mvWidth, mvHeight, VPI_IMAGE_FORMAT_2S16_BL, 0, &imgMotionVecBL));

        // First convert the first frame to NV12_BL. It'll be used as previous frame when the algorithm is called.
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgPrevFramePL, imgPrevFrameTmp, nullptr));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC, imgPrevFrameTmp, imgPrevFrameBL, nullptr));

        // Create a output image which holds the rendered motion vector image.
        cv::Mat mvOutputImage;

        // Fetch a new frame until video ends
        int idxFrame = 1;
        while (invid.read(cvCurFrame))
        {
            printf("Processing frame %d\n", idxFrame++);
            // Wrap frame into a VPIImage, reusing the existing imgCurFramePL.
            CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgCurFramePL, cvCurFrame));

            // Convert current frame to NV12_BL format
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgCurFramePL, imgCurFrameTmp, nullptr));
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_VIC, imgCurFrameTmp, imgCurFrameBL, nullptr));

            CHECK_STATUS(
                vpiSubmitOpticalFlowDense(stream, backend, payload, imgPrevFrameBL, imgCurFrameBL, imgMotionVecBL));

            // Wait for processing to finish.
            CHECK_STATUS(vpiStreamSync(stream));

            // Render the resulting motion vector in the output image
            ProcessMotionVector(imgMotionVecBL, mvOutputImage);

            // Save to output video
            outVideo << mvOutputImage;

            // Swap previous frame and next frame
            std::swap(cvPrevFrame, cvCurFrame);
            std::swap(imgPrevFramePL, imgCurFramePL);
            std::swap(imgPrevFrameBL, imgCurFrameBL);
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    // Destroy all resources used
    vpiStreamDestroy(stream);
    vpiPayloadDestroy(payload);

    vpiImageDestroy(imgPrevFramePL);
    vpiImageDestroy(imgPrevFrameTmp);
    vpiImageDestroy(imgPrevFrameBL);
    vpiImageDestroy(imgCurFramePL);
    vpiImageDestroy(imgCurFrameTmp);
    vpiImageDestroy(imgCurFrameBL);
    vpiImageDestroy(imgMotionVecBL);

    return retval;
}
