#include <opencv2/core/version.hpp>

#if CV_MAJOR_VERSION >= 3
#include <opencv2/imgcodecs.hpp>
#else
#include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Image.h>
#include <vpi/LensDistortionModels.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Remap.h>

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

static void PrintUsage(const char *progname, std::ostream &out)
{
    out << "Usage: " << progname << " <-c W,H> [-s win] <image1> [image2] [image3] ...\n"
        << " where,\n"
        << " W,H\tcheckerboard with WxH squares\n"
        << " win\tsearch window width around checkerboard vertex used\n"
        << "\tin refinement, default is 0 (disable refinement)\n"
        << " imageN\tinput images taken with a fisheye lens camera" << std::endl;
}

static char *my_basename(char *path)
{
#ifdef WIN32
    char *name = strrchr(path, '\\');
#else
    char *name = strrchr(path, '/');
#endif
    if (name != NULL)
    {
        return name;
    }
    else
    {
        return path;
    }
}

struct Params
{
    cv::Size vtxCount;                // Number of internal vertices the checkerboard has
    int searchWinSize;                // search window size around the checkerboard vertex for refinement.
    std::vector<const char *> images; // input image names.
};

static Params ParseParameters(int argc, char *argv[])
{
    Params params = {};

    cv::Size cbSize;

    for (int i = 1; i < argc; ++i)
    {
        if (argv[i][0] == '-')
        {
            if (strlen(argv[i] + 1) == 1)
            {
                switch (argv[i][1])
                {
                case 'h':
                    PrintUsage(my_basename(argv[0]), std::cout);
                    return {};

                default:
                    throw std::invalid_argument(std::string("Option -") + (argv[i] + 1) + " not recognized");
                }
            }
            else
            {
                throw std::invalid_argument(std::string("Option -") + (argv[i] + 1) + " not recognized");
            }
        }
        else
        {
            params.images.push_back(argv[i]);
        }
    }

    if (params.images.empty())
    {
        throw std::invalid_argument("At least one image must be defined");
    }

    return params;
}

int main(int argc, char *argv[])
{
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    cv::Mat cvImage;

    // VPI objects that will be used
    VPIStream stream = NULL;
    VPIPayload remap = NULL;
    VPIImage tmpIn = NULL, tmpOut = NULL;
    VPIImage vimg = nullptr;
    VPIImage vdst = nullptr;

    int retval = 0;

    try
    {
        // First parse command line paramers
        Params params = ParseParameters(argc, argv);
        if (params.images.empty()) // user just wanted the help message?
        {
            return 0;
        }


        // Store image size. All input images must have same size.
        cv::Size imgSize = {};
        cv::Size dstSize{640, 235};

        for (unsigned i = 0; i < params.images.size(); ++i)
        {
            // Load input image and do some sanity check
            cv::Mat img = cv::imread(params.images[i]);
            if (img.empty())
            {
                throw std::runtime_error("Can't read " + std::string(params.images[i]));
            }

            if (imgSize == cv::Size{})
            {
                imgSize = img.size();
            }
            else if (imgSize != img.size())
            {
                throw std::runtime_error("All images must have same size");
            }
        }

        // Now use VPI to undistort the input images:

        // Allocate a dense map.
        VPIWarpMap map = {};
        map.grid.numHorizRegions = 1;
        map.grid.numVertRegions = 1;
        map.grid.regionWidth[0] = dstSize.width;
        map.grid.regionHeight[0] = dstSize.height;
        map.grid.horizInterval[0] = 1;
        map.grid.vertInterval[0] = 1;
        CHECK_STATUS(vpiWarpMapAllocData(&map));

        // Load the precomputed map from disk.
        std::vector<float> xVec(dstSize.width * dstSize.height), yVec(dstSize.width * dstSize.height);
        FILE *fx = fopen("xmap_640.f32", "rb");
        FILE *fy = fopen("ymap_640.f32", "rb");
        if (!fx || !fy)
        {
            std::cerr << "cannot open *.f32\n";
            return 1;
        }
        size_t ret;
        ret = fread(xVec.data(), sizeof(float), dstSize.width * dstSize.height, fx); 
        if (ret != dstSize.width * dstSize.height)
        {
            std::cerr << "error reading xmap.f32\n";
            fclose(fx);
            return 1;
        }
        ret = fread(yVec.data(), sizeof(float), dstSize.width * dstSize.height, fy); 
        if (ret != dstSize.width * dstSize.height)
        {
            std::cerr << "error reading ymap.f32\n";
            fclose(fy);
            return 1;
        }
        fclose(fx);
        fclose(fy);

        std::cout << "===========: " << map.numHorizPoints << " x " << map.numVertPoints << " points.\n";

        VPIKeypointF32 *mapData = (VPIKeypointF32 *)map.keypoints;
        for (int i = 0; i < map.numHorizPoints; ++i)
        {
            for (int j = 0; j < dstSize.height; ++j)
            {
                if(i >= map.numHorizPoints)
                {
                    break;
                }

                mapData[j * map.numHorizPoints + i].x = xVec[j * dstSize.width + i];
                mapData[j * map.numHorizPoints + i].y = yVec[j * dstSize.width + i];
            }
        }


        // Create the Remap payload for undistortion given the map generated above.
        CHECK_STATUS(vpiCreateRemap(VPI_BACKEND_CUDA, &map, &remap));

        // Now that the remap payload is created, we can destroy the warp map.
        vpiWarpMapFreeData(&map);

        // Create a stream where operations will take place. We're using CUDA
        // processing.
        CHECK_STATUS(vpiStreamCreate(VPI_BACKEND_CUDA, &stream));

        // Temporary input and output images in NV12 format.
        CHECK_STATUS(vpiImageCreate(imgSize.width, imgSize.height, VPI_IMAGE_FORMAT_BGR8, 0, &tmpIn));
        CHECK_STATUS(vpiImageCreate(imgSize.width, imgSize.height, VPI_IMAGE_FORMAT_BGR8, 0, &tmpOut));

        // For each input image,
        for (unsigned i = 0; i < params.images.size(); ++i)
        {
            // Read it from disk.
            cvImage = cv::imread(params.images[i]);
            assert(!cvImage.empty());
            cv::Mat dst(dstSize.height, dstSize.width, CV_8UC3);

            // Wrap it into a VPIImage
            if (vimg == nullptr)
            {
                // Now create a VPIImage that wraps it.
                CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImage, 0, &vimg));
                CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(dst, 0, &vdst));
            }
            else
            {
                CHECK_STATUS(vpiImageSetWrappedOpenCVMat(vimg, cvImage));
                CHECK_STATUS(vpiImageSetWrappedOpenCVMat(vdst, dst));
            }

            // Convert BGR -> NV12
            // CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, vimg, tmpIn, NULL));

            // Undistorts the input image.
            CHECK_STATUS(vpiSubmitRemap(stream, VPI_BACKEND_CUDA, remap, vimg, vdst, VPI_INTERP_CATMULL_ROM,
                                        VPI_BORDER_ZERO, 0));

            // Convert the result NV12 back to BGR, writing back to the input image.
            // CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, tmpOut, vdst, NULL));

            // Wait until conversion finishes.
            CHECK_STATUS(vpiStreamSync(stream));

            // Since vimg is wrapping the OpenCV image, the result is already there.
            // We just have to save it to disk.
            char buf[64];
            snprintf(buf, sizeof(buf), "undistort_%03d.jpg", i);
            imwrite(buf, dst);
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        PrintUsage(my_basename(argv[0]), std::cerr);

        retval = 1;
    }

    vpiStreamDestroy(stream);
    vpiPayloadDestroy(remap);
    vpiImageDestroy(tmpIn);
    vpiImageDestroy(tmpOut);
    vpiImageDestroy(vimg);
    vpiImageDestroy(vdst);

    return retval;
}
