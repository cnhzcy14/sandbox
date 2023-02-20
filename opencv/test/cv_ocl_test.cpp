#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

int main(int argc, char **argv)
{

    if (!cv::ocl::haveOpenCL())
        std::cout << "OpenCL is not avaiable..." << std::endl;
    else
        std::cout << "OpenCL is AVAILABLE! :) " << std::endl; // this is the output
    
    cv::ocl::Context context = cv::ocl::Context::getDefault();
    std::cout << context.ndevices() << " GPU devices are detected." << std::endl;
    for (int i = 0; i < context.ndevices(); i++)
    {
        cv::ocl::Device device = context.device(i);
        std::cout << "name:              " << device.name() << std::endl;
        std::cout << "available:         " << device.available() << std::endl;
        std::cout << "imageSupport:      " << device.imageSupport() << std::endl;
        std::cout << "OpenCL_C_Version:  " << device.OpenCL_C_Version() << std::endl;
        std::cout << std::endl;
    } // this works & i can see my video card name & opencl version
    cv::ocl::Device(context.device(0));

    cv::ocl::setUseOpenCL(true);

    cv::Mat image;
    cv::UMat img, gray;

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened())
    {
        std::cerr << "Could not open the input video: " << argv[1] << std::endl;
        return 1;
    }

    while (1)
    {
        cap.read(image);
        if (image.empty())
        {
            break;
        }
        image.copyTo(img);
        // img = image.getUMat(cv::ACCESS_RW);
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        std::cout << "=======cvtColor" << std::endl;
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 1.5);
        std::cout << "=======GaussianBlur" << std::endl;
        cv::Canny(gray, gray, 0, 50);
        std::cout << "=======Canny" << std::endl;
		cv::imshow("window", gray);
		int k = cv::waitKey(1);
		if(k == 27) break;
    }

    // cv::imread(argv[1], cv::IMREAD_COLOR).copyTo(img);
    // img = cv::imread(argv[1], cv::IMREAD_COLOR);

    return 0;
}
