#include <iostream>
#include <opencv2/opencv.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/videoio.hpp>
// #include <opencv2/video.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    const string about =
        "This sample demonstrates Lucas-Kanade Optical Flow calculation.\n"
        "The example file can be downloaded from:\n"
        "  https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4";
    const string keys =
        "{ h help |      | print this help message }"
        "{ @image | vtest.avi | path to image file }";
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    string filename = samples::findFile(parser.get<string>("@image"));
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

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

    VideoCapture capture;
    capture.open(filename + "/%08d.png");
    if (!capture.isOpened())
    {
        // error in opening the video input
        cerr << "Unable to open file!" << endl;
        return 0;
    }

    // Create some random colors
    vector<Scalar> colors;
    RNG rng;
    for (int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r, g, b));
    }

    UMat old_frame, old_gray;
    vector<Point2f> p0, p1;
    std::vector<cv::KeyPoint> new_features;

    // Take first frame and find corners in it
    capture >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);

    cv::Ptr<cv::FastFeatureDetector> feature_detector_ptr_ = cv::FastFeatureDetector::create(30, true);
    feature_detector_ptr_->detect(old_gray, new_features);
    // goodFeaturesToTrack(old_gray, p0, 300, 0.01, 7, Mat(), 7, false, 0.04);
    for (const auto &feature : new_features)
    {
        p0.push_back(feature.pt);
    }

    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    while (true)
    {
        UMat frame, frame_gray;

        capture >> frame;
        if (frame.empty())
            break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // calculate optical flow
        vector<uchar> status;
        vector<float> err;

        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(21, 21), 1, criteria);
        std::cout << "++++++++++++" << std::endl;

        vector<Point2f> good_new;
        for (uint i = 0; i < p0.size(); i++)
        {
            // Select good points
            if (status[i] == 1)
            {
                good_new.push_back(p1[i]);
                // draw the tracks
                line(mask, p1[i], p0[i], colors[i], 2);
                circle(frame, p1[i], 5, colors[i], -1);
            }
        }

        Mat img;
        add(frame, mask, img);
        imshow("Frame", img);
        int keyboard = waitKey(5);
        if (keyboard == 'q' || keyboard == 27)
            break;

        // Now update the previous frame and previous points
        old_gray = frame_gray.clone();
        p0 = good_new;
    }
}
