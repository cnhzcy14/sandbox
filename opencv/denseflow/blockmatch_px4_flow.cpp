//=============================================================================
//
// farneback_flow.cpp
// Main file for testing OpenCV GPU Farnerback Optical Flow
// Author: Pablo F. Alcantarilla
// Institution: ALCoV, Université d'Auvergne
// Date: 14/06/2012
// Email: pablofdezalc@gmail.com
//=============================================================================

#include "blockmatch_px4_flow.h"

// Namespaces
using namespace std;
using namespace cv;

// Some global variables for the optical flow
const uint32_t searchSize = 6;
const uint32_t flowFeatureThreshold = 30;
const uint32_t flowValueThreshold = 3000;

//******************************************************************************
//******************************************************************************

/** Main Function */
int main( int argc, char *argv[] )
{
    // Variables for CUDA Farneback Optical flow
    // GpuMat frame0GPU, frame1GPU, uGPU, vGPU, buffer;
    Mat frame0_rgb_, frame1_rgb_, frame0_rgb, frame1_rgb, frame0, frame1;
    Mat imgU, imgV;
    Mat motion_flow, flow_rgb;
    int nframes = 0, width = 0, height = 0;
    float flowX, flowY;

    // Variables for measuring computation times
    struct timeval tod1;
    double t1 = 0.0, t2 = 0.0, tdflow = 0.0, tvis = 0.0;

    // Check input arguments
    if( argc != 2 )
    {
        cout << "Error introducing input arguments!!" << endl;
        cout << "Number of input arguments: " << argc << endl;
        cout << "The format is: ./farneback_flow video_file.avi" << endl;
        return -1;
    }

    // Show CUDA information
    // cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    // Create OpenCV windows
    namedWindow("Dense Flow",WINDOW_NORMAL);
    namedWindow("Motion Flow",WINDOW_NORMAL);



    // Open the video file
    VideoCapture cap(argv[1]);
    if( cap.isOpened() == 0 )
    {
        return -1;
    }

    cap >> frame1_rgb_;

    frame1_rgb = cv::Mat(Size(frame1_rgb_.cols,frame1_rgb_.rows),CV_8UC3);
    width = frame1_rgb.cols;
    height = frame1_rgb.rows;
    frame1_rgb_.copyTo(frame1_rgb);

    // Create the optical flow object
    std::unique_ptr<PX4Flow> dflow(new PX4Flow(width, searchSize, flowFeatureThreshold, flowValueThreshold));

    // Allocate memory for the images
    frame0_rgb = cv::Mat(Size(width,height),CV_8UC3);
    frame0 = cv::Mat(Size(width,height),CV_8UC1);
    frame1 = cv::Mat(Size(width,height),CV_8UC1);
    flow_rgb = cv::Mat(Size(width,height),CV_8UC3);
    motion_flow = cv::Mat(Size(width,height),CV_8UC3);

    // Convert the image to grey and float
    cv::cvtColor(frame1_rgb,frame1,COLOR_BGR2GRAY);

    while( frame1.empty() == false )
    {
        if( nframes >= 1 )
        {
            gettimeofday(&tod1,NULL);
            auto t0 = std::chrono::high_resolution_clock::now();

            // Upload images to the GPU
            // frame1GPU.upload(frame1);
            // frame0GPU.upload(frame0);

            // Do the dense optical flow
            dflow->compute_flow((uint8_t *)frame0.data, (uint8_t *)frame1.data, 0, 0, 0, &flowX, &flowY);

            // uGPU.download(imgU);
            // vGPU.download(imgV);


		    auto t1 = std::chrono::high_resolution_clock::now();
		    auto dt = 1.e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
		    std::cout << "Block Matching Time : " << 1000.0f / dt << "fps " << dt << "ms " << std::endl;
        }

        if( nframes >= 1 )
        {
            std::cout << "=====x: " << flowX << "======y: " << flowY << std::endl;
            // gettimeofday(&tod1,NULL);
            // t1 = tod1.tv_sec + tod1.tv_usec / 1000000.0;

            // // Draw the optical flow results
            // drawColorField(imgU,imgV,flow_rgb);

            // frame1_rgb.copyTo(motion_flow);
            // drawMotionField(imgU,imgV,motion_flow,15,15,.0,1.0,CV_RGB(0,255,0));

            // // Visualization
            // imshow("Dense Flow",flow_rgb);
            // imshow("Motion Flow",motion_flow);

            // waitKey(3);

            // gettimeofday(&tod1,NULL);
            // t2 = tod1.tv_sec + tod1.tv_usec / 1000000.0;
            // tvis = 1000.0*(t2-t1);
        }

        // Set the information for the previous frame
        frame1_rgb.copyTo(frame0_rgb);
        cv::cvtColor(frame0_rgb,frame0,COLOR_BGR2GRAY);

        // Read the next frame
        nframes++;
        cap >> frame1_rgb_;

        if( frame1_rgb_.empty() == false )
        {
            frame1_rgb_.copyTo(frame1_rgb);
            cv::cvtColor(frame1_rgb,frame1,COLOR_BGR2GRAY);
        }
        else
        {
            break;
        }

        cout << "Frame Number: " << nframes << endl;
        cout << "Time Dense Flow: " << tdflow << endl;
        cout << "Time Visualization: " << tvis << endl << endl;
    }

    // Destroy the windows
    destroyAllWindows();

    return 0;
}

