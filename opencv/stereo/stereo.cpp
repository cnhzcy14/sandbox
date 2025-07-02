#include "opencv2/calib3d.hpp"
//测试opencv的stereo match耗时
void test_opencv_stereo_mathch()
{
    std::chrono::time_point<std::chrono::high_resolution_clock> st, et;
    int imgWidth = 1280;
    int imgHeight = 1024;
    int numDisparities = 224;
    int minDisp = 64;
    int winSize=5;
    {
        cv::Mat img_left(imgHeight, imgWidth, CV_8UC1);
        cv::Mat img_right(imgHeight, imgWidth, CV_8UC1);
        cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(numDisparities, 9);
        st = std::chrono::high_resolution_clock::now();
        cv::Mat disp;
        bm->compute(img_left, img_right, disp);
        et = std::chrono::high_resolution_clock::now();
        auto time_span =std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
        std::cout<<"bm time:"<<time_span<<std::endl;
    }

    {
        cv::Mat img_left(imgHeight, imgWidth, CV_8UC3);
        cv::Mat img_right(imgHeight, imgWidth, CV_8UC3);
        img_left = cv::imread("img/IR-L.png", cv::IMREAD_COLOR);
        img_right = cv::imread("img/IR-R.png", cv::IMREAD_COLOR);
        if(img_left.empty() || img_right.empty())
        {
            std::cout<<"------load image failed\n";
            return;
        }
        std::cout<<"img_left:"<<img_left.cols<<","<<img_left.rows<<","<<img_left.channels()<<std::endl;
        std::cout<<"img_right:"<<img_right.cols<<","<<img_right.rows<<","<<img_right.channels()<<std::endl;

        //MODE_SGBM采用5个方向，MODE_HH采用8个方向，MODE_SGBM_3WAY采用3个方向,MODE_HH48个方向加速版本
        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(minDisp, numDisparities, winSize);
        sgbm->setPreFilterCap(63);
        sgbm->setBlockSize(winSize);
        int cn = img_left.channels();
        sgbm->setP1(8 * cn*winSize*winSize);
        sgbm->setP2(32 * cn*winSize*winSize);
        sgbm->setMinDisparity(minDisp);
        sgbm->setNumDisparities(numDisparities);
        sgbm->setUniquenessRatio(10);
        sgbm->setSpeckleWindowSize(100);
        sgbm->setSpeckleRange(2);
        sgbm->setDisp12MaxDiff(1);

        cv::Mat disp;
        sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
        st = std::chrono::high_resolution_clock::now();      
        sgbm->compute(img_left, img_right, disp);
        et = std::chrono::high_resolution_clock::now();
        auto time_span =std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
        std::cout<<"MODE_SGBM_3WAY SGBM time:"<<time_span<<std::endl;
        std::cout<<"disp:"<<disp.cols<<","<<disp.rows<<","<<disp.channels()<<","<<disp.type()<<std::endl;
        cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite("img/disp_3way.png", disp);

        sgbm->setMode(cv::StereoSGBM::MODE_HH4);
        st = std::chrono::high_resolution_clock::now();      
        sgbm->compute(img_left, img_right, disp);
        et = std::chrono::high_resolution_clock::now();
        time_span =std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
        std::cout<<"MODE_HH4 SGBM time:"<<time_span<<std::endl;
        cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite("img/disp_hh4.png", disp);

        sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
        st = std::chrono::high_resolution_clock::now();      
        sgbm->compute(img_left, img_right, disp);
        et = std::chrono::high_resolution_clock::now();
        time_span =std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
        std::cout<<"MODE_SGBM SGBM time:"<<time_span<<std::endl;
        cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite("img/disp_5.png", disp);

        sgbm->setMode(cv::StereoSGBM::MODE_HH);
        st = std::chrono::high_resolution_clock::now();      
        sgbm->compute(img_left, img_right, disp);
        et = std::chrono::high_resolution_clock::now();
        time_span =std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
        std::cout<<"MODE_HH SGBM time:"<<time_span<<std::endl;
        cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite("img/disp_hh.png", disp);
    }

}
