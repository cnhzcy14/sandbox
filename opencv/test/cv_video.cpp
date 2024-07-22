#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char **argv)
{
	cv::namedWindow("window", cv::WINDOW_NORMAL);
	// cv::setWindowProperty("window", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

	cv::VideoCapture capture;
    capture.open(std::string(argv[1]));
	
	cv::Mat src, dst;
	// src = cv::imread(argv[1], 1);
	// if (argc != 2 || !src.data)
	// {
	// 	printf("No image data \n");
	// 	return -1;
	// }


	while (1)
	{
        capture >> src;
        if (src.empty())
            break;


		cv::imshow("window", src);
		int k = cv::waitKey(1);
		if (k == 'q' || k == 27)
			break;
		// eventCallback(&eventData, k);

		// if(eventData.shouldStop)	printf("shouldStop\n");
		// if(eventData.pause)	printf("pause\n");
		// if(eventData.showPointCloud)	printf("showPointCloud\n");
	}
	return 0;
}
