#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
#include <iostream>


int main(int argc, char **argv)
{
	cv::namedWindow("window", cv::WINDOW_NORMAL);
	// cv::setWindowProperty("window", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
	cv::Mat src, dst;
	src = cv::imread(argv[1], 1);
	if (argc != 2 || !src.data)
	{
		printf("No image data \n");
		return -1;
	}

	cv::resize(src, dst, cv::Size(), 1/4.0, 1/4.0);
	cv::imwrite("res.png", dst);

	while (1)
	{
		cv::imshow("window", dst);
		int k = cv::waitKey(1);
		if (k == 27)
			break;
		// eventCallback(&eventData, k);

		// if(eventData.shouldStop)	printf("shouldStop\n");
		// if(eventData.pause)	printf("pause\n");
		// if(eventData.showPointCloud)	printf("showPointCloud\n");
	}
	return 0;
}
