#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>

struct EventData
{
	EventData(): shouldStop(false), pause(false), showPointCloud(false) {}

	bool shouldStop;
	bool pause;
	bool showPointCloud;
};

static void eventCallback(void* eventData, char key)
{
	EventData* data = static_cast<EventData*>(eventData);

	if (key == 'a')
	{
		data->shouldStop = true;
	}
	else if (key == 's')
	{
		data->pause = !data->pause;
	}
	else if (key == 'p')
	{
		data->showPointCloud = !data->showPointCloud;
	}
}


int main(int argc, char **argv)
{
	cv::namedWindow("window", cv::WINDOW_NORMAL);
	// cv::setWindowProperty("window", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
	cv::Mat src, i420, mono,dst;
	src = cv::imread(argv[1], 1);
	if (argc != 2 || !src.data)
	{
		printf("No image data \n");
		return -1;
	}


	// cv::circle(src, cv::Point2f(100, 100), 3, cv::Scalar(0,0,255), 3, cv::LINE_AA);
	cv::cvtColor(src, i420, cv::COLOR_BGR2YUV_I420);
	
	mono = cv::Mat(src.rows, src.cols, CV_8UC1, i420.data);
	
	cv::cvtColor(i420, dst, cv::COLOR_YUV2GRAY_I420);


	while (1)
	{
		cv::imshow("window", mono);
		int k = cv::waitKey(1);
		if(k == 27) break;
		// eventCallback(&eventData, k);

		// if(eventData.shouldStop)	printf("shouldStop\n");
		// if(eventData.pause)	printf("pause\n");
		// if(eventData.showPointCloud)	printf("showPointCloud\n");
	}
	return 0;
}
