#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
#include <iostream>

struct EventData
{
	EventData() : shouldStop(false), pause(false), showPointCloud(false) {}

	bool shouldStop;
	bool pause;
	bool showPointCloud;
};

static void eventCallback(void *eventData, char key)
{
	EventData *data = static_cast<EventData *>(eventData);

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
	cv::Mat src, dst;
	src = cv::imread(argv[1], 1);
	if (argc != 2 || !src.data)
	{
		printf("No image data \n");
		return -1;
	}

	// Check the relationship between InputArray, Mat and std::vector
	{

		float d[17 * 2];
		cv::Mat test(17, 2, CV_32FC1, d);
		for (int i = 0; i < 17 * 2; i++)
		{
			((float *)test.data)[i] = 0.1 * i;
		}
		// std::cout << " ==== " << test << std::endl;

		cv::Mat another = ((cv::InputArray)test).getMat();
		// std::cout << " **** " << another << std::endl;
		another.at<float>(16, 0) = 0;
		another.at<float>(16, 1) = -1e4;

		// cv::OutputArray same = cv::OutputArray: create(test.size(), test.type(), -1, true);
		// same.create(test.size(), test.type(), -1, true);

		// cv::Mat over = (another>=0);
		// std::cout << " .... " << over << std::endl;

		cv::Point2f *data = (cv::Point2f *)test.data;
		std::vector<cv::Point2f> input;
		std::vector<cv::Point2f> output;
		input.assign(data, data + 17);
		input[16].x = -1;
		// for(int i=0; i<input.size(); i++)
		// {
		// 	std::cout << input[i] << std::endl;
		// }

		// std::cout << " ---- " << input << std::endl;

		for (int i = 0; i < 17 * 2; i++)
		{
			std::cout << d[i] << std::endl;
		}
		std::cout << sizeof(output) << "=---=" << sizeof(input) << std::endl;
	}

	cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
	EventData eventData;
	std::cout << dst.cols << " " << dst.rows << std::endl;
	dst.adjustROI(-50, -50, -50, -50);
	std::cout << dst.cols << " " << dst.rows << std::endl;
	dst.adjustROI(50, 50, 50, 50);

	int winsize = 32;

	int row = static_cast<int>(round(40) / winsize);
	int col = static_cast<int>(round(40) / winsize);

	int offset_x = 0, offset_y = 0;
	// offset_x = (src.cols - src.cols/winsize*winsize)/2;
	// offset_y = (src.rows - src.rows/winsize*winsize)/2;
	std::cout << ceil((float)src.cols / winsize) << " ===== " << ceil((float)src.rows / winsize) << std::endl;
	// std::cout << static_cast<int>(round(479.0) / winsize) << " ----- " << static_cast<int>(round(43.5) / winsize) << std::endl;

	for (int i = 0; i <= src.cols / winsize; i++)
	{
		cv::line(src, cv::Point(winsize * i + offset_x, 0), cv::Point(winsize * i + offset_x, src.rows), cv::Scalar(0, 255, 0));
	}
	for (int i = 0; i <= src.rows / winsize; i++)
	{
		cv::line(src, cv::Point(0, winsize * i + offset_y), cv::Point(src.cols, winsize * i + offset_y), cv::Scalar(0, 255, 0));
	}

	cv::Mat mask = cv::Mat::zeros(src.rows, src.cols, src.type());
	std::vector<cv::Point> pts;
	pts.push_back(cv::Point(336,0));
	pts.push_back(cv::Point(480,0));
	pts.push_back(cv::Point(480,192));
	pts.push_back(cv::Point(336,192));

	cv::fillConvexPoly(mask, pts, cv::Scalar(0xff, 0xff, 0xff));

	cv::bitwise_and	(src,mask,src);

	while (1)
	{
		cv::imshow("window", src);
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
