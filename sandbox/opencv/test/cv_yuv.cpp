#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

const int width = 1280;
const int height = 1440;
const int frameSize = width * height;

cv::Mat imreadYUV(const cv::String &filename)
{
	FILE* fin = fopen(filename.c_str(), "rb+");
	unsigned char* bufYUV = new unsigned char[frameSize];
	size_t ret = fread(bufYUV, frameSize*sizeof(unsigned char), 1, fin);
	cv::Mat imgYUV, imgBGR;
	imgYUV.create(height, width, CV_8UC1);
	memcpy(imgYUV.data, bufYUV, frameSize*sizeof(unsigned char));
	// cv::cvtColor(imgYUV, imgBGR, cv::COLOR_YUV2BGR_NV12);
	fclose(fin);
	delete bufYUV;
	return imgYUV;
}

int main(int argc, char **argv)
{
	cv::namedWindow("window", cv::WINDOW_NORMAL);
	// cv::setWindowProperty("window", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
	cv::Mat src, left, right;
	// src = cv::imread(argv[1], 1);
	src = imreadYUV(argv[1]);
	left = src(cv::Rect(0, 720, 1280, 720));
	right = src(cv::Rect(0, 0, 1280, 720));

	if (argc != 2 || !src.data)
	{
		printf("No image data \n");
		return -1;
	}



	imwrite("0.png", left);
	imwrite("1.png", right);

	while (1)
	{
		cv::imshow("window", left);
		int k = cv::waitKey(1);
		if (k == 27)
			break;
	}
	return 0;
}
