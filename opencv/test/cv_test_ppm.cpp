#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
// #include <opencv2/core/mat.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
#include <iostream>
#include <omp.h>

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

	// FILE *irImg;
	// irImg = fopen("/mnt/appdata/resource/Jobs/smart_camera_bs/irImg.pgm", "wb");
	// fprintf(irImg, "P5\n");
	// fprintf(irImg, "%d %d\n", irImages[k].width, irImages[k].height);
	// fprintf(irImg, "255\n");
	// fwrite((void *)irImages[k].imageData, irImages[k].width * irImages[k].height, 1, irImg);
	// fclose(irImg);

	// open the file to read just the header reading
	

	int ret;
	char pSix[10];
	int width;
	int height;

	FILE* fr = fopen("j00.ppm", "rb");
	ret = fscanf(fr, "%s\n", pSix);
	ret = fscanf(fr, "%d %d\n", &width, &height);
	ret = fscanf(fr, "%s\n", pSix);
	src =  cv::Mat::zeros(height, width, CV_8UC3);
	ret = fread((void *)src.data, width * height * 3, 1, fr);



	// src = cv::imread(argv[1], 1);
	// if (argc != 2 || !src.data)
	// {
	// 	printf("No image data \n");
	// 	return -1;
	// }

	// src.convertTo(dst, CV_32FC1, 1/255.0f);
	// float *dst_ptr = (float *)dst.data;
	// uint8_t *src_ptr = (uint8_t *)src.data;
	// #pragma omp parallel for simd
    // for (size_t i = 0; i < dst.cols * dst.rows; ++i)
    // {
	// 	src_ptr[i] = dst_ptr[i] > 0.5 ? 255 : 0;
    // }

	// std::vector<int> compression_params;
	// compression_params.push_back(cv::IMWRITE_PXM_BINARY);
	// compression_params.push_back(1);

	// writing as ppm image
	// cv::imwrite("m01.ppm", src, compression_params);

	while (1)
	{
		cv::imshow("window", src);
		int k = cv::waitKey(1);
		if (k == 27)
			break;
	}
	return 0;
}
