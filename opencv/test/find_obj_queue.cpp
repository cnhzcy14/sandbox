#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
const char *keys =
	"{ help h | | Print help message. }"
	"{ input1 | box.png | Path to input image 1. }"
	"{ input2 | box_in_scene.png | Path to input image 2. }";
int main(int argc, char *argv[])
{
	cv::namedWindow("window", cv::WINDOW_NORMAL);
	cv::setWindowProperty("window", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
	CommandLineParser parser(argc, argv, keys);

	cv::VideoCapture capture;
	capture.open(std::string(argv[1]));

	Mat img_object;
	// Mat img_object = imread(argv[1], 1);
	Mat img_scene = imread(argv[2], 1);

	if (img_scene.empty())
	{
		cout << "Could not open or find the image!\n"
			 << endl;
		parser.printMessage();
		return -1;
	}

	int i = 0;
	while (1)
	{
		capture >> img_object;
		if (img_object.empty())
			break;

		//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
		int minHessian = 400;
		Ptr<SURF> detector = SURF::create(minHessian, 4, 3, false, false);
		// Ptr<SIFT> detector = SIFT::create();
		std::vector<KeyPoint> keypoints_object, keypoints_scene;
		Mat descriptors_object, descriptors_scene;
		detector->detectAndCompute(img_object, noArray(), keypoints_object, descriptors_object);
		detector->detectAndCompute(img_scene, noArray(), keypoints_scene, descriptors_scene);
		//-- Step 2: Matching descriptor vectors with a FLANN based matcher
		// Since SURF is a floating-point descriptor NORM_L2 is used
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		std::vector<std::vector<DMatch>> knn_matches;
		matcher->knnMatch(descriptors_object, descriptors_scene, knn_matches, 2);
		//-- Filter matches using the Lowe's ratio test
		const float ratio_thresh = 0.75f;
		std::vector<DMatch> good_matches;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			{
				good_matches.push_back(knn_matches[i][0]);
			}
		}
		//-- Draw matches
		Mat img_matches;
		cv::drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
					Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//-- Localize the object
		std::vector<Point2f> obj;
		std::vector<Point2f> scene;
		for (size_t i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
		}
		Mat H = findHomography(obj, scene, RANSAC);
		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = Point2f(0, 0);
		obj_corners[1] = Point2f((float)img_object.cols, 0);
		obj_corners[2] = Point2f((float)img_object.cols, (float)img_object.rows);
		obj_corners[3] = Point2f(0, (float)img_object.rows);
		std::vector<Point2f> scene_corners(4);
		cv::perspectiveTransform(obj_corners, scene_corners, H);
		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		cv::line(img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
			 scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
		cv::line(img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
			 scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
		cv::line(img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
			 scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
		cv::line(img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
			 scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
		//-- Show detected matches

		cv::putText(img_matches,						 // target image
					std::to_string(i),			 // text
					cv::Point(10, img_matches.rows / 2), // top-left position
					cv::FONT_HERSHEY_DUPLEX,
					2.0,
					CV_RGB(118, 185, 0), // font color
					2);
		cv::imshow("window", img_matches);
		int k = cv::waitKey(1);
		if (k == 'q' || k == 27)
			break;
		
		i++;
	}

	return 0;
}
