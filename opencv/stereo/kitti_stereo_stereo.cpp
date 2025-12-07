#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

struct KITTIParams {
    Mat K_00, K_01, K_02;  // 内参矩阵
    Mat D_00, D_01, D_02;  // 畸变系数
    Mat R_00, R_01, R_02;  // 旋转矩阵
    Mat T_00, T_01, T_02;  // 平移向量
    Mat R_rect_00, R_rect_01, R_rect_02;  // 校正旋转矩阵
    Mat P_rect_00, P_rect_01, P_rect_02;  // 校正投影矩阵
    Size img_size;
    Mat Q;  // 视差到深度的转换矩阵
};

// 解析KITTI标定文件
bool parseKITTICalibration(const string& calib_file, KITTIParams& params) {
    ifstream file(calib_file);
    if (!file.is_open()) {
        cout << "无法打开标定文件: " << calib_file << endl;
        return false;
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        
        istringstream iss(line);
        string key;
        iss >> key;
        
        if (key == "S_00:") {
            double width, height;
            iss >> width >> height;
            params.img_size = Size(static_cast<int>(width), static_cast<int>(height));
            cout << "图像尺寸: " << params.img_size.width << "x" << params.img_size.height << endl;
        }
        else if (key == "K_00:") {
            params.K_00 = Mat::eye(3, 3, CV_64F);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    iss >> params.K_00.at<double>(i, j);
            cout << "K_00 矩阵已解析" << endl;
        }
        else if (key == "K_01:") {
            params.K_01 = Mat::eye(3, 3, CV_64F);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    iss >> params.K_01.at<double>(i, j);
            cout << "K_01 矩阵已解析" << endl;
        }
        else if (key == "K_02:") {
            params.K_02 = Mat::eye(3, 3, CV_64F);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    iss >> params.K_02.at<double>(i, j);
            cout << "K_02 矩阵已解析" << endl;
        }
        else if (key == "D_00:") {
            params.D_00 = Mat::zeros(5, 1, CV_64F);
            for (int i = 0; i < 5; i++)
                iss >> params.D_00.at<double>(i);
            cout << "D_00 畸变系数已解析" << endl;
        }
        else if (key == "D_01:") {
            params.D_01 = Mat::zeros(5, 1, CV_64F);
            for (int i = 0; i < 5; i++)
                iss >> params.D_01.at<double>(i);
            cout << "D_01 畸变系数已解析" << endl;
        }
        else if (key == "D_02:") {
            params.D_02 = Mat::zeros(5, 1, CV_64F);
            for (int i = 0; i < 5; i++)
                iss >> params.D_02.at<double>(i);
            cout << "D_02 畸变系数已解析" << endl;
        }
        else if (key == "R_00:") {
            params.R_00 = Mat::eye(3, 3, CV_64F);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    iss >> params.R_00.at<double>(i, j);
            cout << "R_00 旋转矩阵已解析" << endl;
        }
        else if (key == "R_01:") {
            params.R_01 = Mat::eye(3, 3, CV_64F);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    iss >> params.R_01.at<double>(i, j);
            cout << "R_01 旋转矩阵已解析" << endl;
        }
        else if (key == "R_02:") {
            params.R_02 = Mat::eye(3, 3, CV_64F);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    iss >> params.R_02.at<double>(i, j);
            cout << "R_02 旋转矩阵已解析" << endl;
        }
        else if (key == "T_00:") {
            params.T_00 = Mat::zeros(3, 1, CV_64F);
            for (int i = 0; i < 3; i++)
                iss >> params.T_00.at<double>(i);
            cout << "T_00 平移向量已解析" << endl;
        }
        else if (key == "T_01:") {
            params.T_01 = Mat::zeros(3, 1, CV_64F);
            for (int i = 0; i < 3; i++)
                iss >> params.T_01.at<double>(i);
            cout << "T_01 平移向量已解析" << endl;
        }
        else if (key == "T_02:") {
            params.T_02 = Mat::zeros(3, 1, CV_64F);
            for (int i = 0; i < 3; i++)
                iss >> params.T_02.at<double>(i);
            cout << "T_02 平移向量已解析" << endl;
        }
        else if (key == "R_rect_00:") {
            params.R_rect_00 = Mat::eye(3, 3, CV_64F);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    iss >> params.R_rect_00.at<double>(i, j);
            cout << "R_rect_00 校正旋转矩阵已解析" << endl;
        }
        else if (key == "R_rect_01:") {
            params.R_rect_01 = Mat::eye(3, 3, CV_64F);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    iss >> params.R_rect_01.at<double>(i, j);
            cout << "R_rect_01 校正旋转矩阵已解析" << endl;
        }
        else if (key == "R_rect_02:") {
            params.R_rect_02 = Mat::eye(3, 3, CV_64F);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    iss >> params.R_rect_02.at<double>(i, j);
            cout << "R_rect_02 校正旋转矩阵已解析" << endl;
        }
        else if (key == "P_rect_00:") {
            params.P_rect_00 = Mat::zeros(3, 4, CV_64F);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    iss >> params.P_rect_00.at<double>(i, j);
            cout << "P_rect_00 校正投影矩阵已解析" << endl;
        }
        else if (key == "P_rect_01:") {
            params.P_rect_01 = Mat::zeros(3, 4, CV_64F);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    iss >> params.P_rect_01.at<double>(i, j);
            cout << "P_rect_01 校正投影矩阵已解析" << endl;
        }
        else if (key == "P_rect_02:") {
            params.P_rect_02 = Mat::zeros(3, 4, CV_64F);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    iss >> params.P_rect_02.at<double>(i, j);
            cout << "P_rect_02 校正投影矩阵已解析" << endl;
        }
    }
    
    file.close();
    
    // 验证必要的参数是否都已解析
    if (params.img_size.width == 0 || params.img_size.height == 0) {
        cout << "错误: 未解析到图像尺寸" << endl;
        return false;
    }
    if (params.K_00.empty() || params.K_01.empty() || params.D_00.empty() || params.D_01.empty()) {
        cout << "错误: 相机内参或畸变系数不完整" << endl;
        return false;
    }
    if (params.R_01.empty() || params.T_01.empty()) {
        cout << "错误: 立体相机外参不完整" << endl;
        return false;
    }
    
    // 计算Q矩阵用于视差到深度的转换
    // 使用cam0和cam1的基线
    double baseline = abs(params.P_rect_01.at<double>(0, 3) / params.P_rect_01.at<double>(0, 0));
    double fx = params.P_rect_00.at<double>(0, 0);
    double cx = params.P_rect_00.at<double>(0, 2);
    double cy = params.P_rect_00.at<double>(1, 2);
    
    params.Q = Mat::eye(4, 4, CV_64F);
    params.Q.at<double>(0, 3) = -cx;
    params.Q.at<double>(1, 3) = -cy;
    params.Q.at<double>(2, 3) = fx;
    params.Q.at<double>(3, 2) = 1.0 / baseline;
    params.Q.at<double>(3, 3) = 0;
    
    cout << "标定文件解析完成" << endl;
    cout << "基线长度: " << baseline << endl;
    
    return true;
}

// 立体校正
bool stereoRectify(const KITTIParams& params, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q) {
    // 检查必要的参数是否有效
    if (params.K_00.empty() || params.K_01.empty() || params.D_00.empty() || params.D_01.empty() ||
        params.R_01.empty() || params.T_01.empty() || params.img_size.width == 0 || params.img_size.height == 0) {
        cout << "标定参数不完整或无效" << endl;
        return false;
    }
    
    try {
        stereoRectify(params.K_00, params.D_00, params.K_01, params.D_01, params.img_size,
                      params.R_01, params.T_01, R1, R2, P1, P2, Q,
                      CALIB_ZERO_DISPARITY, 0, params.img_size);
        return true;
    } catch (const cv::Exception& e) {
        cout << "立体校正失败: " << e.what() << endl;
        return false;
    }
}

// 使用SGBM算法计算视差图
Mat computeDisparitySGBM(const Mat& img_left_rect, const Mat& img_right_rect, int num_disparities = 128, int block_size = 9) {
    Mat disp;
    
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, num_disparities, block_size);
    sgbm->setPreFilterCap(63);
    sgbm->setP1(8 * img_left_rect.channels() * block_size * block_size);
    sgbm->setP2(32 * img_left_rect.channels() * block_size * block_size);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(2);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);
    
    sgbm->compute(img_left_rect, img_right_rect, disp);
    
    return disp;
}

// 将视差图转换为深度图
Mat disparityToDepth(const Mat& disparity, const KITTIParams& params) {
    Mat depth = Mat::zeros(disparity.size(), CV_32F);
    
    double baseline = abs(params.P_rect_01.at<double>(0, 3) / params.P_rect_01.at<double>(0, 0));
    double fx = params.P_rect_00.at<double>(0, 0);
    
    for (int y = 0; y < disparity.rows; y++) {
        for (int x = 0; x < disparity.cols; x++) {
            float d = disparity.at<float>(y, x);
            if (d > 0) {
                depth.at<float>(y, x) = (fx * baseline) / d;
            }
        }
    }
    
    return depth;
}

// 将深度图对齐到cam2坐标系
Mat alignDepthToCam2(const Mat& depth_cam0, const KITTIParams& params) {
    Mat depth_cam2 = Mat::zeros(depth_cam0.size(), CV_32F);
    
    // 获取相机参数
    Mat K_00 = params.K_00;
    Mat K_02 = params.K_02;
    Mat D_02 = params.D_02;
    Mat R_02 = params.R_02;
    Mat T_02 = params.T_02;
    
    // 计算从cam0到cam2的变换
    Mat R_cam0_to_cam2 = R_02.t();
    Mat t_cam0_to_cam2 = -R_02.t() * T_02;
    
    // 创建映射
    Mat map1, map2;
    initUndistortRectifyMap(K_00, Mat::zeros(5, 1, CV_64F), R_cam0_to_cam2, K_02, depth_cam0.size(), CV_32FC1, map1, map2);
    
    // 重映射深度图
    remap(depth_cam0, depth_cam2, map1, map2, INTER_LINEAR);
    
    return depth_cam2;
}

// 将深度图转换为点云
Mat depthToPointCloud(const Mat& depth, const Mat& color, const KITTIParams& params) {
    Mat point_cloud;
    
    double fx = params.P_rect_02.at<double>(0, 0);
    double fy = params.P_rect_02.at<double>(1, 1);
    double cx = params.P_rect_02.at<double>(0, 2);
    double cy = params.P_rect_02.at<double>(1, 2);
    
    vector<Vec3f> points;
    vector<Vec3b> colors;
    
    for (int y = 0; y < depth.rows; y += 2) {  // 降采样以提高性能
        for (int x = 0; x < depth.cols; x += 2) {
            float d = depth.at<float>(y, x);
            if (d > 0 && d < 100) {  // 限制深度范围
                Vec3f point;
                point[0] = (x - cx) * d / fx;  // X
                point[1] = (y - cy) * d / fy;  // Y
                point[2] = d;                  // Z
                
                points.push_back(point);
                colors.push_back(color.at<Vec3b>(y, x));
            }
        }
    }
    
    // 创建点云
    point_cloud = Mat(points.size(), 1, CV_32FC3, points.data());
    
    return point_cloud;
}

// 使用cv::viz显示点云
void visualizePointCloud(const Mat& point_cloud, const vector<Vec3b>& colors) {
    viz::Viz3d window("Point Cloud");
    
    // 创建彩色点云
    Mat color_cloud(point_cloud.size(), CV_8UC3);
    for (int i = 0; i < point_cloud.rows; i++) {
        color_cloud.at<Vec3b>(i) = colors[i];
    }
    
    // 显示点云
    viz::WCloud cloud_widget(point_cloud, color_cloud);
    window.showWidget("Point Cloud", cloud_widget);
    
    // 设置相机视角
    window.setViewerPose(Affine3f().translate(Vec3f(0, 0, -50)));
    
    cout << "按任意键关闭点云显示..." << endl;
    window.spin();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "使用方法: " << argv[0] << " <kitti_data_path> [frame_id]" << endl;
        cout << "示例: " << argv[0] << " /media/cnhzcy14/work/data/kitti/2011_09_26_drive_0027_extract/ 0" << endl;
        return -1;
    }
    
    string kitti_path = argv[1];
    int frame_id = 0;
    if (argc > 2) {
        frame_id = atoi(argv[2]);
    }
    
    // 解析标定文件
    KITTIParams params;
    string calib_file = kitti_path + "/calib_cam_to_cam.txt";
    if (!parseKITTICalibration(calib_file, params)) {
        return -1;
    }
    
    // 构建图像文件路径
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/image_00/data/%010d.png", kitti_path.c_str(), frame_id);
    string img0_path = filename;
    snprintf(filename, sizeof(filename), "%s/image_01/data/%010d.png", kitti_path.c_str(), frame_id);
    string img1_path = filename;
    snprintf(filename, sizeof(filename), "%s/image_02/data/%010d.png", kitti_path.c_str(), frame_id);
    string img2_path = filename;
    
    // 读取图像
    Mat img0 = imread(img0_path, IMREAD_GRAYSCALE);
    Mat img1 = imread(img1_path, IMREAD_GRAYSCALE);
    Mat img2 = imread(img2_path, IMREAD_COLOR);
    
    if (img0.empty() || img1.empty() || img2.empty()) {
        cout << "无法读取图像文件" << endl;
        return -1;
    }
    
    cout << "图像尺寸: " << img0.cols << "x" << img0.rows << endl;
    
    // 步骤1: 计算立体校正参数
    cout << "正在计算立体校正参数..." << endl;
    Mat R1, R2, P1, P2, Q;
    if (!stereoRectify(params, R1, R2, P1, P2, Q)) {
        return -1;
    }
    
    // 步骤2: 计算校正映射
    Mat map11, map12, map21, map22;
    initUndistortRectifyMap(params.K_00, params.D_00, R1, P1, params.img_size, CV_32FC1, map11, map12);
    initUndistortRectifyMap(params.K_01, params.D_01, R2, P2, params.img_size, CV_32FC1, map21, map22);
    
    // 检查映射是否有效
    if (map11.empty() || map12.empty() || map21.empty() || map22.empty()) {
        cout << "校正映射计算失败" << endl;
        return -1;
    }
    
    // 步骤3: 应用校正映射（去畸变+立体校正）
    Mat img0_rect, img1_rect;
    remap(img0, img0_rect, map11, map12, INTER_LINEAR);
    remap(img1, img1_rect, map21, map22, INTER_LINEAR);
    
    // 检查校正后的图像是否有效
    if (img0_rect.empty() || img1_rect.empty()) {
        cout << "图像校正失败" << endl;
        return -1;
    }
    
    // 步骤4: 计算视差图（使用校正后的图像）
    cout << "正在计算视差图..." << endl;
    Mat disparity = computeDisparitySGBM(img0_rect, img1_rect);
    
    // 步骤5: 将视差图转换为深度图
    cout << "正在转换深度图..." << endl;
    Mat depth = disparityToDepth(disparity, params);
    
    // 步骤6: 对cam2图像进行去畸变
    cout << "正在对cam2图像去畸变..." << endl;
    Mat img2_undistorted;
    undistort(img2, img2_undistorted, params.K_02, params.D_02);
    
    // 步骤7: 将深度图对齐到cam2坐标系
    cout << "正在对齐深度图到cam2..." << endl;
    Mat depth_cam2 = alignDepthToCam2(depth, params);
    
    // 步骤8: 将深度图转换为点云
    cout << "正在生成点云..." << endl;
    Mat point_cloud = depthToPointCloud(depth_cam2, img2_undistorted, params);
    
    // 提取颜色信息
    vector<Vec3b> colors;
    double fx = params.P_rect_02.at<double>(0, 0);
    double fy = params.P_rect_02.at<double>(1, 1);
    double cx = params.P_rect_02.at<double>(0, 2);
    double cy = params.P_rect_02.at<double>(1, 2);
    
    for (int y = 0; y < depth_cam2.rows; y += 2) {
        for (int x = 0; x < depth_cam2.cols; x += 2) {
            float d = depth_cam2.at<float>(y, x);
            if (d > 0 && d < 100) {
                colors.push_back(img2_undistorted.at<Vec3b>(y, x));
            }
        }
    }
    
    // 保存结果图像
    Mat disp_vis;
    normalize(disparity, disp_vis, 0, 255, NORM_MINMAX, CV_8U);
    // applyColorMap(disp_vis, disp_vis, COLORMAP_JET);
    
    char output_name[256];
    snprintf(output_name, sizeof(output_name), "disparity_%06d.png", frame_id);
    imwrite(output_name, disp_vis);
    
    snprintf(output_name, sizeof(output_name), "depth_%06d.png", frame_id);
    imwrite(output_name, depth_cam2 * 0.01);  // 缩放以便显示
    
    snprintf(output_name, sizeof(output_name), "img2_undistorted_%06d.png", frame_id);
    imwrite(output_name, img2_undistorted);
    
    cout << "结果已保存到 disparity_" << frame_id << ".png, depth_" << frame_id << ".png, img2_undistorted_" << frame_id << ".png" << endl;
    
    // 显示点云（如果可能）
    if (!point_cloud.empty() && !colors.empty()) {
        cout << "显示点云..." << endl;
        visualizePointCloud(point_cloud, colors);
    }
    
    return 0;
}