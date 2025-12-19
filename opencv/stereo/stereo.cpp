#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>

struct StereoConfig {
    // 图像路径
    std::string leftImagePath = "img/R-L.png";
    std::string rightImagePath = "img/R-R.png";
    
    // StereoBM 参数
    int bmBlockSize = 9;
    
    // StereoSGBM 参数
    int minDisparity = 0;
    int numDisparities = 128;
    int windowSize = 9;
    int preFilterCap = 63;
    int uniquenessRatio = 10;
    int speckleWindowSize = 100;
    int speckleRange = 2;
    int disp12MaxDiff = 1;
    
    // 输出参数
    std::string outputDir = "img";
};

struct StereoResult {
    std::string mode;
    double time_ms;
    cv::Mat disparity;
};

class StereoMatcher {
private:
    StereoConfig config_;
    
public:
    StereoMatcher(const StereoConfig& config) : config_(config) {}
    
    // 测试 StereoBM
    StereoResult testStereoBM(const cv::Mat& left, const cv::Mat& right) {
        cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(config_.numDisparities, config_.bmBlockSize);
        
        auto st = std::chrono::high_resolution_clock::now();
        cv::Mat disp;
        bm->compute(left, right, disp);
        auto et = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
        
        return {"StereoBM", time_ms, disp};
    }
    
    // 测试 StereoSGBM 的不同模式
    StereoResult testStereoSGBM(const cv::Mat& left, const cv::Mat& right, int mode, const std::string& modeName) {
        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
            config_.minDisparity, config_.numDisparities, config_.windowSize);
        
        sgbm->setPreFilterCap(config_.preFilterCap);
        sgbm->setBlockSize(config_.windowSize);
        int cn = left.channels();
        sgbm->setP1(8 * cn * config_.windowSize * config_.windowSize);
        sgbm->setP2(32 * cn * config_.windowSize * config_.windowSize);
        sgbm->setMinDisparity(config_.minDisparity);
        sgbm->setNumDisparities(config_.numDisparities);
        sgbm->setUniquenessRatio(config_.uniquenessRatio);
        sgbm->setSpeckleWindowSize(config_.speckleWindowSize);
        sgbm->setSpeckleRange(config_.speckleRange);
        sgbm->setDisp12MaxDiff(config_.disp12MaxDiff);
        sgbm->setMode(mode);
        
        auto st = std::chrono::high_resolution_clock::now();
        cv::Mat disp;
        sgbm->compute(left, right, disp);
        auto et = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
        
        return {modeName, time_ms, disp};
    }
    
    // 保存视差图
    void saveDisparity(const StereoResult& result, const std::string& filename) {
        cv::Mat disp_normalized;
        cv::normalize(result.disparity, disp_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(config_.outputDir + "/" + filename, disp_normalized);
    }
    
    // 运行所有测试
    void runTests() {
        std::vector<StereoResult> results;
        
        std::cout << "\n=== 测试真实图像 ===" << std::endl;
        
        cv::Mat img_left = cv::imread(config_.leftImagePath, cv::IMREAD_COLOR);
        cv::Mat img_right = cv::imread(config_.rightImagePath, cv::IMREAD_COLOR);
        
        if (img_left.empty() || img_right.empty()) {
            std::cout << "错误: 无法加载图像文件" << std::endl;
            std::cout << "左图像: " << config_.leftImagePath << std::endl;
            std::cout << "右图像: " << config_.rightImagePath << std::endl;
            return;
        }
        
        std::cout << "图像尺寸 - 左图: " << img_left.cols << "x" << img_left.rows
                  << " 通道: " << img_left.channels() << std::endl;
        std::cout << "图像尺寸 - 右图: " << img_right.cols << "x" << img_right.rows
                  << " 通道: " << img_right.channels() << std::endl;
        
        // 首先测试StereoBM
        cv::Mat img_left_gray, img_right_gray;
        cv::cvtColor(img_left, img_left_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img_right, img_right_gray, cv::COLOR_BGR2GRAY);
        
        auto bm_result = testStereoBM(img_left_gray, img_right_gray);
        std::cout << bm_result.mode << " time: " << bm_result.time_ms << " ms" << std::endl;
        saveDisparity(bm_result, "disp_StereoBM.png");
        std::cout << "  视差图已保存: " << config_.outputDir << "/disp_StereoBM.png" << std::endl;
        results.push_back(bm_result);
        
        // 测试不同的 SGBM 模式
        std::vector<std::pair<int, std::string>> modes = {
            {cv::StereoSGBM::MODE_SGBM_3WAY, "MODE_SGBM_3WAY"},
            {cv::StereoSGBM::MODE_HH4, "MODE_HH4"},
            {cv::StereoSGBM::MODE_SGBM, "MODE_SGBM"},
            {cv::StereoSGBM::MODE_HH, "MODE_HH"}
        };
        
        for (const auto& mode : modes) {
            auto result = testStereoSGBM(img_left, img_right, mode.first, mode.second);
            std::cout << result.mode << " time: " << result.time_ms << " ms" << std::endl;
            
            // 保存视差图
            std::string filename = "disp_" + mode.second + ".png";
            saveDisparity(result, filename);
            std::cout << "  视差图已保存: " << config_.outputDir << "/" << filename << std::endl;
        }
        
        // 打印结果摘要
        std::cout << "\n=== 测试结果摘要 ===" << std::endl;
        for (const auto& result : results) {
            std::cout << result.mode << ": " << result.time_ms << " ms" << std::endl;
        }
    }
};

// 打印使用说明
void printUsage(const char* programName) {
    std::cout << "使用说明: " << programName << " [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --left <路径>          左图像路径 (默认: img/IR-L.png)" << std::endl;
    std::cout << "  --right <路径>         右图像路径 (默认: img/IR-R.png)" << std::endl;
    std::cout << "  --min-disp <数值>      最小视差 (默认: 0)" << std::endl;
    std::cout << "  --num-disp <数值>      视差数量 (默认: 128)" << std::endl;
    std::cout << "  --window-size <数值>   SGBM窗口大小 (默认: 9)" << std::endl;
    std::cout << "  --bm-block-size <数值> BM块大小 (默认: 9)" << std::endl;
    std::cout << "  --output <路径>        输出目录 (默认: img)" << std::endl;
    std::cout << "  --help                 显示此帮助信息" << std::endl;
}

// 解析命令行参数
StereoConfig parseArguments(int argc, char** argv) {
    StereoConfig config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            exit(0);
        } else if (arg == "--left" && i + 1 < argc) {
            config.leftImagePath = argv[++i];
        } else if (arg == "--right" && i + 1 < argc) {
            config.rightImagePath = argv[++i];
        } else if (arg == "--min-disp" && i + 1 < argc) {
            config.minDisparity = std::stoi(argv[++i]);
        } else if (arg == "--num-disp" && i + 1 < argc) {
            config.numDisparities = std::stoi(argv[++i]);
        } else if (arg == "--window-size" && i + 1 < argc) {
            config.windowSize = std::stoi(argv[++i]);
        } else if (arg == "--bm-block-size" && i + 1 < argc) {
            config.bmBlockSize = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            config.outputDir = argv[++i];
        } else {
            std::cerr << "警告: 未知参数 '" << arg << "'" << std::endl;
        }
    }
    
    return config;
}

int main(int argc, char** argv) {
    try {
        // 解析命令行参数
        StereoConfig config = parseArguments(argc, argv);
        
        std::cout << "=== OpenCV 立体匹配性能测试 ===" << std::endl;
        std::cout << "配置参数:" << std::endl;
        std::cout << "  视差范围: [" << config.minDisparity << ", "
                  << config.minDisparity + config.numDisparities << "]" << std::endl;
        std::cout << "  SGBM窗口大小: " << config.windowSize << std::endl;
        std::cout << "  BM块大小: " << config.bmBlockSize << std::endl;
        std::cout << "  左图像: " << config.leftImagePath << std::endl;
        std::cout << "  右图像: " << config.rightImagePath << std::endl;
        std::cout << "  输出目录: " << config.outputDir << std::endl;
        
        // 创建立体匹配器并运行测试
        StereoMatcher matcher(config);
        matcher.runTests();
        
        std::cout << "\n测试完成!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}