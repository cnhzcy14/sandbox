from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout

class CompressorRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        self.requires("zlib/1.3.1")
        # self.requires("onnxruntime/1.14.1")
        # self.requires("openjpeg/2.3.1")
        self.requires("opencv/4.5.3")
       
    def layout(self):
        cmake_layout(self)

    def config_options(self):
        # self.options["onnxruntime"].onnxruntime_ENABLE_CPUINFO = False
                # 最小化ONNX Runtime配置
        # self.options["onnxruntime"].shared = True
        # self.options["onnxruntime"].build_shared_lib = True
        # self.options["onnxruntime"].minimal_build = True
        # self.options["onnxruntime"].enable_training = False
        # self.options["onnxruntime"].use_boost = False


        # 设置OpenCV基本选项
        self.options["opencv"].shared = False  
        self.options["opencv"].parallel = "openmp" 
        self.options["opencv"].with_opencl = False
        self.options["opencv"].with_cuda = False
        self.options["opencv"].with_cublas = False
        self.options["opencv"].with_cudnn = False

        self.options["opencv"].gapi = False  
        self.options["opencv"].ml = False  
        self.options["opencv"].dnn = False  
        self.options["opencv"].videoio = False  
        self.options["opencv"].photo = False
        self.options["opencv"].highgui = False
        self.options["opencv"].objdetect = False
        self.options["opencv"].contrib = False


        self.options["opencv"].with_png = True    
        self.options["opencv"].with_openexr = False
        self.options["opencv"].with_tiff = False
        self.options["opencv"].with_webp = False   
        self.options["opencv"].with_gdal = False   
        self.options["opencv"].with_gdcm = False   
        self.options["opencv"].with_imgcodec_hdr = False   
        self.options["opencv"].with_imgcodec_pfm = False   
        self.options["opencv"].with_imgcodec_pxm = False   
        self.options["opencv"].with_imgcodec_sunraster = False   
        self.options["opencv"].with_msmf = False   
        self.options["opencv"].with_msmf_dxva = False   


    # def generate(self):
    #     tc = CMakeToolchain(self)
    #     # 完整的交叉编译设置
    #     tc.variables["CMAKE_SYSTEM_NAME"] = "Linux"
    #     tc.variables["CMAKE_SYSTEM_PROCESSOR"] = "aarch64"
    #     tc.variables["CMAKE_C_COMPILER"] = "aarch64-linux-gnu-gcc"
    #     tc.variables["CMAKE_CXX_COMPILER"] = "aarch64-linux-gnu-g++"
    #     # 关键：禁用cpuinfo（使用全大写的选项名）
    #     tc.variables["ONNXRUNTIME_ENABLE_CPUINFO"] = "OFF"
    #     tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()