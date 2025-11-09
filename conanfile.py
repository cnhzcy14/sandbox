from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.system.package_manager import Apt
import os

class CompressorRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"

    def requirements(self):
        self.requires("zlib/1.3.1")
        self.requires("openssl/1.1.1w")
        # self.requires("onnxruntime/1.14.1")
        self.requires("opencv/4.5.3")

    def system_requirements(self):
        if self.settings.os == "Linux":
            apt = Apt(self)
            # 检查并安装系统依赖
            # apt.install(["libssl-dev"],["vpi3-dev"])
            apt.install(["vpi3-dev"])
            # 安装TensorRT 10.3相关依赖（TensorRT会自动依赖CUDA）
            apt.install(["libnvinfer-dev"])
     
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

        self.options["openssl"].shared = False
        self.options["openssl"].fPIC = True
        
        # 禁用不需要的特性
        self.options["openssl"].enable_weak_ssl_ciphers = False
        self.options["openssl"].enable_capieng = False
        self.options["openssl"].enable_trace = False
        self.options["openssl"].capieng_dialog = False
        self.options["openssl"]["386"] = False
        
        # 禁用所有其他加密算法（只保留AES）
        self.options["openssl"].no_aria = True
        self.options["openssl"].no_bf = True
        self.options["openssl"].no_blake2 = True
        self.options["openssl"].no_camellia = True
        self.options["openssl"].no_cast = True
        self.options["openssl"].no_chacha = True
        self.options["openssl"].no_des = True
        self.options["openssl"].no_idea = True
        self.options["openssl"].no_md2 = True
        self.options["openssl"].no_poly1305 = True
        self.options["openssl"].no_md4 = True
        self.options["openssl"].no_mdc2 = True
        self.options["openssl"].no_rc2 = True
        self.options["openssl"].no_rc4 = True
        self.options["openssl"].no_rc5 = True
        self.options["openssl"].no_rmd160 = True
        self.options["openssl"].no_seed = True
        self.options["openssl"].no_sm2 = True
        self.options["openssl"].no_sm3 = True
        self.options["openssl"].no_sm4 = True
        self.options["openssl"].no_whirlpool = True
        
        # 禁用公钥加密算法
        self.options["openssl"].no_dh = True
        self.options["openssl"].no_dsa = True
        self.options["openssl"].no_ec = True
        self.options["openssl"].no_ecdh = True
        self.options["openssl"].no_ecdsa = True
        self.options["openssl"].no_gost = True
        self.options["openssl"].no_srp = True
        
        # 禁用SSL/TLS协议
        self.options["openssl"].no_ssl = True
        self.options["openssl"].no_ssl3 = True
        self.options["openssl"].no_tls1 = True
        self.options["openssl"].no_dgram = True
        
        # 禁用其他组件
        self.options["openssl"].no_apps = True
        self.options["openssl"].no_cms = True
        self.options["openssl"].no_comp = True
        self.options["openssl"].no_ct = True
        self.options["openssl"].no_dso = True
        self.options["openssl"].no_engine = True
        self.options["openssl"].no_fips = True
        self.options["openssl"].no_legacy = True
        self.options["openssl"].no_module = True
        self.options["openssl"].no_ocsp = True
        self.options["openssl"].no_pinshared = True
        self.options["openssl"].no_rfc3779 = True
        self.options["openssl"].no_srtp = True
        self.options["openssl"].no_ts = True
        self.options["openssl"].no_sock = True
        self.options["openssl"].no_zlib = False
        self.options["openssl"].no_async = True
        self.options["openssl"].no_autoload_config = True

        self.options["openssl"].no_asm = False
        self.options["openssl"].no_sse2 = True
        self.options["openssl"].no_stdio = True
        self.options["openssl"].no_tests = True
        self.options["openssl"].no_filenames = True
        self.options["openssl"].no_threads = False

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


    def generate(self):
        tc = CMakeToolchain(self)
        # 完整的交叉编译设置
        # tc.variables["CMAKE_SYSTEM_NAME"] = "Linux"
        # tc.variables["CMAKE_SYSTEM_PROCESSOR"] = "aarch64"
        # tc.variables["CMAKE_C_COMPILER"] = "aarch64-linux-gnu-gcc-8"
        # tc.variables["CMAKE_CXX_COMPILER"] = "aarch64-linux-gnu-g++-8"
        
        # 编译选项现在通过profile中的环境变量传递
        # 这里可以添加一些通用的、不依赖具体硬件的选项
        if self.settings.build_type == "Release":
            # 如果环境变量中没有设置，则使用基本优化
            if not os.environ.get("CFLAGS"):
                tc.variables["CMAKE_C_FLAGS"] = "-O2"
            if not os.environ.get("CXXFLAGS"):
                tc.variables["CMAKE_CXX_FLAGS"] = "-O2"
            if not os.environ.get("LDFLAGS"):
                tc.variables["CMAKE_EXE_LINKER_FLAGS"] = ""
                tc.variables["CMAKE_SHARED_LINKER_FLAGS"] = ""
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()