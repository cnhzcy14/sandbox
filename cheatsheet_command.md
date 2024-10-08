## nvprof

```bash
nvprof -f --output-profile profile.log
export NVX_PROF=nvtx
```

## linux
```bash
right compare
ls -l 
```

## v4l2

```bash
v4l2-ctl -d /dev/video0 --list-ctrls
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --stream-mmap=3 --stream-skip=3 --stream-count=1 --stream-poll --stream-to=cif.out

./dmabuf-sharing -M i915 -i /dev/video2 -S 640,480 -f UYVY -F UYVY -b 2 -s 640,480@0,0 -t 640,480@0,0
```

## ffmpeg
```bash
$ ffmpeg -video_size 1920x1080 -framerate 25 -f x11grab -i :1 -pix_fmt yuv420p output.mp4
```

## nfs

```bash
/etc/exports: /home/cnhzcy14/work *(rw,nohide,insecure,no_subtree_check,no_root_squash,sync)
sudo /etc/init.d/nfs-kernel-server restart
sudo mount -t nfs 10.35.17.134:/home/cnhzcy14/work/tx1 tx1
sudo umount tx1

sudo mount -t nfs 192.168.1.100:/home/cnhzcy14/work work
```

## power

```bash
gsettings set org.gnome.settings-daemon.plugins.power button-power 'shutdown'
```

## graphic card memory

```bash
glxinfo | egrep -i 'device|memory'
```

## imagemagik

```bash
mogrify -format ppm *.png
mogrify -resize 50% *.ppm
convert -delay 100 -loop 0 *.png animation.gif
convert tag16_05_00014.png -scale 10000% convert tag16_05_00014_.png
```

## opencv compile

```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_OPENCL=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_OPENMP=ON -D WITH_GSTREAMER=ON -D OPENCV_ENABLE_NONFREE=ON -D WITH_IPP=ON -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D WITH_CUFFT=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D CUDA_ARCH_BIN=3.0 -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=/home/cnhzcy14/work/sw/opencv/opencv_contrib-4.4.0/modules ..

cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_OPENMP=ON -D OPENCV_ENABLE_NONFREE=ON -D ENABLE_FAST_MATH=ON -D WITH_OPENCL=ON -D ENABLE_NEON=ON -D OPENCV_GENERATE_PKGCONFIG=ON ..

cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_OPENMP=ON -D OPENCV_ENABLE_NONFREE=ON -D ENABLE_FAST_MATH=ON -D WITH_OPENCL=ON -D ENABLE_NEON=ON -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_opencv_xfeatures2d=OFF -D OPENCV_GENERATE_PKGCONFIG=ON ..



cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_OPENGL=ON -D WITH_OPENMP=ON -D OPENCV_ENABLE_NONFREE=ON -D ENABLE_FAST_MATH=ON -D WITH_OPENCL=ON -D ENABLE_NEON=ON -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_opencv_xfeatures2d=OFF -D OPENCV_GENERATE_PKGCONFIG=ON ..


cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_OPENCL=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_OPENMP=ON -D WITH_GSTREAMER=ON -D OPENCV_ENABLE_NONFREE=ON -D WITH_IPP=ON -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D WITH_CUFFT=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D CUDA_ARCH_BIN=3.0 -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=/home/cnhzcy14/work/project/opencv_contrib/modules -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_opencv_alphamat=OFF -D BUILD_opencv_bgsegm=OFF -D BUILD_opencv_bioinspired=OFF -D BUILD_opencv_ccalib=OFF -D BUILD_opencv_datasets=OFF -D BUILD_opencv_dnn_objdetect=OFF -D BUILD_opencv_dnn_superres=OFF -D BUILD_opencv_dpm=OFF -D BUILD_opencv_face=OFF -D BUILD_opencv_freetype=OFF -D BUILD_opencv_fuzzy=OFF -D BUILD_opencv_hdf=OFF -D BUILD_opencv_hfs=OFF -D BUILD_opencv_intensity_transform=OFF -D BUILD_opencv_line_descriptor=OFF -D BUILD_opencv_optflow=OFF -D BUILD_opencv_phase_unwrapping=OFF -D BUILD_opencv_plot=OFF -D BUILD_opencv_quality=OFF -D BUILD_opencv_rapid=OFF -D BUILD_opencv_reg=OFF -D BUILD_opencv_rgbd=OFF -D BUILD_opencv_structured_light=OFF -D BUILD_opencv_superres=OFF -D BUILD_opencv_surface_matching=OFF -D BUILD_opencv_text=OFF -D BUILD_opencv_tracking=OFF -D BUILD_opencv_videostab=OFF -D BUILD_opencv_xobjdetect=OFF -D BUILD_opencv_xphoto=OFF -D BUILD_opencv_ximgproc=OFF ..
```

## ort runtime compile
```bash
./build.sh --cudnn_home /usr/lib/x86_64-linux-gnu --cuda_home /usr/local/cuda --use_tensorrt --tensorrt_home /usr/src/tensorrt --config Release
```

## bash

```bash
source ~/.bashrc
```

## glog

```bash
export GLOG_log_dir=/tmp
export GLOG_logtostderr=1
export GLOG_max_log_size=1
export GLOG_minloglevel=1
export GLOG_stderrthreshold=1
export GLOG_v=11
export GLOG_log_prefix=0
```

## usb

```bash
sudo fdisk -l
```

## other

```bash
./larvio ~/work/data/vio/test/imu_data.csv ~/work/data/vio/test/image_time.csv ~/work/data/vio/test/image/ ~/work/project/larvio/config/d435i_640_480.json

OMP_PLACES={4},{5} OMP_NUM_THREADS=3 OMP_PROC_BIND=true  ./larvio /home/rock/work/project/data/vio/test/imu_data.csv /home/rock/work/project/data/vio/test/image_time.csv /home/rock/work/project/data/vio/test/image/ /home/rock/work/project/data/vio/test/d435i_640_480.yaml

./larvio ~/work/data/vio/rk_07281/imu_larvio_200.csv ~/work/data/vio/rk_07281/image.csv 'multifilesrc location=/home/cnhzcy14/work/data/vio/rk_07281/image/%08d.png index=1 ! pngdec ! appsink' ~/work/project/larvio/config/sc132_220725_1.json

```

## omp

```bash
OMP_PLACES={4},{5} OMP_NUM_THREADS=3 OMP_PROC_BIND=true
```

## ldd

```bash
check dependency
```

## git

```bash
git clone --single-branch --branch <branchname> <remote-repo>
repo sync -l
```

## add external swap
```bash
sudo -i
mkswap /dev/sda1
swapon /dev/sda1
echo 0 > /proc/sys/kernel/hung_task_timeout_secs
```

## gdal
```bash
gdalwarp -s_srs epsg:32651 -t_srs epsg:3857 odm_orthophoto.tif ~/target.tif
```

## calib
```bash
./opencv/aruco/calibrate_camera_charuco --cp=/home/cnhzcy14/work/project/config/calib/default.json --dp=/home/cnhzcy14/work/project/config/aruco/detector_params.json   -v=/home/cnhzcy14/work/data/vio/calib0/%02d.png r.json
stereo_calib_yuv -w=4 -h=5 -s=0.2 /home/cnhzcy14/work/data/stereo/imx290_up/yuv_0/imx290_up_0.xml
```

## abd
```bash
sudo adb kill-server
sudo adb shell
```

## cpu frequency
 
```bash
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor

echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor

echo 800000000 > /sys/class/devfreq/ff9a0000.gpu/max_freq
echo performance > /sys/class/devfreq/ff9a0000.gpu/governor 

cat /sys/class/devfreq/dmc/governor
```

# sha512
```bash
sha512sum -c xxxx.sha512
```

# gnome
```bash
gsettings set org.gnome.shell.extensions.desktop-icons show-trash false
gsettings set org.gnome.shell.extensions.desktop-icons show-home false
gsettings set org.gnome.shell.extensions.dash-to-dock extend-height false
```

# clash
```bash
sudo vi /etc/environment

http_proxy=http://127.0.0.1:7890/
https_proxy=http://127.0.0.1:7890/
#ftp_proxy=http://myproxy.server.com:8080/
no_proxy="localhost,127.0.0.0/8,::1"
HTTP_PROXY=http://127.0.0.1:7890/
HTTPS_PROXY=http://127.0.0.1:7890/
#FTP_PROXY=http://myproxy.server.com:8080/
NO_PROXY="localhost,127.0.0.0/8,::1"

sudo vi /etc/apt/apt.conf.d/proxy.conf

Acquire::http::Proxy "http://127.0.0.1:7890/";
Acquire::https::Proxy "http://127.0.0.1:7890/";
```
# linux user
```bash
sudo adduser _00 --force-badname # add user _00
or
sudo useradd -s /bin/bash -m -G sudo _00
sudo usermod -aG sudo _00 # add admin to _00 if needed
sudo passwd _00 # change password of _00 if needed
sudo userdel -r _00 # delet user and his home

sudo usermod -a -G docker $USER # add _00 to docker group

sudo update-locale  LANG=C.UTF-8

chown star:star star/ # change folder owner and group
```