```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! avenc_mpeg4 ! avdec_mpeg4 ! xvimagesink

gst-launch-1.0 -v udpsrc uri=udp://192.168.1.33:5000 ! tsparse ! tsdemux ! h264parse ! avdec_h264 ! videoconvert ! ximagesink sync=false

gst-launch-1.0 -v udpsrc uri=udp://192.168.1.33:5000 ! application/x-rtp, encoding-name=H264,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! ximagesink sync=false

gst-launch-1.0 -v udpsrc address=192.168.1.101 port=5000 caps='application/x-rtp, encoding-name=(string)H264, payload=(int)96' ! rtph264depay ! queue ! h264parse ! avdec_h264 ! videoconvert ! ximagesink sync=false

gst-launch-1.0 -v rtspsrc location='rtsp://admin:admin@192.168.1.108/cam/realmonitor?channel=1&subtype=0' ! decodebin ! videoconvert ! ximagesink sync=false

gst-launch-1.0 -v rtspsrc location='rtsp://127.0.0.1:8554/test' ! decodebin ! videoconvert ! ximagesink sync=false

gst-launch-1.0 -v rtspsrc location='rtsp://admin:admin@192.168.1.108/cam/realmonitor?channel=1&subtype=0' ! decodebin ! video/x-raw,format=I420 ! glimagesink sync=false

gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,format=GRAY8,width=1024,height=1024 ! videoconvert ! omxh264enc ! mpegtsmux ! udpsink host=192.168.1.33 port=5000

gst-launch-1.0 v4l2src device=/dev/video1 ! video/x-raw,format=GRAY8,width=1024,height=1024 ! videoconvert ! nvvidconv left=0 right=1024 top=0 bottom=512 ! 'video/x-raw(memory:NVMM),width=1024,height=512' ! omxh264enc ! mpegtsmux ! udpsink host=192.168.1.33 port=5000

gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,format=GRAY8,width=1024,height=1024 m.sink_0 v4l2src device=/dev/video1 ! video/x-raw,format=GRAY8,width=1024,height=1024 m.sink_1 compositor name=m sink_1::xpos=160 ! video/x-raw,width=2048,height=1024 ! nvvidconv ! omxh264enc ! mpegtsmux ! udpsink host=192.168.1.33 port=5000

gst-launch-1.0 videotestsrc ! video/x-raw,width=1280,height=720 ! videoconvert ! x264enc ! mpegtsmux ! udpsink host=192.168.1.101 port=5000

gst-launch-1.0 -v ximagesrc use-damage=0 ! nvvidconv ! 'video/x-raw(memory:NVMM),alignment=(string)au,format=(string)I420,framerate=(fraction)25/1,pixel-aspect-ratio=(fraction)1/1' ! omxh264enc !  'video/x-h264,stream-format=(string)byte-stream' ! h264parse ! matroskamux ! filesink location=screen.mkv

gst-launch-1.0 rtspsrc location='rtsp://192.168.1.24:8554/test' ! rtph264depay ! h264parse ! avdec_h264 ! glimagesink sync=false

gst-launch-1.0 rtspsrc latency=10 location='rtsp://192.168.1.24:8558/stereo' ! rtph264depay ! h264parse ! avdec_h264 ! glimagesink sync=false

gst-launch-1.0 videotestsrc  ! 'video/x-raw,width=640,height=480,framerate=30/1,format=I420' ! x264enc ! rtph264pay ! rtph264depay ! avdec_h264 ! glimagesink

gst-launch-1.0 nvcamerasrc num-buffers=150 ! tee name=t t. ! queue ! omxh264enc ! filesink location=a.h264 t. ! queue ! nvtee ! nvoverlaysink

gst-launch-1.0 videotestsrc ! 'video/x-raw,format=NV12,width=1280,height=720,framerate=30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)NV12' ! nvivafilter cuda-process=true customer-lib-name="libnvsample_cudaprocess.so" ! 'video/x-raw(memory:NVMM), format=(string)NV12' ! nvegltransform ! nveglglessink

gst-launch-1.0 filesrc location=landing_1.avi ! queue ! h264parse ! avdec_h264 ! videoconvert ! autovideosink

gst-launch-1.0 filesrc location=landing_1.avi ! queue ! h264parse ! omxh264dec ! nvvidconv ! 'video/x-raw, format=(string)I420_10LE' ! autovideosink

"rtspsrc location=rtsp://192.168.1.24:8554/test ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, format=(string)BGR, width=1024, height=1024 ! appsink sync=false"

gst-launch-1.0 multifilesrc location="%06d.png" index=0 caps="image/png,framerate=12/1" ! pngdec ! glimagesink

gst-launch-1.0 multifilesrc location="%08d.yuv" index=0 caps="video/x-raw,width=640,height=480,format=NV12" ! videoparse width=640 height=480 format=nv12 framerate=30/1 ! glimagesink sync=true

gst-launch-1.0 videotestsrc pattern=ball ! 'video/x-raw,width=1920,height=1080,framerate=1/1' ! queue2 ! nvjpegenc ! multifilesink location=test%04d.jpg sync=true

GST_PLUGIN_PATH=/usr/local/lib/x86_64-linux-gnu gst-launch-1.0  v4l2src device=/dev/video2 ! video/x-raw,width=640,height=480 ! videoconvert ! edgedetect threshold1=10 ! glimagesink

GST_DEBUG="*:3" gst-launch-1.0 videotestsrc ! video/x-raw,width=1920,height=1080,format=RGB,framerate=30/1 ! tee name=t ! queue ! videoconvert ! "video/x-raw,format=NV12,colorimetry=bt709" ! videoconvert ! video/x-raw,format=GRAY8 ! videoconvert ! ximagesink t. ! queue ! videoconvert ! video/x-raw,format=NV12,colorimetry=bt601 ! videoconvert ! video/x-raw,format=GRAY8  ! videoconvert ! ximagesink

OMP_PLACES={4},{5} OMP_NUM_THREADS=3 OMP_PROC_BIND=true GST_PLUGIN_PATH=/usr/local/lib/aarch64-linux-gnu/ gst-launch-1.0 rkv4l2src device=/dev/video0 ! video/x-raw,format=NV12,width=640,height=480, framerate=25/1 ! sparsefeaturetracker ! kmssink sync=true

gst-launch-1.0 v4l2src num-buffers=50 ! queue ! x264enc ! mp4mux ! filesink location=video.mp4

gst_rtsp -l 'v4l2src device=/dev/video0 ! video/x-raw,format=NV12,width=480,height=640,framerate=30/1 ! queue2 ! markerdetect dictionaryConfig=/usr/share/uavconfig/dict_params.json detectorConfig=/usr/share/uavconfig/detector_params.json ! ocvosd ! zmqmsg pubAddress=ipc://255.0.0 ! queue2 ! mpph264enc profile=high ! rtph264pay name=pay0 pt=96' -p 8555

GST_DEBUG=0 gst-launch-1.0 v4l2src device=/dev/video2 ! video/x-raw,format=GRAY8,width=640,height=480 ! queue2 ! markerdetect dictionaryConfig=/home/cnhzcy14/work/project/config/aruco/dict_params.json detectorConfig=/home/cnhzcy14/work/project/config/aruco/detector_params.json ! ocvosd ! queue2 ! glimagesink sync=true

OMP_PLACES={4},{5} OMP_PROC_BIND=true GST_DEBUG_FILE=/home/rock/work/log/gst.csv GST_DEBUG_NO_COLOR=1 GST_DEBUG=0,sparsefeaturetracker:4 gst-launch-1.0 multifilesrc location="/home/rock/test/image/%08d.png" index=0 start-index=0 caps="image/png,framerate=25/1" ! pngdec ! sparsefeaturetracker timestamp-file=/home/rock/test/image_time.csv ! fakevideosink sync=false

OMP_PLACES={5} OMP_PROC_BIND=true gst-launch-1.0 v4l2src device=/dev/video5 ! video/x-raw,format=NV12,width=640,height=480, framerate=25/1 ! queue2 use-buffering=true max-size-bytes=134217728 max-size-buffers=10000 ! sparsefeaturetracker ! fakevideosink sync=false

gst-launch-1.0 multifilesrc location="/home/cnhzcy14/work/data/vio/calib0/%02d.png" index=0 loop=true caps="image/png,framerate=1/1" ! pngdec ! calibboarddetect imageFolder=/home/cnhzcy14/work/data/vio/ delay=0 total=26 ! glimagesink

gst-launch-1.0 v4l2src device=/dev/video2 ! video/x-raw,width=640,height=480 ! videoconvert ! calibboarddetect  delay=2 boardPosConfig=/home/cnhzcy14/work/project/gst-ocv/config/aruco/board_pos_v3.json ! glimagesink

gst-launch-1.0 v4l2src device=/dev/video2 ! video/x-raw,format=GRAY8,width=640,height=480 ! tee name=t allow-not-linked=1 ! queue2 ! glimagesink  t. ! queue2 ! videorate max-rate=30 rate=30 ! glimagesink sync=false

gst-launch-1.0 multifilesrc location="/home/cnhzcy14/work/data/vio/test/image/%08d.png" index=0 caps="image/png,framerate=30/1" ! pngdec ! nvvideoconvert ! m.sink_0 nvstreammux name=m batch-size=1 width=1280 height=960 ! nvof preset-level=2 ! nvofvisual ! nvmultistreamtiler width=1280 height=960 ! nveglglessink

gst-launch-1.0 v4l2src ! video/x-raw,width=640,height=480 ! videoconvert ! nvvideoconvert ! queue ! m.sink_0 nvstreammux name=m batch-size=1 width=640 height=480 batched-push-timeout=40000 live-source=1 nvbuf-memory-type=0 ! nvof preset-level=2 ! nvofvisual ! nveglglessink

gst-launch-1.0 filesrc location=20240410_90_rgb.avi ! queue ! h264parse ! avdec_h264 ! videoconvert ! videorate ! "video/x-raw,framerate=90/1" ! videorate ! "video/x-raw,framerate=1/1" ! queue2 ! pngenc ! multifilesink location=img/%08d.png sync=true

gst-launch-1.0 filesrc location=pedestrians.mp4 ! decodebin ! autovideoconvert ! autovideosink
```
