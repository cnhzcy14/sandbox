## calibration
```
1. Keep every circle diameter in the picture is about 15-20 pixels (20 is the best, 10 – 20 is also ok), the bigger size will cause an accuracy loss.
2. It is recommended to keep at least 1 April tag in one image.
3. Keep images covered different parts of FoV.
4. Ambarella recommends to capture images at 3 different distances, such as 1.5 / 2 / 2.5 meters. (capturing lots of images in fixed distance(1.5 meters), and 3 images in far distance (2 / 2.5 meters)).
5. Keep the board in different angles (pitch / yaw / roll), but the angles should not be too big, refer to the image above.
```

## stereo
```
init.sh --na
rmmod ambarella_fb
modprobe ambarella_fb mode=clut8bpp resolution=1280x720 buffernum=4
modprobe b8 id=0x30
modprobe imx290_mipi_brg
test_aaa_service -a &
test_encode --resource-cfg cv2_vin2_720p_stereo.lua --hdmi 720p

test_stereo_server --fb-dsi 1 --no-dewarp --clut-start 0 --clut-end 127 &
modprobe cavalry
cavalry_load -f /lib/firmware/cavalry.bin -r
test_cavalry_stereo_live -b /usr/local/vproc/vproc.bin -t 0 --sb 2
```

## rtsp
```
test_encode -A -h 720p -b 0 -e
rtsp_server
```

## pip
```
test_encode --resource-cfg cv2_vin0_1_1080p_for_pip.lua --hdmi 1080p
test_multi_chan -c 1 -J --boutsize 720x480 --boutoffset 1200x0
test_multi_chan  -C 4 -o 0 -o 1
```