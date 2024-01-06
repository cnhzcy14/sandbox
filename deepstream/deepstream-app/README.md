
## usage:
```
$./deepstream-app -c <config-file>

Please refer "../../apps-common/includes/deepstream_config.h" to modify
application parameters like maximum number of sources etc.
```


## note:
```
1. Prerequisites to use nvdrmvideosink (Jetson only)
   a. Ensure that X server is not running.
      Command to stop X server:
          $sudo service gdm stop
          $sudo pkill -9 Xorg
   b. If "Could not get EGL display connection" error is encountered,
      use command $unset DISPLAY
   c. Ensure that a display device is connected to the Jetson board.
```

