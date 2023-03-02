#!/bin/sh
trap "exit" INT
cat /dev/null > cpu_therm
cat /dev/null > gpu_therm
while :
do
	cat /sys/class/thermal/thermal_zone1/temp >> cpu_therm
	cat /sys/class/thermal/thermal_zone2/temp >> gpu_therm
	sleep 2
done
