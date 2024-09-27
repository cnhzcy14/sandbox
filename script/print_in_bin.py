from PIL import Image
import cv2 as cv
import numpy as np

y = np.fromfile("D:\share3\model-globalMatting\in\in.bin", np.uint8)
x = y.reshape(3, 512, 640)
b = Image.fromarray(x[0,:,:])
g = Image.fromarray(x[1,:,:])
r = Image.fromarray(x[2,:,:])
img = Image.merge("RGB", (r, g, b))
img.show()
