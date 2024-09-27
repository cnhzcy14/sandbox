from PIL import Image
import numpy as np

y = np.fromfile("/home/cnhzcy14/out_m01.bin", np.float32)
x = y.reshape(512, 640) > 0.5
img = Image.fromarray(x)
img.show()
