import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.spatial.transform import Rotation as Ro
import cv2 as cv

np.set_printoptions(suppress=True)

# ocvK = np.matrix([[2458.517,0.0,1952.3],[0.0,2447.0,1090.3],[0,0,1]])
# ocvD = np.matrix([[-0.08674945926693, 0.03405476070122, 0.0, 0.0, 0.0]])
ocvK = np.matrix([[2338.6,0.0,1981.7],[0.0,2325.6,1082.6],[0,0,1]])
ocvD = np.matrix([[-0.091354126557530, 0.022961632459289, 0.0, 0.0, 0.0]])
K, validPixROI = cv.getOptimalNewCameraMatrix(ocvK, ocvD, (3840, 2160), 0, (3840, 2160))
print(K)

index = 0
with open('/home/cnhzcy14/work/project/data/0528_zj/0528_zj.nvm.cmvs/01/output.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        c = np.array([float(x) for x in row[4:]]).reshape(3,1)
        R = Ro.from_quat(row[0:4])
        R = Ro.from_euler('xyz',-R.as_euler('xyz', degrees=True),degrees=True)   
        t = -np.matmul(R.as_matrix(), c)
        Rt = np.concatenate((R.as_matrix(), t), axis=1)
        # print(Rt)
        P = np.matmul(K, Rt)
        np.savetxt(str(index).zfill(8) + '.txt', P, comments='', header='CONTOUR', fmt='%f')
        index+=1


