# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import vpi
import numpy as np
from argparse import ArgumentParser
import cv2

# ============================
# Parse command line arguments

parser = ArgumentParser()
parser.add_argument('-c', metavar='W,H', required=True,
                    help='Checkerboard with WxH squares')

parser.add_argument('-s', metavar='win', type=int,
                    help='Search window width around checkerboard verted used in refinement, default is 0 (disable refinement)')

parser.add_argument('images', nargs='+',
                    help='Input images taken with a fisheye lens camera')

args = parser.parse_args();

# Parse checkerboard size
try:
    cbSize = np.array([int(x) for x in args.c.split(',')])
except ValueError:
    exit("Error parsing checkerboard information")

# =========================================
# Calculate fisheye calibration from images

# OpenCV expects number of interior vertices in the checkerboard,
# not number of squares. Let's adjust for that.
vtxCount = cbSize-1

# -------------------------------------------------
# Determine checkerboard coordinates in image space

imgSize = None
corners2D = []

for imgName in args.images:
    # Load input image and do some sanity check
    img = cv2.imread(imgName)
    curImgSize = (img.shape[1], img.shape[0])

    if imgSize == None:
        imgSize = curImgSize
    elif imgSize != curImgSize:
        exit("All images must have the same size")

    # Find the checkerboard pattern on the image, saving the 2D
    # coordinates of checkerboard vertices in cbVertices.
    # Vertex is the point where 4 squares (2 white and 2 black) meet.
    found, corners = cv2.findChessboardCorners(img, tuple(vtxCount), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        # Needs to perform further corner refinement?
        if args.s != None and args.s >= 2:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.0001)
            imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            corners = cv2.cornerSubPix(imgGray, corners, (args.s//2, args.s//2), (-1,-1), criteria)
        corners2D.append(corners)
    else:
        exit("Warning: checkerboard pattern not found in image {}".format(input))

# Create the vector that stores 3D coordinates for each checkerboard pattern on a space
# where X and Y are orthogonal and run along the checkerboard sides, and Z==0 in all points on
# checkerboard.
cbCorners = np.zeros((1, vtxCount[0]*vtxCount[1], 3))
cbCorners[0,:,:2] = np.mgrid[0:vtxCount[0], 0:vtxCount[1]].T.reshape(-1,2)
corners3D = [cbCorners.reshape(-1,1,3) for i in range(len(corners2D))]

# ---------------------------------------------
# Calculate fisheye lens calibration parameters
camMatrix = np.eye(3)
coeffs = np.zeros((4,))
rms, camMatrix, coeffs, rvecs, tvecs = cv2.fisheye.calibrate(corners3D, corners2D, imgSize, camMatrix, coeffs, flags=cv2.fisheye.CALIB_FIX_SKEW)

# Print out calibration results
print("rms error: {}".format(rms))
print("Fisheye coefficients: {}".format(coeffs))
print("Camera matrix:")
print(camMatrix)

# ======================
# Undistort input images

# Create an uniform grid
grid = vpi.WarpGrid(imgSize)

# Create undistort warp map from the calibration parameters and the grid
undist_map = vpi.WarpMap.fisheye_correction(grid,
                                            K=camMatrix[0:2,:], X=np.eye(3,4), coeffs=coeffs,
                                            mapping=vpi.FisheyeMapping.EQUIDISTANT)

# Go through all input images,
idx=0
for imgName in args.images:
    # Load input image and do some sanity check
    img = cv2.imread(imgName)

    # Using the CUDA backend,
    with vpi.Backend.CUDA:
        # Convert image to NV12_ER, apply the undistortion map and convert image back to RGB8
        imgCorrected = vpi.asimage(img).convert(vpi.Format.NV12_ER).remap(undist_map, interp=vpi.Interp.CATMULL_ROM).convert(vpi.Format.RGB8)

    # Write undistorted image to disk
    cv2.imwrite("undistort_python{}_{:03d}.jpg".format(sys.version_info[0],idx), imgCorrected.cpu())
    idx += 1
