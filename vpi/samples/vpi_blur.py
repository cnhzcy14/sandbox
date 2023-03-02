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

# 1. Import needed modules
# -----------------------------------------------------------------------------

import sys
import vpi

import numpy as np
from PIL import Image
from argparse import ArgumentParser

# 2. Parse command line parameters
# -----------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument('input',
                    help='Image to be used as input')
args = parser.parse_args();

# 3. Load input and wrap it into a VPI image.
# -----------------------------------------------------------------------------

# `Image.open` returns a Pillow image that is then interpreted as
# a numpy array. This array is finally wrapped in a VPI image suitable
# for use by VPI algorithms.
try:
    input = vpi.asimage(np.asarray(Image.open(args.input)))
except IOError:
    sys.exit("Input file not found")
except:
    sys.exit("Error with input file")

# 4. Convert it to grayscale and blur it with a 5x5 box filter
#    with ZERO border condition.
# -----------------------------------------------------------------------------

# Enabling the CUDA backend in a python context like done below makes
# VPI algorithms use CUDA for execution by default. This can be overriden
# if needed by specifying the parameter `backend=` when calling the algorithm.
with vpi.Backend.CUDA:
    # `image.convert` will return a new VPI image with the desired format, in
    # this case U8 (grayscale, 8-bit unsigned pixels).
    # Algorithms returning a new VPI image allows for chaining operations like
    # done below, as the result of the conversion is then filtered.
    # The end result is finally stored in a new `output` VPI image.
    output = input.convert(vpi.Format.U8) \
                  .box_filter(5, border=vpi.Border.ZERO)

# 5. Save result to disk
# -----------------------------------------------------------------------------

# The read-lock context enabled below makes sure all processing is finished and
# results are stored in `output`.
with output.rlock_cpu() as outData:
    # `outData` is a numpy array view (not a copy) of `output`
    # contents, accessible by host (cpu). 
    # The numpy array is then converted into a Pillo image and saved
    # to the disk
    Image.fromarray(outData).save('tutorial_blurred_python.png')
