from scipy import misc
from skimage import feature
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, roberts, sobel, scharr, prewitt
from skimage.color import rgb2gray
import numpy as np
import numpy.matlib
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
from skimage import img_as_ubyte, io
import os, sys
import MLP
import CNN
import spectral_roi
import Helper
import Display

# img_data = []
# def func(block):
#     # Check if not all zeros
#     if(numpy.any(block)):
#         #io.imsave("Block2/{}.png".format(Helper.generate_random_id()), block)
#         img_data.append(block)
#
# img = img_as_ubyte(io.imread("Wheat_Images/001.jpg"))
# roi_img = spectral_roi.extract_roi(img)
# Helper.block_proc(roi_img, (20,20), func)
img_data = Helper.unserialize("xxx.data")
# Display.show_image(img_data[100])
# Display.show_image(img_data[200])1
# Display.show_image(img_data[300])0
r = CNN.classify(img_data,model_file=None,featureRepresentation='glcm')
print(r)
print("COUNT: {}".format(r.tolist().count(1)))
