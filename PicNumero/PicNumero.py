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
import glcm
import MLP
import CNN
import SVM
import spectral_roi
import Helper
import Display

# Initialise list for storing sub-image (ie block data).
img_data = []

# Block processing function
def blockfunc(block):
    global img_data

    # Check if not all zeros
    if(numpy.any(block)):
        #io.imsave("Block2/{}.png".format(Helper.generate_random_id()), block)
        img_data.append(block)


################### COUNTING BY REGRESSION #####################################

def run_with_glcm(image_filename="../Wheat_Images/004.jpg"):
    '''
	Estimates the number of grains in a given image using a
	regression approach and glcm features.

	Args:
		image_filename: The path to the image from which a grain count
			is to be obtained.

	Returns:
		count: An estimate of the number of grains in the provided image.
	'''
    model = glcm.train()
    count = glcm.count(image_filename, model)
    print("COUNT: {}".format(count))
    return count

################### COUNTING BY DETECTION #####################################

def run_with_svm(image_filename="../Wheat_Images/004.jpg", ser_filename=None):
    '''
	Estimates the number of grains in a given image using a
	Support Vector Machine.

	Args:
		image_filename: The path to the image from which a grain count
			is to be obtained.

		ser_filename: path to serialized list of isub-images already extracted
		from the image from which a grain count is to be obtained.

	Returns:
		count: An estimate of the number of grains in the provided image.
	'''

    global img_data

    # Chop image up into sub-images and serilaise or just load serialised data if
    # it already exists.
    if(ser_filename == None and image_filename == "../Wheat_Images/004.jpg"):
		ser_filename = "../Wheat_Images/xxx_004.data"
    if(Helper.unserialize(ser_filename) == None):
        img = img_as_ubyte(io.imread(image_filename))
        roi_img = spectral_roi.extract_roi(img, [1])
        Helper.block_proc(roi_img, (20,20), blockfunc)
        #Helper.serialize(ser_filename, img_data)
    else:
        img_data = Helper.unserialize(ser_filename)

    # classify
    r = SVM.classify(img_data, featureRepresentation='glcm', shouldSaveResult=True)

    # Count number of '1s' in the result and return
    count = r.tolist().count(1)
    print("COUNT: {}".format(count))
    return count

def run_with_mlp(image_filename="../Wheat_Images/004.jpg", ser_filename=None):
    '''
	Estimates the number of grains in a given image using a
	Multilayer Perceptron neural network.

	Args:
		image_filename: The path to the image from which a grain count
			is to be obtained.

		ser_filename: path to serialized list of isub-images already extracted
		from the image from which a grain count is to be obtained.

	Returns:
		count: An estimate of the number of grains in the provided image.
    '''
    global img_data

    # Chop image up into sub-images and serilaise or just load serialised data if
    # it already exists.
    if(ser_filename == None and image_filename == "../Wheat_Images/004.jpg"):
		ser_filename = "../Wheat_Images/xxx_004.data"
    if(Helper.unserialize(ser_filename) == None):
        img = img_as_ubyte(io.imread(image_filename))
        roi_img = spectral_roi.extract_roi(img, [1])
        Helper.block_proc(roi_img, (20,20), blockfunc)
        #Helper.serialize(ser_filename, img_data)
    else:
        img_data = Helper.unserialize(ser_filename)

    # classify
    #MLP.build_model('glcm', iters=30, glcm_isMultidirectional=True)
    r = MLP.classify(img_data, featureRepresentation='glcm', shouldSaveResult=True)

    # Count number of '1s' in the result and return
    count = r.tolist().count(1)
    print("COUNT: {}".format(count))
    return count

def run_with_cnn(image_filename="../Wheat_Images/004.jpg", ser_filename=None):
    '''
	Estimates the number of grains in a given image using a
	Convolutional neural network.

	Args:
		image_filename: The path to the image from which a grain count
			is to be obtained.

		ser_filename: path to serialized list of isub-images already extracted
		from the image from which a grain count is to be obtained.

	Returns:
		count: An estimate of the number of grains in the provided image.
	'''

    global img_data

    # Chop image up into sub-images and serilaise or just load serialised data if
    # it already exists.
    if(ser_filename == None and image_filename == "../Wheat_Images/004.jpg"):
		ser_filename = "../Wheat_Images/xxx_004.data"
    if(Helper.unserialize(ser_filename) == None):
        img = img_as_ubyte(io.imread(image_filename))
        roi_img = spectral_roi.extract_roi(img, [1])
        Helper.block_proc(roi_img, (20,20), blockfunc)
        #Helper.serialize(ser_filename, img_data)
    else:
        img_data = Helper.unserialize(ser_filename)

    # classify
    r = CNN.classify(img_data[0], model_file=None,featureRepresentation='glcm', shouldSaveResult=True)

    # Count number of '1s' in the result and return
    count = r.tolist().count(1)
    print("COUNT: {}".format(count))
    return count


