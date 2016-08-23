import Display
from skimage.color import rgb2gray
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import string
import random
import pickle
import os, sys, shutil
from skimage.feature import greycomatrix, greycoprops
from skimage import img_as_ubyte, io

def generate_random_id(length=8):
    '''Returns a unique string of specified length'''
    identifier = ""
    for i in xrange(length):
        identifier = identifier + random.choice(string.ascii_letters + string.digits)
    return identifier

def block_proc(A, blockSize, blockFunc):
    '''
    Function to somewhat mimic behavior of MATLAB's blocproc function
    (See http://uk.mathworks.com/help/images/ref/blockproc.html).

    Creates a block (kernel), slides block across image and applies the
    specified function to each block.

    Args:
        A: 2D image array.
        blockSize: Tuple (width,height) specifying dimensions for each block.
        blockFunc: A lambda or function with the signature blockFunc(block).
            This should be implemented to specify operations to be performed on
            each block.
    '''
    xStart = 0;
    xStop = A.shape[1]
    if(xStop % blockSize[0] != 0):
        xStop = int(xStop/blockSize[0]) * blockSize[0]

    yStart = 0;
    yStop = A.shape[0]
    if(yStop % blockSize[1] != 0):
        yStop = int(yStop/blockSize[1]) * blockSize[1]

    for x in xrange(xStart, xStop, blockSize[0]):
        for y in xrange(yStart, yStop, blockSize[1]):
            block = A[y:y+blockSize[1], x:x+blockSize[0]]
            blockFunc(block)


def serialize(filename, obj):
    '''Save object to file'''
    f = open(filename, 'wb+')
    pickle.dump(obj, f)
    f.close()

def unserialize(filename):
    '''Return object from file'''
    try:
        f = open(filename, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj
    except:
        return None

def get_textural_features(img, isMultidirectional=False, distance=1):
    '''Extract GLCM feature vector from image
    Args:
        img: input image.

        isMultidirectional: Controls whether co-occurence should be calculated
            in other directions (ie 45 degrees, 90 degrees and 135 degrees).

        distance: Distance between pixels for co-occurence.

    Returns:
        features: if isMultidirectional=False, this is a 4 element vector of
        [dissimilarity, correlation,homogeneity, energy]. If not it is a 16
        element vector containing each of the above properties in each direction.
    '''
    if(isMultidirectional):
        img = img_as_ubyte(rgb2gray(img))
        glcm = greycomatrix(img, [distance], [0, 0.79, 1.57, 2.36], 256, symmetric=True, normed=True)
        dissimilarity_1 = greycoprops(glcm, 'dissimilarity')[0][0]
        dissimilarity_2 = greycoprops(glcm, 'dissimilarity')[0][1]
        dissimilarity_3 = greycoprops(glcm, 'dissimilarity')[0][2]
        dissimilarity_4 = greycoprops(glcm, 'dissimilarity')[0][3]
        correlation_1 = greycoprops(glcm, 'correlation')[0][0]
        correlation_2 = greycoprops(glcm, 'correlation')[0][1]
        correlation_3 = greycoprops(glcm, 'correlation')[0][2]
        correlation_4 = greycoprops(glcm, 'correlation')[0][3]
        homogeneity_1 = greycoprops(glcm, 'homogeneity')[0][0]
        homogeneity_2 = greycoprops(glcm, 'homogeneity')[0][1]
        homogeneity_3 = greycoprops(glcm, 'homogeneity')[0][2]
        homogeneity_4 = greycoprops(glcm, 'homogeneity')[0][3]
        energy_1 = greycoprops(glcm, 'energy')[0][0]
        energy_2 = greycoprops(glcm, 'energy')[0][1]
        energy_3 = greycoprops(glcm, 'energy')[0][2]
        energy_4 = greycoprops(glcm, 'energy')[0][3]
        feature = np.array([dissimilarity_1, dissimilarity_2, dissimilarity_3,\
         dissimilarity_4, correlation_1, correlation_2, correlation_3, correlation_4,\
         homogeneity_1, homogeneity_2, homogeneity_3, homogeneity_4, energy_1,\
         energy_2, energy_3, energy_4])
        return feature
    else:
        img = img_as_ubyte(rgb2gray(img))
        glcm = greycomatrix(img, [distance], [0], 256, symmetric=True, normed=True)
        dissimilarity = greycoprops(glcm, 'dissimilarity')[0][0]
        correlation = greycoprops(glcm, 'correlation')[0][0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0][0]
        energy = greycoprops(glcm, 'energy')[0][0]
        feature = np.array([dissimilarity, correlation, homogeneity, energy])
        return feature

def save_feature_dataset(ser_filename,
                         featureRepresentation='image',
                         glcm_distance=1,
                         glcm_isMultidirectional=False):
    '''
    Convenience method to extract features from images and serialize data.

    Args:
        ser_filename: name to store serialized dataset.

        featureRepresentation: Type of features to be used in classification.
            Can ake of one of the values 'image', 'pca' or 'glcm'.

        glcm_distance: Distance between pixels for co-occurence. Only used if
            featureRepresentation=glcm.

        isMultidirectional: Controls whether co-occurence should be calculated
            in other directions (ie 45 degrees, 90 degrees and 135 degrees).
            Only used if featureRepresentation=glcm.

    Return:
        dataset: Tuple containing (train_data, train_targets, test_data, test_targets)
    '''
    # Load train data
    train_filenames = []
    for filename in os.listdir("train/positive"):
     if(filename != ".DS_Store"): train_filenames.append("train/positive/" + filename)
    train_targets = [1]*(len(os.listdir("train/positive"))-1)

    for filename in os.listdir("train/negative"):
     if(filename != ".DS_Store"): train_filenames.append("train/negative/" + filename)
    train_targets = train_targets + [0]*(len(os.listdir("train/negative"))-1)

    n_train_samples = len(train_filenames)
    if(featureRepresentation == 'glcm'):
     if(glcm_isMultidirectional):
         sample_size = 16
     else:
         sample_size = 4
    else:
     sample_size = 20*20
    train_data = np.zeros((n_train_samples, sample_size))
    i = 0
    for filename in train_filenames:
     img = io.imread(filename)
     if(featureRepresentation == 'image'):
         train_data[i] = img.flatten()
     elif(featureRepresentation == 'pca'):
         train_data[i] = decomposition.PCA(n_components=8).fit_transform(img.flatten())
     elif(featureRepresentation == 'glcm'):
         train_data[i] = get_textural_features(img, glcm_distance, glcm_isMultidirectional)
     i = i + 1;


    # Load test data
    test_filenames = []
    expected = []
    for filename in os.listdir("test"):
     if(filename != ".DS_Store"):
         test_filenames.append("test/" + filename)
         expected.append(int(filename.split('_')[1].split('.')[0]))

    n_test_samples = len(test_filenames)
    test_data = np.zeros((n_test_samples, sample_size))
    i = 0
    for filename in test_filenames:
     img = io.imread(filename)
     if(featureRepresentation == 'image'):
         test_data[i] = img.flatten()
     elif(featureRepresentation == 'pca'):
         test_data[i] = decomposition.PCA(n_components=8).fit_transform(img.flatten())
     elif(featureRepresentation == 'glcm'):
         test_data[i] = get_textural_features(img, glcm_distance, glcm_isMultidirectional)
     i = i + 1;

     dataset = (train_data, train_targets, test_data, expected)
     serialize(ser_filename, dataset)
     return dataset

def extract_features_from_old_data(featureRepresentation='image', glcm_distance=1, glcm_isMultidirectional=False):
    '''
    Convenience method to extract features from images in "train" and "test"
    foldersand serialize data.

    Args:
        ser_filename: name to store serialized dataset.

        featureRepresentation: Type of features to be used in classification.
            Can ake of one of the values 'image', 'pca' or 'glcm'.

        glcm_distance: Distance between pixels for co-occurence. Only used if
            featureRepresentation=glcm.

        isMultidirectional: Controls whether co-occurence should be calculated
            in other directions (ie 45 degrees, 90 degrees and 135 degrees).
            Only used if featureRepresentation=glcm.

    Return:
        dataset: Tuple containing (train_data, train_targets, test_data, test_targets)
    '''
    # Load train data
    train_filenames = []
    for filename in os.listdir("train/positive"):
        if(filename != ".DS_Store"): train_filenames.append("train/positive/" + filename)
    train_targets = [1]*(len(os.listdir("train/positive"))-1)

    for filename in os.listdir("train/negative"):
        if(filename != ".DS_Store"): train_filenames.append("train/negative/" + filename)
    train_targets = train_targets + [0]*(len(os.listdir("train/negative"))-1)

    n_train_samples = len(train_filenames)
    if(featureRepresentation == 'glcm'):
        if(glcm_isMultidirectional):
            sample_size = 16
        else:
            sample_size = 4
    else:
        sample_size = 20*20
    train_data = np.zeros((n_train_samples, sample_size))
    i = 0
    for filename in train_filenames:
        img = io.imread(filename)
        if(featureRepresentation == 'image'):
            train_data[i] = img.flatten()
        elif(featureRepresentation == 'pca'):
            train_data[i] = decomposition.PCA(n_components=8).fit_transform(img.flatten())
        elif(featureRepresentation == 'glcm'):
            train_data[i] = get_textural_features(img, glcm_distance, glcm_isMultidirectional)
        i = i + 1;


    # Load test data
    test_filenames = []
    expected = []
    for filename in os.listdir("test"):
        if(filename != ".DS_Store"):
            test_filenames.append("test/" + filename)
            expected.append(int(filename.split('_')[1].split('.')[0]))

    n_test_samples = len(test_filenames)
    test_data = np.zeros((n_test_samples, sample_size))
    i = 0
    for filename in test_filenames:
        img = io.imread(filename)
        if(featureRepresentation == 'image'):
            test_data[i] = img.flatten()
        elif(featureRepresentation == 'pca'):
            test_data[i] = decomposition.PCA(n_components=8).fit_transform(img.flatten())
        elif(featureRepresentation == 'glcm'):
            test_data[i] = get_textural_features(img, glcm_distance, glcm_isMultidirectional)
        i = i + 1;

    dataset = (train_data, train_targets, test_data, expected)
    return dataset

def extract_features_from_new_data(featureRepresentation='image', glcm_distance=1, glcm_isMultidirectional=False, train_size=0.75):
    '''
    Convenience method to extract features from images in "grain_images"
    folder and serialize data.

    Args:
        ser_filename: name to store serialized dataset.

        featureRepresentation: Type of features to be used in classification.
            Can ake of one of the values 'image', 'pca' or 'glcm'.

        glcm_distance: Distance between pixels for co-occurence. Only used if
            featureRepresentation=glcm.

        isMultidirectional: Controls whether co-occurence should be calculated
            in other directions (ie 45 degrees, 90 degrees and 135 degrees).
            Only used if featureRepresentation=glcm.

        train_size: Fraction of dataset to be used for training. The remainder is
            used for testing.

    Return:
        dataset: Tuple containing (train_data, train_targets, test_data, test_targets)
    '''
    image_filenames = []
    expected = []
    targets = []
    for filename in os.listdir("grain_images"):
        if(filename != ".DS_Store"):
            image_filenames.append("grain_images/" + filename)
            targets.append(int(filename.split('_')[1].split('.')[0]))

    if(featureRepresentation == 'glcm'):
        if(glcm_isMultidirectional):
            sample_size = 16
        else:
            sample_size = 4
    else:
        sample_size = 20*20
    train_filenames = image_filenames[:int(train_size*len(image_filenames))]
    test_filenames = image_filenames[int(train_size*len(image_filenames)):]

    train_data = np.zeros((len(train_filenames), sample_size))
    train_targets = targets[:int(train_size*len(targets))]

    test_data = np.zeros((len(test_filenames), sample_size))
    test_targets = targets[int(train_size*len(targets)):]

    for i in xrange(0,len(train_filenames)):
        img = io.imread(train_filenames[i])
        if(featureRepresentation == 'image'):
            train_data[i] = img.flatten()
        elif(featureRepresentation == 'pca'):
            train_data[i] = decomposition.PCA(n_components=8).fit_transform(img.flatten())
        elif(featureRepresentation == 'glcm'):
            train_data[i] = get_textural_features(img, glcm_distance, glcm_isMultidirectional)

    for i in xrange(0,len(test_filenames)):
        img = io.imread(test_filenames[i])
        if(featureRepresentation == 'image'):
            test_data[i] = img.flatten()
        elif(featureRepresentation == 'pca'):
            test_data[i] = decomposition.PCA(n_components=8).fit_transform(img.flatten())
        elif(featureRepresentation == 'glcm'):
            test_data[i] = get_textural_features(img, glcm_distance, glcm_isMultidirectional)

    print("EXTRACTED FEATURES FROM IMAGES AND STORED AS DATASET\n1: {} | 0: {}".format(train_targets.count(1), train_targets.count(0)))

    return (train_data, train_targets, test_data, test_targets)
