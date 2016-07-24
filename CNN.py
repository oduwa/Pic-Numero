import Display
import Helper
from skimage.color import rgb2gray
import numpy as np
from scipy import misc
from sklearn import svm, grid_search, metrics
from sklearn.neural_network import MLPClassifier
from skimage.feature import greycomatrix, greycoprops
from skimage import img_as_ubyte, io
from sklearn import decomposition
import matplotlib.pyplot as plt
import string
import random
import os, sys, shutil
import tqdm
import tensorflow as tf
from tensorflow.contrib import learn as skflow


# The name of the file where we will store serialized classifier
CLASSIFIER_FILE = 'Models/CNN_d1_a4'

#N_FEATURES = 16
#N_FEATURES = 400

def get_model(filename=CLASSIFIER_FILE):
    ''' Get CNN classifier object from file or create one if none exists on file.'''
    if(filename == None):
        # Load dataset
        train_data, train_targets, test_data, expected = Helper.unserialize("Datasets/old_data.data")
        #train_data, train_targets, test_data, expected = Helper.unserialize("Datasets/grain_raw.data")

        # Build Classifier
        # classifier = skflow.TensorFlowEstimator(model_fn=multilayer_conv_model, n_classes=2,
        #                                         steps=500, learning_rate=0.05, batch_size=128)
        # classifier = skflow.TensorFlowEstimator(model_fn=multilayer_conv_model, n_classes=2,
        #                                         steps=500, learning_rate=0.05, batch_size=8,
        #                                         optimizer='Ftrl')
        classifier = skflow.TensorFlowEstimator(model_fn=multilayer_conv_model, n_classes=2,
                                                steps=500, learning_rate=0.05, batch_size=8,
                                                optimizer='Ftrl')
        classifier.fit(train_data, train_targets)
        return classifier
    else:
        serialized_classifier = Helper.unserialize(filename)
        return serialized_classifier

# TensorFlow models using Scikit Flow ops
def conv_model(X, y):
    '''1-Layer CNN'''
    N_FEATURES = 16
    N_FILTERS = 20
    WINDOW_SIZE = 1
    X = tf.reshape(X, [-1, N_FEATURES, 1, 1])  # to form a 4d tensor of shape batch_size x n_features x 1 x 1
    features = skflow.ops.conv2d(X, N_FILTERS, [WINDOW_SIZE, 1], padding='VALID') # this will give me a sliding window/filter of size [WINDOW_SIZE x 1].
    features = tf.reduce_max(features, 1)
    #features = tf.nn.relu(features)
    pool = tf.squeeze(features, squeeze_dims=[1])
    return skflow.models.logistic_regression(pool, y)

def multilayer_conv_model(X, y):
    '''2-Layer CNN'''
    N_FEATURES = 16
    N_FILTERS = 20
    WINDOW_SIZE_1 = 2
    WINDOW_SIZE_2 = 1
    X = tf.reshape(X, [-1, N_FEATURES, 1, 1])  # to form a 4d tensor of shape batch_size x n_features x 1 x 1

    with tf.variable_scope('CNN_Layer1'):
        # this will give you sliding window/filter of size [WINDOW_SIZE x 1].
        features = skflow.ops.conv2d(X, N_FILTERS, [WINDOW_SIZE_1, 1], padding='SAME')
        pool1 = tf.nn.max_pool(features, ksize=[1, 8, 1, 1],
                               strides=[1, 4, 1, 1], padding='SAME')
        # Transpose matrix so that n_filters from convolution becomes width.
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])

    with tf.variable_scope('CNN_Layer2'):
        # Second level of convolution filtering.
        conv2 = skflow.ops.conv2d(pool1, N_FILTERS, [WINDOW_SIZE_2, 1], padding='VALID')
        pool2 = tf.reshape(tf.reduce_max(conv2, [1,2]), [-1, 20])
        return skflow.models.logistic_regression(pool2, y)


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


## featureRepresentation = {'image', 'pca', 'glcm'}
def classify(img, featureRepresentation='image', model_file=CLASSIFIER_FILE, shouldSaveResult=False):
    '''
    Classifies a sub-image as grain (1) or not grain (0).

    Args:
        img: Input sub-image or list of input sub-images.

        featureRepresentation: Type of features to be used in classification.
            Can ake of one of the values 'image', 'pca' or 'glcm'. Note that the
            classifier must have also been built using the same
            feature representation.

        model_file: filepath of serialized classifier to be used.

    Return:
        1 if grain and 0 otherwise.
    '''
    if(isinstance(img, np.ndarray)):
        img_features = None
        if(featureRepresentation == 'image'):
            img_features = img.flatten()
        elif(featureRepresentation == 'pca'):
            img_features = decomposition.PCA(n_components=8).fit_transform(img.flatten())
        elif(featureRepresentation == 'glcm'):
            img_features = get_textural_features(img, 1, True)
        clf = get_model(model_file)
        result = clf.predict(img_features.reshape(1,-1))

        if(shouldSaveResult == True):
            # Save image with result in filename
            if os.path.exists("Results"):
                shutil.rmtree("Results")
            os.makedirs("Results")
            io.imsave("Results/{}_{}.png".format(Helper.generate_random_id(8), result), img)

        return result
    elif(isinstance(img, list)):
        if(featureRepresentation == 'glcm'):
            sample_size = 16
        else:
            sample_size = 20*20

        test_data = np.zeros((len(img), sample_size))
        i = 0
        for image in img:
            if(featureRepresentation == 'image'):
                test_data[i] = image.flatten()
            elif(featureRepresentation == 'pca'):
                test_data[i] = decomposition.PCA(n_components=8).fit_transform(image.flatten())
            elif(featureRepresentation == 'glcm'):
                test_data[i] = get_textural_features(image, 1, True)
            i = i+1

        clf = get_model(model_file)
        result = clf.predict(test_data)

        if(shouldSaveResult == True):
            # Save image with result in filename
            if os.path.exists("Results"):
                shutil.rmtree("Results")
            os.makedirs("Results")
            for i in xrange(0,len(img)):
                io.imsave("Results/{}_{}.png".format(Helper.generate_random_id(8), result[i]), img[i])

        return result
    else:
        return None

def build_model(featureRepresentation='image', iters=10, glcm_distance=1, glcm_isMultidirectional=False):
    '''
    Creates, trains and serialises an MLP classifier.

    Args:
        featureRepresentation: Type of features to be used in classification.
            Can ake of one of the values 'image', 'pca' or 'glcm'.

        iters: Number of training iterations.

        glcm_distance: Distance between pixels for co-occurence. Only used if
            featureRepresentation=glcm.

        isMultidirectional: Controls whether co-occurence should be calculated
            in other directions (ie 45 degrees, 90 degrees and 135 degrees).
            Only used if featureRepresentation=glcm.
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
     Helper.serialize(ser_filename, dataset)
     return dataset

def experiment_with_parameters(ser_filename,
                               batch_sizes=[64],
                               learning_rates=[0.05],
                               optimizers=['Ftrl', 'RMSProp', 'Adam', 'Adagrad', 'SGD'],
                               class_weights=[[0.4,0.6], [0.6,0.4]]):
    '''
    Calculate and print accuracies for different combinations of hyper-paramters.
    '''
    # Load dataset
    train_data, train_targets, test_data, expected = Helper.unserialize(ser_filename)

    # Build Classifier
    for b_size in batch_sizes:
        for l_rate in learning_rates:
            for optimizer in optimizers:
                for class_weight in class_weights:
                    classifier = skflow.TensorFlowEstimator(model_fn=multilayer_conv_model, n_classes=2,
                                                            steps=500, learning_rate=l_rate, batch_size=b_size,
                                                            optimizer=optimizer, class_weight=class_weight)
                    classifier.fit(train_data, train_targets)

                    # Assess
                    predictions = classifier.predict(test_data)
                    accuracy = metrics.accuracy_score(expected, predictions)
                    confusion_matrix = metrics.confusion_matrix(expected, predictions)
                    print('Accuracy for batch_size %.2d learn_rate %.3f Cost Function %s: %f' % (b_size, l_rate, optimizer, accuracy))
                    print("Confusion matrix:\n%s" % confusion_matrix)


def run_with_dataset(ser_filename):
    '''
    Apply a CNN on a dataset and print test accuracies.
    That is, train it on training data and test it on test data.
    '''
    # Load dataset
    train_data, train_targets, test_data, expected = Helper.unserialize(ser_filename)

    # Build Classifier
    classifier = skflow.TensorFlowEstimator(model_fn=multilayer_conv_model, n_classes=2,
                                            steps=500, learning_rate=0.05, batch_size=128)
    classifier.fit(train_data, train_targets)

    # Assess
    predictions = classifier.predict(test_data)
    accuracy = metrics.accuracy_score(expected, predictions)
    confusion_matrix = metrics.confusion_matrix(expected, predictions)
    print("Confusion matrix:\n%s" % confusion_matrix)
    print('Accuracy: %f' % (accuracy))

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


def run(featureRepresentation='image', glcm_distance=1, glcm_isMultidirectional=False):
    '''
    Apply a CNN on the grain_images dataset and print test accuracies.
    That is, train it on training data and test it on test data.
    '''
    train_data, train_targets, test_data, expected = extract_features_from_new_data(featureRepresentation, glcm_distance, glcm_isMultidirectional, train_size=0.5)
    Helper.serialize("Datasets/grain_glcm_d1_a4_2_new.data", (train_data, train_targets, test_data, expected))

    # Build Classifier
    classifier = skflow.TensorFlowEstimator(model_fn=multilayer_conv_model, n_classes=2,
                                            steps=500, learning_rate=0.05, batch_size=128)
    classifier.fit(train_data, train_targets)

    # Assess
    predictions = classifier.predict(test_data)
    accuracy = metrics.accuracy_score(expected, predictions)
    confusion_matrix = metrics.confusion_matrix(expected, predictions)
    print("Confusion matrix:\n%s" % confusion_matrix)
    print('Accuracy: %f' % accuracy)


def main():
    #run('glcm', glcm_isMultidirectional=True)
    #save_feature_dataset("Datasets/grain_glcm_d1_a4.data", 'glcm', glcm_isMultidirectional=True)
    save_feature_dataset("Datasets/grain_glcm_d1_a4.data", 'glcm', glcm_isMultidirectional=True)
    #run_with_dataset("Datasets/grain_glcm_d1_a4_2_new.data")
    #experiment_with_parameters("Datasets/grain_glcm_d1_a4_2.data", batch_sizes=[4,8,16,32,64,128], learning_rates=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5])
    #experiment_with_parameters("Datasets/grain_glcm_d1_a4_2.data")
#main()
