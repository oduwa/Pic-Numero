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
#https://www.tensorflow.org/versions/r0.10/tutorials/deep_cnn/index.html


# The name of the file where we will store serialized classifier
CLASSIFIER_FILE = 'Models/CNN_d1_a4'

def get_model(filename=CLASSIFIER_FILE):
    ''' Get CNN classifier object from file or create one if none exists on file.'''
    if(filename == None):
        # Load dataset
        train_data, train_targets, test_data, expected = Helper.unserialize("Datasets/raw_new_80.data")
        train_data2, train_targets2, test_data2, expected2 = Helper.unserialize("Datasets/raw.data")

        train_data = np.concatenate((train_data, train_data2), axis=0)
        train_targets = np.concatenate((train_targets, train_targets2), axis=0)
        test_data = np.concatenate((test_data, test_data2), axis=0)
        expected = np.concatenate((expected, expected2), axis=0)
        print(train_data.shape)

        raw_train_data = np.zeros((train_data.shape[0], 20, 20))
        i = 0
        for item in train_data:
            raw_train_data[i] = item.reshape((20,20))
            #Display.show_image(raw_train_data[i])
            i = i+1

        raw_test_data = np.zeros((test_data.shape[0], 20, 20))
        i = 0
        for item in test_data:
            raw_test_data[i] = item.reshape((20,20))
            #Display.show_image(raw_test_data[i])
            i = i+1


        # Build Classifier
        # classifier = skflow.TensorFlowEstimator(model_fn=multilayer_conv_model, n_classes=2,
        #                                         steps=500, learning_rate=0.05, batch_size=128)
        classifier = skflow.TensorFlowEstimator(model_fn=conv_model, n_classes=2,
                                                steps=500, learning_rate=0.05, batch_size=128,
                                                optimizer='Ftrl')
        classifier.fit(raw_train_data, train_targets)

        # Assess built classifier
        predictions = classifier.predict(raw_test_data)
        accuracy = metrics.accuracy_score(expected, predictions)
        confusion_matrix = metrics.confusion_matrix(expected, predictions)
        print("Confusion matrix:\n%s" % confusion_matrix)
        print('Accuracy: %f' % accuracy)

        return classifier
    else:
        serialized_classifier = Helper.unserialize(filename)
        return serialized_classifier



def conv_modell(X, y):
    print("X BEFORE EXANSION: {}".format(X.get_shape()))
    X = tf.expand_dims(X, 3)
    print("X AFTER EXANSION: {}".format(X.get_shape()))

    N_FILTERS = 8
    #conv1 = skflow.ops.conv2d(X, N_FILTERS, [4, 4], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = skflow.ops.conv2d(X, N_FILTERS, [4, 4], strides=[1, 1, 1, 1], padding='SAME')
    print("CONV1: {}".format(conv1.get_shape()))

    pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print("MAX POOL: {}".format(pool.get_shape()))

    pool1 = tf.reduce_max(conv1, [1, 2])
    print("POOL BEFORE RESHAPE: {}".format(pool1.get_shape()))
    pool1 = tf.reshape(pool1, [-1, N_FILTERS])
    print("POOL AFTER RESHAPE: {}".format(pool1.get_shape()))
    return skflow.models.logistic_regression(pool1, y)

def conv_model(X, y):
    print("X BEFORE EXANSION: {}".format(X.get_shape()))
    X = tf.expand_dims(X, 3)
    print("X AFTER EXANSION: {}".format(X.get_shape()))

    N_FILTERS = 12
    CONV_WINDOW_1 = [2, 2]
    CONV_STRIDE_1 = [1, 1, 1, 1]
    POOLING_WINDOW = [1, 2, 2, 1]
    POOLING_STRIDE = [1, 2, 2, 1]

    CONV_WINDOW_2 = [2, 2]
    CONV_STRIDE_2 = [1, 1, 1, 1]

    with tf.variable_scope('CNN_Layer1'):
        #conv1 = skflow.ops.conv2d(X, N_FILTERS, [4, 4], strides=[1, 4, 4, 1], padding='SAME')
        conv1 = skflow.ops.conv2d(X, N_FILTERS, CONV_WINDOW_1, strides=CONV_STRIDE_1, padding='SAME')
        print("CONV1: {}".format(conv1.get_shape()))
        #conv1 = tf.nn.relu(conv1)
        pool = tf.nn.max_pool(conv1, ksize=POOLING_WINDOW, strides=POOLING_STRIDE, padding='VALID')
        print("MAX POOL: {}".format(pool.get_shape()))

    with tf.variable_scope('CNN_Layer2'):
        conv2 = skflow.ops.conv2d(pool, N_FILTERS, CONV_WINDOW_2, CONV_STRIDE_2, padding='SAME')
        #conv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print("MAX POOL2: {}".format(pool2.get_shape()))
        # features = tf.reduce_max(pool2, [2])
        # print("POOL BEFORE RESHAPE: {}".format(features.get_shape()))
        # features = tf.reshape(features, [-1, N_FILTERS*features.get_shape()[1].value])
        # print("POOL AFTER RESHAPE: {}".format(features.get_shape()))
        # return skflow.models.logistic_regression(features, y)

    with tf.variable_scope('CNN_Layer3'):
        conv3 = skflow.ops.conv2d(pool2, N_FILTERS, [2, 2], strides=[1, 1, 1, 1], padding='SAME')
        pool3 = tf.nn.avg_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        print("MAX POOL3: {}".format(pool3.get_shape()))
        features = pool3
        #features = tf.reduce_max(features, [1])
        features = tf.reduce_mean(features, [1,2])
        print("POOL BEFORE RESHAPE: {}".format(features.get_shape()))
        #features = tf.reshape(features, [-1, N_FILTERS * features.get_shape()[1].value * features.get_shape()[2].value])
        features = tf.reshape(features, [-1, N_FILTERS])
        print("POOL AFTER RESHAPE: {}".format(features.get_shape()))

    return skflow.models.logistic_regression(features, y)


# # TensorFlow models using Scikit Flow ops
# def conv_model(X, y):
#     '''1-Layer CNN'''
#     N_FEATURES = 16
#     N_FILTERS = 20
#     WINDOW_SIZE = 1
#     X = tf.reshape(X, [-1, N_FEATURES, 1, 1])  # to form a 4d tensor of shape batch_size x n_features x 1 x 1
#     features = skflow.ops.conv2d(X, N_FILTERS, [WINDOW_SIZE, 1], padding='VALID') # this will give me a sliding window/filter of size [WINDOW_SIZE x 1].
#     features = tf.reduce_max(features, 1)
#     #features = tf.nn.relu(features)
#     pool = tf.squeeze(features, squeeze_dims=[1])
#     return skflow.models.logistic_regression(pool, y)
#
# def multilayer_conv_model(X, y):
#     '''2-Layer CNN'''
#     N_FEATURES = 16
#     N_FILTERS = 20
#     WINDOW_SIZE_1 = 2
#     WINDOW_SIZE_2 = 1
#     X = tf.reshape(X, [-1, N_FEATURES, 1, 1])  # to form a 4d tensor of shape batch_size x n_features x 1 x 1
#
#     with tf.variable_scope('CNN_Layer1'):
#         # this will give you sliding window/filter of size [WINDOW_SIZE x 1].
#         features = skflow.ops.conv2d(X, N_FILTERS, [WINDOW_SIZE_1, 1], padding='SAME')
#         pool1 = tf.nn.max_pool(features, ksize=[1, 8, 1, 1],
#                                strides=[1, 4, 1, 1], padding='SAME')
#         # Transpose matrix so that n_filters from convolution becomes width.
#         pool1 = tf.transpose(pool1, [0, 1, 3, 2])
#
#     with tf.variable_scope('CNN_Layer2'):
#         # Second level of convolution filtering.
#         conv2 = skflow.ops.conv2d(pool1, N_FILTERS, [WINDOW_SIZE_2, 1], padding='VALID')
#         pool2 = tf.reshape(tf.reduce_max(conv2, [1,2]), [-1, 20])
#         return skflow.models.logistic_regression(pool2, y)


## featureRepresentation = {'image', 'pca', 'glcm'}
def classify(img, featureRepresentation='image', model_file=CLASSIFIER_FILE, shouldSaveResult=False):
    '''
    Classifies a sub-image or list of sub-images as grain (1) or not grain (0).

    Args:
        img: Input sub-image or list of input sub-images.

        featureRepresentation: Type of features to be used in classification.
            Can ake of one of the values 'image', 'pca' or 'glcm'. Note that the
            classifier must have also been built using the same
            feature representation.

        model_file: filepath of serialized classifier to be used.

        shouldSaveResult: If this boolean flag is set to true, this function
            will save the sub-images and their classifictions to the "Results"
            folder after classification.

    Return:
        scalar or list of 1 if grain and 0 otherwise.
    '''
    if(isinstance(img, np.ndarray)):
        img_features = None
        if(featureRepresentation == 'image'):
            img_features = img.flatten()
        elif(featureRepresentation == 'pca'):
            img_features = decomposition.PCA(n_components=8).fit_transform(img.flatten())
        elif(featureRepresentation == 'glcm'):
            img_features = Helper.get_textural_features(img, 1, True)
        clf = get_model(model_file)
        return clf.predict(img_features.reshape(1,-1))
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
                test_data[i] = Helper.get_textural_features(image, 1, True)
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
    else:
        return None

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

def run(featureRepresentation='image', glcm_distance=1, glcm_isMultidirectional=False):
    '''
    Apply a CNN on the grain_images dataset and print test accuracies.
    That is, train it on training data and test it on test data.
    '''
    train_data, train_targets, test_data, expected = Helper.extract_features_from_new_data(featureRepresentation, glcm_distance, glcm_isMultidirectional, train_size=0.5)
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
    #run_with_dataset("Datasets/grain_glcm_d1_a4_2_new.data")
    experiment_with_parameters("Datasets/grain_glcm_d1_a4_2.data", batch_sizes=[4,8,16,32,64,128], learning_rates=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5])
    #experiment_with_parameters("Datasets/grain_glcm_d1_a4_2.data")
