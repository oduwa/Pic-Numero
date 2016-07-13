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
import os, sys
import tqdm


# The name of the file where we will store serialized classifier
MLP_FILE = 'Models/MLP_glcmdistance1.data'#'Models/MLP.data'

def get_textural_features(img):
    img = img_as_ubyte(rgb2gray(img))
    glcm = greycomatrix(img, [1], [0], 256, symmetric=True, normed=True)
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    feature = np.array([dissimilarity, correlation, homogeneity, energy])
    return feature

## featureRepresentation = {'image', 'pca', 'glcm'}
def main(featureRepresentation='image'):
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
            train_data[i] = ecomposition.PCA(n_components=8).fit_transform(img.flatten())
        elif(featureRepresentation == 'glcm'):
            train_data[i] = get_textural_features(img)
        i = i + 1;

    # Apply pca to compute reduced representation in case needed
    #train_data_reduced = decomposition.PCA(n_components=8).fit_transform(train_data)


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
            test_data[i] = ecomposition.PCA(n_components=8).fit_transform(img.flatten())
        elif(featureRepresentation == 'glcm'):
            test_data[i] = get_textural_features(img)
        i = i + 1;

    # Apply pca to compute reduced representation in case needed
    #test_data_reduced = decomposition.PCA(n_components=8).fit_transform(test_data)



    # Create a classifier: a support vector classifier
    # param_grid = {'C': [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.05, 0.01, 0.5, 0.1], 'kernel': ['rbf', 'poly'] }
    # clf = grid_search.GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
    # clf.fit(train_data, train_targets)
    # print(clf.best_estimator_)
    # classifier = clf.best_estimator_
    #classifier = svm.SVC()
    classifier = MLPClassifier()
    param_grid = {"algorithm":["l-bfgs", "sgd", "adam"], "activation":["logistic", "relu", "tanh"], "hidden_layer_sizes":[(5,2), (5), (100), (150), (200)] }
    clf = grid_search.GridSearchCV(MLPClassifier(), param_grid)
    clf.fit(train_data, train_targets)
    print(clf);
    classifier = clf

    # Get previous model and assess
    serialized_classifier = Helper.unserialize(MLP_FILE)
    predictions = serialized_classifier.predict(test_data)
    confusion_matrix = metrics.confusion_matrix(expected, predictions)
    print("Old Confusion matrix:\n%s" % confusion_matrix)
    serialized_n_correct = confusion_matrix[0][0] + confusion_matrix[1][1]
    predictions = classifier.predict(test_data)
    confusion_matrix = metrics.confusion_matrix(expected, predictions)
    print("New Confusion matrix:\n%s" % confusion_matrix)
    n_correct = confusion_matrix[0][0] + confusion_matrix[1][1]
    if(n_correct > serialized_n_correct):
        Helper.serialize(MLP_FILE, classifier)
        print("SAVED MODEL")

## featureRepresentation = {'image', 'pca', 'glcm'}
def generate_model(featureRepresentation='image', iters=10):
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
            train_data[i] = ecomposition.PCA(n_components=8).fit_transform(img.flatten())
        elif(featureRepresentation == 'glcm'):
            train_data[i] = get_textural_features(img)
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
            test_data[i] = ecomposition.PCA(n_components=8).fit_transform(img.flatten())
        elif(featureRepresentation == 'glcm'):
            test_data[i] = get_textural_features(img)
        i = i + 1;



    # Perform build iterations
    for i in tqdm.tqdm(range(0, iters)):
        # Build Classifier
        param_grid = {"algorithm":["l-bfgs", "sgd", "adam"], "activation":["logistic", "relu", "tanh"], "hidden_layer_sizes":[(5,2), (5), (100), (150), (200)] }
        classifier = grid_search.GridSearchCV(MLPClassifier(), param_grid)
        classifier.fit(train_data, train_targets)

        # Get previous classifier and assess
        serialized_classifier = Helper.unserialize(MLP_FILE)
        if(serialized_classifier):
            predictions = serialized_classifier.predict(test_data)
            confusion_matrix = metrics.confusion_matrix(expected, predictions)
            serialized_n_correct = confusion_matrix[0][0] + confusion_matrix[1][1]
            predictions = classifier.predict(test_data)
            confusion_matrix = metrics.confusion_matrix(expected, predictions)
            n_correct = confusion_matrix[0][0] + confusion_matrix[1][1]
            if(n_correct > serialized_n_correct):
                Helper.serialize(MLP_FILE, classifier)
        else:
            Helper.serialize(MLP_FILE, classifier)

    # Display final model performance
    serialized_classifier = Helper.unserialize(MLP_FILE)
    predictions = serialized_classifier.predict(test_data)
    confusion_matrix = metrics.confusion_matrix(expected, predictions)
    print("Old Confusion matrix:\n%s" % confusion_matrix)






#main('glcm');
generate_model('glcm', iters=20)
