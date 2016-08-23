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
import os, sys, shutil, timeit
import tqdm

# The name of the file where we will store serialized classifier
SVM_FILE = 'Models/SVM_d1_a4_olddata.data'

def get_model(filename=SVM_FILE):
    ''' Fetch SVM classifier object from file'''
    classifier = Helper.unserialize(filename)

    if(classifier == None):
        classifier = build_model('glcm', dataset_file='Datasets/old_data.data')
        Helper.serialize(filename, classifier)

    return classifier

def build_model(featureRepresentation='image', dataset_file=None, iters=10, glcm_distance=1, glcm_isMultidirectional=True):
    '''
    Creates, trains and serialises an MLP classifier.

    Args:
        featureRepresentation: Type of features to be used in classification.
            Can ake of one of the values 'image', 'pca' or 'glcm'.

        dataset_file: filename of serialized data set upon which to build the
            SVM. If none, default dataset is used.

        iters: Number of training iterations.

        glcm_distance: Distance between pixels for co-occurence. Only used if
            featureRepresentation=glcm.

        isMultidirectional: Controls whether co-occurence should be calculated
            in other directions (ie 45 degrees, 90 degrees and 135 degrees).
            Only used if featureRepresentation=glcm.
    '''
    if(dataset_file == None):
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
                train_data[i] = Helper.get_textural_features(img, glcm_distance, glcm_isMultidirectional)
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
                test_data[i] = Helper.get_textural_features(img, glcm_distance, glcm_isMultidirectional)
            i = i + 1;
    else:
        (train_data, train_targets, test_data, expected) = Helper.unserialize(dataset_file)

    # Perform build iterations
    for i in tqdm.tqdm(range(0, iters)):
        # Build Classifier
        param_grid = {'C': [1e0, 5e0, 1e1, 5e1, 1e2], 'kernel': ['rbf', 'poly'] }
        classifier = svm.SVC(kernel='rbf')#grid_search.GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
        classifier.fit(train_data, train_targets)

        # Get previous classifier and assess
        serialized_classifier = Helper.unserialize(SVM_FILE)
        if(serialized_classifier):
            predictions = serialized_classifier.predict(test_data)
            confusion_matrix = metrics.confusion_matrix(expected, predictions)
            serialized_n_correct = confusion_matrix[0][0] + confusion_matrix[1][1]
            predictions = classifier.predict(test_data)
            confusion_matrix = metrics.confusion_matrix(expected, predictions)
            n_correct = confusion_matrix[0][0] + confusion_matrix[1][1]
            if(n_correct > serialized_n_correct):
                Helper.serialize(SVM_FILE, classifier)
        else:
            Helper.serialize(SVM_FILE, classifier)

    # Display final model performance
    serialized_classifier = Helper.unserialize(SVM_FILE)
    predictions = serialized_classifier.predict(test_data)
    confusion_matrix = metrics.confusion_matrix(expected, predictions)
    print("Confusion matrix:\n%s" % confusion_matrix)
    print("Accuracy: %f" % metrics.accuracy_score(expected, predictions))
    return serialized_classifier

## featureRepresentation = {'image', 'pca', 'glcm'}
def classify(img, featureRepresentation='image', shouldSaveResult=False):
    '''
    Classifies a sub-image as grain (1) or not grain (0).

    Args:
        img: input sub-image.

        featureRepresentation: Type of features to be used in classification.
            Can ake of one of the values 'image', 'pca' or 'glcm'. Note that the
            classifier must have also been built using the same
            feature representation.

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
        clf = get_model()
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
                test_data[i] = Helper.get_textural_features(image, 1, True)
            i = i+1

        clf = get_model()
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


def main():
    dataset = Helper.extract_features_from_old_data(featureRepresentation='glcm', glcm_distance=1, glcm_isMultidirectional=True)
    Helper.serialize("Datasets/old_data.data", dataset)

#main();
#print(timeit.timeit("SVM.build_model(\'glcm\', dataset_file=\"Datasets/old_data.data\", iters=50, glcm_isMultidirectional=True)", setup="import SVM",  number=1))
