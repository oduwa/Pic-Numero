import numpy as np
from scipy import misc
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
import Display
import matplotlib.pyplot as plt
from skimage import data
from skimage import img_as_ubyte
from sklearn import linear_model

PATCH_SIZE = 50
FEATURE_SIZE = 4
COUNTS = [65*22, 80*22, 87*22, 40*22, 71*22, 58*22, 77*22, 68*22, 67*22, 61*22, 46*22, 57*22] # started with 1 and 2
LIN_REGRESSION_MODEL_NAME = "Models/regression.data"

def train():
    '''
    Builds linear regression from wheat images using GLCM properties.

    Returns:
        linear regression model
    '''
    numberOfImages = 12;

    # TODO: AUTOMATICALLY GET NUMBER OF IMAGES
    # Get number of images. Remeber to divide by 2 as for every relevant image,
    # theres also the comparison image.
    # if ".DS_Store" in os.listdir("Wheat_ROIs"):
    #     numberOfImages = (len(os.listdir("Wheat_ROIs")) - 1)/2;
    # else:
    #     numberOfImages = len(os.listdir("Wheat_ROIs"))/2;

    featureList = np.zeros((numberOfImages, FEATURE_SIZE))

    # For each ROI image in folder
    for i in range(1, numberOfImages+1):
        # Load image
        filename = "Wheat_Images/{:03d}.jpg".format(i);
        img = misc.imread(filename);
        img_gray = img_as_ubyte(rgb2gray(img));

        glcm = greycomatrix(img_gray, [5], [0], 256, symmetric=True, normed=True)
        dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
        correlation = greycoprops(glcm, 'correlation')[0, 0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
        energy = greycoprops(glcm, 'energy')[0, 0]
        feature = np.array([dissimilarity, correlation, homogeneity, energy])
        featureList[i-1] = feature
        #print("{} = {}A + {}B + {}C + {}D".format(filename, dissimilarity, correlation, homogeneity, energy))
        #print(feature)

    # Build regression model
    regression_model = linear_model.LinearRegression()
    regression_model.fit(featureList, COUNTS[:numberOfImages])
    Helper.serialize(LIN_REGRESSION_MODEL_NAME, regression_model)
    print("COEFF: {}\nINTERCEPT: {}".format(regression_model.coef_, regression_model.intercept_))
    print("SCORE: {}".format(regression_model.score(featureList, COUNTS[:numberOfImages])))
    return regression_model

def count(filename, model):
    '''
    Returns an estimate of the number of grains in a given wheat image.

    Args:
        filename: Name of image file containing grains to be counted.

        model: regression model for estimating count
    Returns:
        estimation of the number of grains in image.
    '''
    img = misc.imread(filename);
    img_gray = img_as_ubyte(rgb2gray(img));

    glcm = greycomatrix(img_gray, [5], [0], 256, symmetric=True, normed=True)
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    feature = np.array([dissimilarity, correlation, homogeneity, energy])

    count = model.predict(feature)
    return count

def plot_glcm_properties():
    '''Plots different GLCM properties of different areas of the wheat image
    in order to visualise how the GLCM can split/discriminate between them.'''
    image = img_as_ubyte(rgb2gray(misc.imread("Assets/wheat.png")));
    #Display.show_image(image, isGray=True)

    # select some patches from different areas of the image
    spikelet_locations = [(643, 517), (877, 574), (2129, 649), (1342, 454)]
    spikelet_patches = []

    stalk_locations = [(534, 1056), (1017, 857), (1711, 1365), (2199, 1093)]
    stalk_patches = []

    # Extract patches
    for loc in spikelet_locations:
        spikelet_patches.append(image[loc[1]:loc[1] + PATCH_SIZE, loc[0]:loc[0] + PATCH_SIZE])

    for loc in stalk_locations:
        stalk_patches.append(image[loc[1]:loc[1] + PATCH_SIZE, loc[0]:loc[0] + PATCH_SIZE])


    # compute some GLCM properties each patch
    xs = []
    ys = []
    for patch in (spikelet_patches + stalk_patches):
        glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
        xs.append(greycoprops(glcm, 'correlation')[0, 0])
        ys.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        print("({}, {})".format(greycoprops(glcm, 'dissimilarity')[0, 0], greycoprops(glcm, 'correlation')[0, 0]))

    # create the figure
    fig = plt.figure(figsize=(8, 8))

    # display original image with locations of patches
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
              )
    for (x,y) in spikelet_locations:
        ax.plot(x, y, 'gs', markersize=PATCH_SIZE/4)
    for (x,y) in stalk_locations:
        ax.plot(x, y, 'bs', markersize=PATCH_SIZE/4)

    ax.set_xlabel('Original Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

    # for each patch, plot (dissimilarity, correlation)
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(xs[:len(spikelet_patches)], ys[:len(spikelet_patches)], 'go',
            label='Grains')
    ax.plot(xs[len(stalk_patches):], ys[len(stalk_patches):], 'bo',
            label='Stalk')
    ax.set_xlabel('GLCM Correlation')
    ax.set_ylabel('GLVM Dissimilarity')
    ax.legend()

    # display the image patches
    for i, patch in enumerate(spikelet_patches):
        ax = fig.add_subplot(3, len(spikelet_patches), len(spikelet_patches)*1 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
                  )
        ax.set_xlabel('Grains %d' % (i + 1))

    for i, patch in enumerate(stalk_patches):
        ax = fig.add_subplot(3, len(stalk_patches), len(stalk_patches)*2 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
                  )
        ax.set_xlabel('Stalk %d' % (i + 1))

    plt.show()

def plot_residuals():
    numberOfImages = 12
    residuals = []
    featureList = np.zeros((numberOfImages, FEATURE_SIZE))
    model = get_model()

    # Get feautures
    for i in range(1, numberOfImages):
        # Load image
        filename = "Wheat_Images/{:03d}.jpg".format(i);
        img = misc.imread(filename);
        img_gray = img_as_ubyte(rgb2gray(img));
        glcm = greycomatrix(img_gray, [5], [0], 256, symmetric=True, normed=True)
        dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
        correlation = greycoprops(glcm, 'correlation')[0, 0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
        energy = greycoprops(glcm, 'energy')[0, 0]
        feature = np.array([dissimilarity, correlation, homogeneity, energy])
        featureList[i-1] = feature

    # Apply model to data
    predictions = model.predict(featureList)

    # Compute residuals
    for i in range(len(predictions)):
        e = predictions[i] - COUNTS[i]
        residuals.append(e)

    # Plot residual graph
    plt.figure(1)
    plt.scatter(predictions, residuals,  color='blue')
    plt.axhline(0, color='black')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')

    # Plot accuracy graph (ie predicted vs actual)
    plt.figure(2)
    plt.scatter(predictions, COUNTS,  color='blue')
    plt.plot(range(-500, 2500, 250), range(-500, 2500, 250), color='black', linestyle='dotted')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
