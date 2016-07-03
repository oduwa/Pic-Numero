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
COUNTS = [65*22, 80*22, 87*22, 40*22, 71*22, 58*22, 77*22] # started with 1 and 2

def main():
    image = rgb2gray(misc.imread("x.jpg"));
    image = data.camera()
    image = img_as_ubyte(rgb2gray(misc.imread("wheat.png")));
    print("XXX: {}".format(image.flat[np.argmax(image)]))
    #Display.show_image(image, isGray=True)

    # select some patches from different areas of the image
    spikelet_locations = [(643, 517), (877, 574), (2129, 649), (1342, 454)]
    #spikelet_locations = [(643, 517), (877, 574), (1555, 649), (1342, 454)]
    #spikelet_locations = [(474, 291), (440, 433), (466, 18), (462, 236)]
    spikelet_patches = []
    #sky_locations = [(53, 179), (590, 93), (2333, 55), (1447, 62)]
    sky_locations = [(534, 1056), (1017, 857), (1711, 1365), (2199, 1093)] # actually stalks
    #sky_locations = [(53, 179), (590, 93), (1333, 55), (1447, 62)]
    #sky_locations = [(54, 48), (21, 233), (90, 380), (195, 330)]
    sky_patches = []

    for loc in spikelet_locations:
        #print("X: {}".format(image[loc[1]:loc[1] + 2, loc[0]:loc[0] + 2]))
        spikelet_patches.append(image[loc[1]:loc[1] + PATCH_SIZE, loc[0]:loc[0] + PATCH_SIZE])
        #spikelet_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

    for loc in sky_locations:
        sky_patches.append(image[loc[1]:loc[1] + PATCH_SIZE, loc[0]:loc[0] + PATCH_SIZE])
        #sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])


    # compute some GLCM properties each patch
    xs = []
    ys = []
    for patch in (spikelet_patches + sky_patches):
        glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
        xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(greycoprops(glcm, 'correlation')[0, 0])
        print("({}, {})".format(greycoprops(glcm, 'dissimilarity')[0, 0], greycoprops(glcm, 'correlation')[0, 0]))

    # create the figure
    fig = plt.figure(figsize=(8, 8))

    # display original image with locations of patches
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
              )
    for (x,y) in spikelet_locations:
        ax.plot(x, y, 'gs', markersize=PATCH_SIZE/4)
    for (x,y) in sky_locations:
        ax.plot(x, y, 'bs', markersize=PATCH_SIZE/4)

    ax.set_xlabel('Original Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

    # for each patch, plot (dissimilarity, correlation)
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(xs[:len(spikelet_patches)], ys[:len(spikelet_patches)], 'go',
            label='Grass')
    ax.plot(xs[len(sky_patches):], ys[len(sky_patches):], 'bo',
            label='Stalk')
    ax.set_xlabel('GLCM Dissimilarity')
    ax.set_ylabel('GLVM Correlation')
    ax.legend()

    # display the image patches
    for i, patch in enumerate(spikelet_patches):
        ax = fig.add_subplot(3, len(spikelet_patches), len(spikelet_patches)*1 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
                  )
        ax.set_xlabel('Grass %d' % (i + 1))

    for i, patch in enumerate(sky_patches):
        ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
                  )
        ax.set_xlabel('Sky %d' % (i + 1))

    plt.show()

def regression():
    regression_model = linear_model.LinearRegression()
    #regression_model.fit([[1], [-4], [3]], [6, -4, 10])
    regression_model.fit([-1,4,3], [6, -4, 10])

    print("y = {}x + {}".format(regression_model.coef_, regression_model.intercept_))

def train():
    numberOfImages = 2;

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
    regression_model.fit(featureList, COUNTS)
    print("COEFF: {}\nINTERCEPT: {}".format(regression_model.coef_, regression_model.intercept_))


main();
#train();
