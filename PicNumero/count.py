import os, sys
import tqdm

from scipy import misc
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

# Way to import from matplotlib without warning according to
# https://github.com/matplotlib/matplotlib/issues/5836#issuecomment-223997114
import warnings;
with warnings.catch_warnings():
    warnings.simplefilter("ignore");
    import matplotlib.pyplot as plt


def main():
    numberOfImages = 11;

    # TODO: AUTOMATICALLY GET NUMBER OF IMAGES
    # Get number of images. Remeber to divide by 2 as for every relevant image,
    # theres also the comparison image.
    # if ".DS_Store" in os.listdir("Wheat_ROIs"):
    #     numberOfImages = (len(os.listdir("Wheat_ROIs")) - 1)/2;
    # else:
    #     numberOfImages = len(os.listdir("Wheat_ROIs"))/2;

    # For each ROI image in folder
    for i in tqdm.tqdm(range(1, numberOfImages+1)):
        # Load image
        filename = "../Wheat_ROIs/{:03d}_ROI.png".format(i);
        img = misc.imread(filename);
        img_gray = rgb2gray(img);

        # Detect blobs. See http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_doh
        # for function documentation
        blobs = blob_doh(img_gray, min_sigma=1, max_sigma=100, threshold=.01)

        # Display blobs on image and save image
        fig, ax = plt.subplots()
        plt.title("Number of Blobs Detected: {}".format(blobs.shape[0]))
        plt.grid(False)
        ax.imshow(img, interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            ax.add_patch(c)
        fig.savefig("../Wheat_ROIs/{:03d}_Blob.png".format(i))





main();
