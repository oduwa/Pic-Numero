from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from scipy import misc
from skimage.color import rgb2gray
import numpy as np
import Helper
import Display


def spectral_cluster(filename, compactness_val=30, n=6):
    '''
    Apply spectral clustering to a given image using k-means clustering and
    display results.

    Args:
        filename: name of the image to segment.

        compactness_val: Controls the "boxyness" of each segment. Higher values
          mean a more boxed shape.

        n = number of clusters.
     '''
    img = misc.imread(filename)
    labels1 = segmentation.slic(img, compactness=compactness_val, n_segments=n)
    out1 = color.label2rgb(labels1, img, kind='overlay', colors=['red','green','blue','cyan','magenta','yellow'])

    fig, ax = plt.subplots()
    ax.imshow(out1, interpolation='nearest')
    ax.set_title("Compactness: {} | Segments: {}".format(compactness_val, n))
    plt.show()

def experiment_with_parameters():
    '''
    Apply spectral clustering to test wheat image using k-means clustering with
    different values of k and different compactness values.

    Saves the results to the Clusters folder for inspection.
    '''
    img = misc.imread("../Assets/wheat.png")

    compactness_values = [30, 50, 70, 100, 200, 300, 500, 700, 1000]
    n_segments_values = [3,4,5,6,7,8,9,10]

    for compactness_val in compactness_values:
        for n in n_segments_values:
            labels1 = segmentation.slic(img, compactness=compactness_val, n_segments=n)
            out1 = color.label2rgb(labels1, img, kind='overlay')

            fig, ax = plt.subplots()
            ax.imshow(out1, interpolation='nearest')
            ax.set_title("Compactness: {} | Segments: {}".format(compactness_val, n))
            plt.savefig("../Clusters/c{}_k{}.png".format(compactness_val, n))
            plt.close(fig)

def extract_roi(img, labels_to_keep=[1,2]):
    '''
    Given a wheat image, this method returns an image containing only the region
    of interest.

    Args:
        img: input image.

        labels_to_keep: cluster labels to be kept in image while pixels
            belonging to clusters besides these ones are removed.

    Return:
        roi_img: Input image containing only the region
        of interest.
    '''
    label_img = segmentation.slic(img, compactness=30, n_segments=6)
    labels = np.unique(label_img);print(labels)
    gray = rgb2gray(img);

    for label in labels:
        if(label not in labels_to_keep):
            logicalIndex = (label_img == label)
            gray[logicalIndex] = 0;

    Display.show_image(gray)
    return gray
