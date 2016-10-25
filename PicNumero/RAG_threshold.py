from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from scipy import misc
from skimage.color import rgb2gray
import numpy as np
import Helper
import Display


def main():
    img = misc.imread("wheat.png")

    # labels1 = segmentation.slic(img, compactness=100, n_segments=9)
    labels1 = segmentation.slic(img, compactness=50, n_segments=4)
    out1 = color.label2rgb(labels1, img, kind='overlay')
    print(labels1.shape)

    g = graph.rag_mean_color(img, labels1)
    labels2 = graph.cut_threshold(labels1, g, 29)
    out2 = color.label2rgb(labels2, img, kind='overlay')

    # get roi
    # logicalIndex = (labels2 != 1)
    # gray = rgb2gray(img);
    # gray[logicalIndex] = 0;


    plt.figure()
    io.imshow(out1)
    plt.figure()
    io.imshow(out2)
    io.show()


def experiment_with_parameters():
    img = misc.imread("wheat.png")

    compactness_values = [30, 50, 70, 100, 200, 300, 500, 700, 1000]
    n_segments_values = [3,4,5,6,7,8,9,10]

    for compactness_val in compactness_values:
        for n in n_segments_values:
            labels1 = segmentation.slic(img, compactness=compactness_val, n_segments=n)
            out1 = color.label2rgb(labels1, img, kind='overlay')

            fig, ax = plt.subplots()
            ax.imshow(out1, interpolation='nearest')
            ax.set_title("Compactness: {} | Segments: {}".format(compactness_val, n))
            plt.savefig("RAG/c{}_k{}.png".format(compactness_val, n))
            plt.close(fig)

def spectral_cluster(filename, compactness_val=30, n=6):
    img = misc.imread(filename)
    labels1 = segmentation.slic(img, compactness=compactness_val, n_segments=n)
    out1 = color.label2rgb(labels1, img, kind='overlay', colors=['red','green','blue','cyan','magenta','yellow'])

    fig, ax = plt.subplots()
    ax.imshow(out1, interpolation='nearest')
    ax.set_title("Compactness: {} | Segments: {}".format(compactness_val, n))
    plt.show()

def extract_roi(img, labels_to_keep=[1,2]):
    label_img = segmentation.slic(img, compactness=30, n_segments=6)
    labels = np.unique(label_img);print(labels)
    gray = rgb2gray(img);

    for label in labels:
        if(label not in labels_to_keep):
            logicalIndex = (label_img == label)
            gray[logicalIndex] = 0;

    Display.show_image(gray)
    io.imsave("grayy.png", gray)

def block_process(img):
    func = lambda block: io.imsave("Block/{}.png".format(Helper.generate_random_id()), block)#Display.save_image("Block/{}.png".format(Helper.generate_random_id()), block)
    Helper.block_proc(img, (50,1900), func)


img = misc.imread("wheat.png")
#experiment_with_parameters()
#spectral_cluster("wheat.png")
#extract_roi(img)
block_process(misc.imread("grayy.png"))

# img = misc.imread("wheat.png")
# labels1 = segmentation.slic(img, compactness=100, n_segments=4)
# out1 = color.label2rgb(labels1, img, kind='overlay')
#
# fig, ax = plt.subplots()
# ax.imshow(out1, interpolation='nearest')
# ax.set_title("Compactness: {} | Segments: {}".format(100, 4))
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# plt.savefig("RAG/c{}_k{}.png".format(100, 4))
