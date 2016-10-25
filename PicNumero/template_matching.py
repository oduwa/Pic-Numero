from collections import namedtuple
import numpy as np
from scipy import misc
from skimage.color import rgb2gray
from skimage import data
from skimage.feature import match_template

# Way to import from matplotlib without warning according to
# https://github.com/matplotlib/matplotlib/issues/5836#issuecomment-223997114
import warnings;
with warnings.catch_warnings():
    warnings.simplefilter("ignore");
    import matplotlib.pyplot as plt


def get_n_max_indices(X, n):
    '''Returns a 1d numpy array containing indices of n biggest values in x.'''
    indices = np.zeros(n);
    for i in range(0,n):
        indices[i] = np.argmax(X);
        X.flat[int(indices[i])] = -1;
    return indices


def match_templates_1(search_image, template_image, n=0):
    '''
    Calculates the n closest matches of some template image in another image and
    displays a figure illustrating the results.

    Args:
        search_image: image within which to match template.
        template_image: image to be matched.
        n: number of matches to be found. ie. closest n matches.
    '''
    Point = namedtuple('Point', ['x', 'y'])

    # Calculate template matches
    match_result = match_template(search_image, template_image);

    # Get closest n matches
    print(match_result.shape)
    if(n == 0):
        n = int(match_result.shape[1]);
    matched_point_list = []
    max_indices = get_n_max_indices(match_result, n)
    for index in max_indices:
        ij = np.unravel_index(int(index), match_result.shape)
        x, y = ij[::-1]
        point = Point(x,y)
        #print(point)
        matched_point_list.append(point)

    # Display
    fig = plt.figure(figsize=(8, 3))
    plt.gray()
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, adjustable='box-forced')
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2, adjustable='box-forced')

    ax1.imshow(template_image)
    ax1.set_axis_off()
    ax1.set_title('grain template')

    # highlight matched regions
    ax2.imshow(search_image)
    ax2.set_axis_off()
    ax2.set_title('image')
    himage, wimage = template_image.shape
    for point in matched_point_list:
        rect = plt.Rectangle((point.x, point.y), wimage, himage, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)

    # highlight matched regions
    ax3.imshow(match_result)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    ax3.autoscale(False)
    for point in matched_point_list:
        ax3.plot(point.x, point.y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    plt.show()

def match_templates_2(search_image, template_image, n=1):
    '''
    Calculates the n closest matches of some template image in another image and
    displays a figure illustrating the results. This is a variation of the
    match_templates_1() method which takes a different approach to matching by
    iteratively matching, removing found match from search image and then running
    match again.

    Args:
        search_image: image within which to match template.
        template_image: image to be matched.
        n: number of matches to be found. ie. closest n matches.
    '''
    Point = namedtuple('Point', ['x', 'y'])
    matched_point_list = []

    # Calculate template matches
    i = 0
    while (i < n):
        print(n)
        match_result = match_template(search_image, template_image);

        # Get closest match and store position
        ij = np.unravel_index(np.argmax(match_result), match_result.shape)
        x, y = ij[::-1]
        point = Point(x,y)
        matched_point_list.append(point)
        hTemplate, wTemplate = template_image.shape

        # Set matched patch to black
        for i in range(0, hTemplate):
            for j in range(0, wTemplate):
                search_image[i + point.y][j + point.x] = 0
        i = i + 1

    # Display
    fig = plt.figure(figsize=(8, 3))
    plt.gray()
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, adjustable='box-forced')
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2, adjustable='box-forced')

    ax1.imshow(template_image)
    ax1.set_axis_off()
    ax1.set_title('template')

    # highlight matched regions
    ax2.imshow(search_image)
    ax2.set_axis_off()
    ax2.set_title('image')
    himage, wimage = template_image.shape
    for point in matched_point_list:
        rect = plt.Rectangle((point.x, point.y), wimage, himage, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)

    # highlight matched regions
    ax3.imshow(match_result)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    ax3.autoscale(False)
    for point in matched_point_list:
        ax3.plot(point.x, point.y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    plt.show()



search_image = rgb2gray(misc.imread("../Assets/bush.png"))
template_image = rgb2gray(misc.imread("../Assets/grain.png"))
match_templates_1(search_image, template_image, 1);
