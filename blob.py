import matplotlib.pyplot as plt
from scipy import misc
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray

# Blob detection as a potential feature representation because of how it might be
# able to represent the dense areas of the wheat images.

image = misc.imread("Assets/001_ROI.png");
image_gray = rgb2gray(image);

blobs = blob_doh(image_gray, min_sigma=1, max_sigma=100, threshold=.01)
print(blobs.shape)

fig, ax = plt.subplots()
ax.imshow(image, interpolation='nearest')
for blob in blobs:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    ax.add_patch(c)

plt.show()
#fig.savefig('foo.png')



main();
