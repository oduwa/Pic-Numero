import Display
import Helper
from skimage.color import rgb2gray
import numpy as np
from scipy import misc
from sklearn import svm, grid_search
from skimage import img_as_ubyte, io
from sklearn import decomposition
import matplotlib.pyplot as plt
import string
import random
import os, sys


# Load train data
train_filenames = []
for filename in os.listdir("train/positive"):
    if(filename != ".DS_Store"): train_filenames.append("train/positive/" + filename)
train_targets = [1]*(len(os.listdir("train/positive"))-1)

for filename in os.listdir("train/negative"):
    if(filename != ".DS_Store"): train_filenames.append("train/negative/" + filename)
train_targets = train_targets + [0]*(len(os.listdir("train/negative"))-1)

n_train_samples = len(train_filenames)
sample_size = 20*20
train_data = np.zeros((n_train_samples, sample_size))
i = 0
for filename in train_filenames:
    img = io.imread(filename)
    train_data[i] = img.flatten()
    i = i + 1;


# Load test data
test_filenames = []
for filename in os.listdir("test"):
    if(filename != ".DS_Store"): test_filenames.append("test/" + filename)

n_test_samples = len(test_filenames)
test_data = np.zeros((n_test_samples, sample_size))
i = 0
for filename in test_filenames:
    img = io.imread(filename)
    test_data[i] = img.flatten()
    i = i + 1;


# Visualise
n_positives = len(os.listdir("train/positive"))-1
train_data_reduced = decomposition.PCA(n_components=2).fit_transform(train_data)
positives = decomposition.PCA(n_components=2).fit_transform(train_data[:n_positives])
negatives = decomposition.PCA(n_components=2).fit_transform(train_data[n_positives:])
# fig, ax1 = plt.subplots()
# ax1.scatter(positives[:, 0], positives[:, 1], color='b')
# ax2 = ax1.twinx()
# ax2.scatter(negatives[:, 0], negatives[:, 1], color='r')
# plt.show()

# create a mesh to plot in
h = 1000  # step size in the mesh
x_min, x_max = positives[:, 0].min() - 1, train_data_reduced[:, 0].max() + 1
y_min, y_max = train_data_reduced[:, 1].min() - 1, train_data_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

fig, ax1 = plt.subplots()
classifier = svm.SVC(kernel='rbf', gamma=0.7, C=0.5)
#classifier = svm.SVC(kernel='rbf', gamma=0.7, C=C)
classifier.fit(train_data_reduced, train_targets)
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
print(Z)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot also the training points
X = train_data_reduced
y = train_targets
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
