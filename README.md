# Pic-Numero
This is a project exploring the counting of objects in images. While the project makes use of the example of counting wheat grains in images of wheat plants, it can be applied to objects of any kind. A very high-level description of the way the system works is that it trains a classifier to recognise images of the object being counted then attempts to recognise instances of this object in any given image and returns the number of recognised matches. As such, the count returned is really an estimate.

The system can make use of different classifiers depending on your choice (currently SVM, MLP neural net and CNN). It might be useful to know that the results of the classifications of the last count can be found in src/Results. "<filename>_0" means it was classified as false (not an object being counted)  while "<filename>_1" means it was classified as true (object being counted).

## Usage ##
First, to train the system to identify objects you want, all you have to do is populate the "train" folder (which contains "positive" and "negative" subfolders) with your training images appropriately.

Then in your code be sure to add `from PicNumero import PicNumero`.

You can then use the `python PicNumero.run_with_svm()` `python PicNumero.run_with_mlp()` or `python PicNumero.run_with_cnn()` with the filename of the image whose objects are to be counted as the argument.

```python
from PicNumero import PicNumero

imagePath = "Path/To/Your/Image.png"
count = PicNumero.run_with_cnn(imagePath)
```

PicNumero also includes a helper standalone program for ROI extraction however it is only available for Mac OS X and certain flavours of Linux at this time. Can be found in dist/gui/. Run from terminal with the command "./dist/gui/gui". Source is contained in "gui.py" and "gui\_checkbox\_handlers.py".

## Dependencies ##
Makes use of Python 2.7 and the scikit machine learning and image processing libraries (which can be found [here](http://scikit-learn.org/stable/) and [here](http://scikit-image.org)).

**Note: _TensorFlow is required to use the CNN functionality (CNN.py). If you do
not have TensorFlow already installed, visit [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)
for detailed instructions on how to install it._**

## Compatibility ##

Tested and working on OS X and certain flavours of Linux. Should work on Windows too but could potentially need some tweaking; I've never tried to run it on windows before.

## Credits ##

Made use of Google's excellent [TensorFlow](https://www.tensorflow.org) library
