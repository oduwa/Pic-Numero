# Wheat Grain Counting

Makes use of Python 2.7 and the scikit machine learning and image processing libraries (which can be found [here](http://scikit-learn.org/stable/) and [here](http://scikit-image.org)).

Also includes a helper standalone program for ROI extraction however it is only available for Mac OS X and certain flavours of Linux at this time. Can be found in dist/gui/. Run from terminal with the command "./dist/gui/gui". Source is contained in "gui.py" and "gui\_checkbox\_handlers.py".

See **main.py_** as the point of entry into the application source. Here, you can call `run\_with\_glcm()` to apply the counting-by-regression approach or `run\_with\_svm()`, `run\_with\_mlp()` or `run\_with\_cnn()` to apply the ounting-by-detection approach. If the count is computed using one of the counting-by-detection approaches, the classifications of the sub-images will be saved to the "Results" folder after it is calculated.

**Note: _TensorFlow is required to use the CNN functionality (CNN.py). If you do
not have TensorFlow already installed, visit [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)
for detailed instructions on how to install it._**
