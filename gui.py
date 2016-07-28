import sys, platform
import gui_checkbox_handlers
import spectral_roi
import Helper
import Display
from PIL import Image
from PIL.ImageQt import ImageQt
from skimage import img_as_ubyte, io
from PyQt4 import QtCore, QtGui
from Foundation import NSURL

CLUSTER_IMAGE_FILENAME = "spectral_cluster.png"

def updateLabelToClusterShowImage(displayLabel, filename, width=240, height=240):
    # Compute cluster memberships
    spectral_roi.spectral_cluster(filename, CLUSTER_IMAGE_FILENAME)

    # Display result
    picture = Image.open(CLUSTER_IMAGE_FILENAME)
    picture.thumbnail((width,height), Image.ANTIALIAS)
    pixmap = QtGui.QPixmap.fromImage(ImageQt(picture))
    displayLabel.setPixmap(pixmap)
    displayLabel.setFixedSize(width, height)

def updateLabelToShowImage(displayLabel, filename, width=240, height=240):
    picture = Image.open(filename)
    picture.thumbnail((width,height), Image.ANTIALIAS)
    pixmap = QtGui.QPixmap.fromImage(ImageQt(picture))
    displayLabel.setPixmap(pixmap)
    displayLabel.setFixedSize(width, height)

class DraggableTextField(QtGui.QLineEdit):

    def __init__(self, parent):
      super(DraggableTextField, self).__init__(parent)
      self.setAcceptDrops(True)
      self.parent = parent

    def dragEnterEvent(self, e):
      print(e.mimeData().hasUrls())

      if e.mimeData().hasUrls():
         e.accept()
      else:
         e.ignore()

    def dropEvent(self, e):
       if(e.mimeData().hasUrls()):
           # Get image url
           pixUrl = e.mimeData().urls()[0]
           if(platform.system() == "Darwin"):
               pixPath = str(NSURL.URLWithString_(str(pixUrl.toString())).filePathURL().path())
           else:
               pixPath = str(pixUrl.toLocalFile())
           print("FILEPATH {}".format(pixPath))
           self.parent.setImageFilePath(pixPath)

           # Display image from url
           updateLabelToClusterShowImage(self.parent.getImageLabel(), pixPath, self.parent.frameSize().width()*0.5, self.parent.frameSize().height()*0.5)

class AppWindow(QtGui.QWidget):

    def __init__(self):
      super(AppWindow, self).__init__()
      self.imageLabel = None
      self.imageFilePath = None
      self.RoiFilePath = None
      self.initUI()

    def getLayout(self):
        return self.layout

    def getImageLabel(self):
        return self.imageLabel

    def setImageFilePath(self, path):
        self.imageFilePath = path

    def getImageFilePath(self):
        return self.imageFilePath

    def didClickFileSelectButton(self, event):
        fname = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file',''))
        self.setImageFilePath(fname)
        updateLabelToClusterShowImage(self.imageLabel, fname, self.frameSize().width()*0.5, self.frameSize().height()*0.5)

    def didClickSubmitButton(self, event):
        print(self.imageFilePath)
        img = img_as_ubyte(io.imread(CLUSTER_IMAGE_FILENAME))
        roi_img = spectral_roi.extract_roi(img, gui_checkbox_handlers.getSelectedClusters())
        roi_img_filename = "{}.png".format(Helper.generate_random_id())
        io.imsave(roi_img_filename, roi_img)
        Display.show_image(roi_img, roi_img_filename)


    def initUI(self):
      # Create layout
      layout = QtGui.QFormLayout()

      # Add title text
      layout.addRow(QtGui.QLabel("Type some text in textbox and drag it into combo box"))

      # Add upload interface
      uploadTextView = DraggableTextField(self)
      uploadTextView.setDragEnabled(True)
      uploadButton = QtGui.QPushButton('Select Image', self)
      uploadButton.clicked.connect(self.didClickFileSelectButton)
      layout.addRow(uploadTextView, uploadButton)

      # Create image display label and add to layout
      self.imageLabel = QtGui.QLabel(self)
      layout.addRow(self.imageLabel)

      # Create checkboxes and add them to layout
      redCheckBox = QtGui.QCheckBox('Red', self)
      greenCheckBox = QtGui.QCheckBox('Green', self)
      blueCheckBox = QtGui.QCheckBox('Blue', self)
      cyanCheckBox = QtGui.QCheckBox('Cyan', self)
      magentaCheckBox = QtGui.QCheckBox('Magenta', self)
      yellowCheckBox = QtGui.QCheckBox('Yellow', self)

      redCheckBox.stateChanged.connect(gui_checkbox_handlers.didPressRedCheckbox)
      greenCheckBox.stateChanged.connect(gui_checkbox_handlers.didPressGreenCheckbox)
      blueCheckBox.stateChanged.connect(gui_checkbox_handlers.didPressBlueCheckbox)
      cyanCheckBox.stateChanged.connect(gui_checkbox_handlers.didPressCyanCheckbox)
      magentaCheckBox.stateChanged.connect(gui_checkbox_handlers.didPressMagentaCheckbox)
      yellowCheckBox.stateChanged.connect(gui_checkbox_handlers.didPressYellowCheckbox)

      layout.addRow(redCheckBox, greenCheckBox)
      layout.addRow(blueCheckBox, cyanCheckBox)
      layout.addRow(magentaCheckBox, yellowCheckBox)

      # Create submit button and add to layout
      submitButton = QtGui.QPushButton('Select Clusters as ROI', self)
      submitButton.clicked.connect(self.didClickSubmitButton)
      layout.addRow(submitButton)


      self.setLayout(layout)
      self.layout = layout
      self.setWindowTitle('Spectral ROI Extractor')

def main():
    # Create an PyQT4 application object.
    a = QtGui.QApplication(sys.argv)

    # The QWidget widget is the base class of all user interface objects in PyQt4.
    w = AppWindow()

    # Set window size.
    w.resize(960, 640)

    # Show window
    w.show()

    sys.exit(a.exec_())


main();
