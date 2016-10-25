import sys, platform
import gui_checkbox_handlers
import spectral_roi
import Helper
import Display
import overlay
from PIL import Image
from PIL.ImageQt import ImageQt
from skimage import img_as_ubyte, io
from PyQt4 import QtCore, QtGui
from Foundation import NSURL

CLUSTER_IMAGE_FILENAME = "./spectral_cluster.png"
CLUSTER_SETTING_OPTIONS = ["2", "4", "6", "9", "11", "13", "15", "17"]
COMPACTNESS_SETTING_OPTIONS = ["5", "10", "15", "20", "25", "30", "35", "40", "50", "75", "100", "150", "200"]
SELECTED_CLUSTER_SETTING_INDEX = 2
SELECTED_COMPACTNESS_SETTING_INDEX = 5

def updateLabelToClusterShowImage(displayLabel, filename, width=240, height=240):
    # Compute cluster memberships
    compactness = int(COMPACTNESS_SETTING_OPTIONS[SELECTED_COMPACTNESS_SETTING_INDEX])
    n = int(CLUSTER_SETTING_OPTIONS[SELECTED_CLUSTER_SETTING_INDEX])
    spectral_roi.spectral_cluster(filename, CLUSTER_IMAGE_FILENAME, compactness, n)

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

def didPressSettingsMenuItem():
    print("BOY HE BOUT TO DO IT")
    d = SettingsDialog()
    d.exec_()

class DraggableTextField(QtGui.QLineEdit):

    def __init__(self, parent):
      super(DraggableTextField, self).__init__(parent)
      self.setAcceptDrops(True)
      self.setReadOnly(True)
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
           self.setText(pixPath)
           self.parent.setImageFilePath(pixPath)

           # Display image from url
           updateLabelToClusterShowImage(self.parent.getImageLabel(), pixPath, self.parent.frameSize().width()*0.5, self.parent.frameSize().height()*0.5)

class AppWindow(QtGui.QWidget):

    def __init__(self):
      super(AppWindow, self).__init__()
      self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)

      self.imageLabel = None
      self.imageFilePath = None
      self.RoiFilePath = None
      self.uploadTextView = None
      self.layout = QtGui.QVBoxLayout()

      self.create_toolbar()

      self.input_section  = QtGui.QFormLayout()
      self.layout.addLayout(self.input_section)
      self.display_section = QtGui.QFormLayout()
      self.layout.addLayout(self.display_section)
      self.extraction_section = QtGui.QFormLayout()
      self.layout.addLayout(self.extraction_section)
      self.final_section = QtGui.QFormLayout()
      self.layout.addLayout(self.final_section)

      self.initUI()

    def create_toolbar(self):
        settingsAction = QtGui.QAction("&Set Cluster Parameters", self)
        settingsAction.setShortcut("Ctrl+Q")
        settingsAction.triggered.connect(didPressSettingsMenuItem)

        settingsMenu = QtGui.QMenu(self)
        settingsMenu.addAction(settingsAction)

        toolBar = QtGui.QToolBar()
        settingsButton = QtGui.QToolButton()
        settingsButton.setPopupMode(QtGui.QToolButton.MenuButtonPopup)
        settingsButton.setText("Settings")
        settingsButton.setMenu(settingsMenu)
        toolBar.addWidget(settingsButton)
        self.layout.addWidget(toolBar)

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
        self.uploadTextView.setText(fname)
        updateLabelToClusterShowImage(self.imageLabel, fname, self.frameSize().width()*0.5, self.frameSize().height()*0.5)

    def didClickSubmitButton(self, event):
        print(self.imageFilePath)
        img = img_as_ubyte(io.imread(CLUSTER_IMAGE_FILENAME))
        roi_img = spectral_roi.extract_roi(img, gui_checkbox_handlers.getSelectedClusters())
        roi_img_filename = "{}.png".format(Helper.generate_random_id())
        io.imsave(roi_img_filename, roi_img)
        Display.show_image(roi_img, roi_img_filename)


    def initUI(self):
      # Create text view and button and add to layout
      uploadTextView = DraggableTextField(self)
      uploadTextView.setDragEnabled(True)
      self.uploadTextView = uploadTextView
      uploadButton = QtGui.QPushButton('Select Image', self)
      uploadButton.clicked.connect(self.didClickFileSelectButton)
      self.input_section.addRow(uploadTextView, uploadButton)


      # Create image display label and add to layout
      self.imageLabel = QtGui.QLabel(self)
      updateLabelToShowImage(self.getImageLabel(), "../Assets/placeholder.png", self.frameSize().width()*0.5, self.frameSize().height()*0.5)
      self.display_section.addRow(self.imageLabel)

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

      self.extraction_section.addRow(redCheckBox, greenCheckBox)
      self.extraction_section.addRow(blueCheckBox, cyanCheckBox)
      self.extraction_section.addRow(magentaCheckBox, yellowCheckBox)

      # Create submit button and add to layout
      submitButton = QtGui.QPushButton('Select Clusters as ROI', self)
      submitButton.clicked.connect(self.didClickSubmitButton)
      self.final_section.addRow(submitButton)


      self.setLayout(self.layout)
      #self.layout = layout
      self.setWindowTitle('Spectral ROI Extractor')

class SettingsDialog(QtGui.QDialog):
    def __init__(self):
      super(SettingsDialog, self).__init__()
      self.setModal(True)
      self.initUI()

    def selectionInClusterComboBox(self, i):
        global SELECTED_CLUSTER_SETTING_INDEX
        SELECTED_CLUSTER_SETTING_INDEX = i

    def selectionInCompactnessComboBox(self, i):
        global SELECTED_COMPACTNESS_SETTING_INDEX
        SELECTED_COMPACTNESS_SETTING_INDEX = i


    def initUI(self):
        # Create layout
      layout = QtGui.QFormLayout()
      label1 = QtGui.QLabel("Clusters")
      combo1 = QtGui.QComboBox(self)
      combo1.addItems(CLUSTER_SETTING_OPTIONS)
      combo1.currentIndexChanged.connect(self.selectionInClusterComboBox)
      combo1.setCurrentIndex(SELECTED_CLUSTER_SETTING_INDEX)
      layout.addRow(label1, combo1)

      label2 = QtGui.QLabel("Compactness")
      combo2 = QtGui.QComboBox(self)
      combo2.addItems(COMPACTNESS_SETTING_OPTIONS)
      combo2.currentIndexChanged.connect(self.selectionInCompactnessComboBox)
      combo2.setCurrentIndex(SELECTED_COMPACTNESS_SETTING_INDEX)
      layout.addRow(label2, combo2)

      self.setLayout(layout)

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
