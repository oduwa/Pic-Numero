from PyQt4 import QtCore, QtGui

selectedClusters = []

def getSelectedClusters():
    return selectedClusters

def didPressRedCheckbox(state):
    if(state == QtCore.Qt.Checked):
        selectedClusters.append(0)
    else:
        selectedClusters.remove(0)

def didPressGreenCheckbox(state):
    if(state == QtCore.Qt.Checked):
        selectedClusters.append(1)
    else:
        selectedClusters.remove(1)

def didPressBlueCheckbox(state):
    if(state == QtCore.Qt.Checked):
        selectedClusters.append(2)
    else:
        selectedClusters.remove(2)

def didPressCyanCheckbox(state):
    if(state == QtCore.Qt.Checked):
        selectedClusters.append(3)
    else:
        selectedClusters.remove(3)

def didPressMagentaCheckbox(state):
    if(state == QtCore.Qt.Checked):
        selectedClusters.append(4)
    else:
        selectedClusters.remove(4)

def didPressYellowCheckbox(state):
    if(state == QtCore.Qt.Checked):
        selectedClusters.append(5)
    else:
        selectedClusters.remove(5)
