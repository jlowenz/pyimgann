import sys
import logging
import pyimgann.ui as ui
import pyimgann.controller as ctrl
from PyQt4 import QtGui

logging.basicConfig()

def run():
    app = QtGui.QApplication(sys.argv)
    mw = ui.MainWindow()
    corrs = ctrl.CorrespondenceController(mw)
    mw.show()
    sys.exit(app.exec_())
