import os
from qgis.PyQt import uic
from qgis.gui import QgsDockWidget
# try:
#     from qgis.PyQt.QtGui import QDockWidget, QWidget
# except:
#     from qgis.PyQt.QtWidgets import QDockWidget, QWidget

cwd = os.path.abspath(os.path.dirname(__file__))
selector_path = os.path.join(cwd, "Selector.ui")
encoder_path = os.path.join(cwd, "EncoderCopilot.ui")

UI_Selector: QgsDockWidget = uic.loadUi(selector_path)
UI_EncoderCopilot: QgsDockWidget = uic.loadUi(encoder_path)
