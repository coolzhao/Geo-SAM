import os
from PyQt5.QtGui import QIcon

cwd = os.path.abspath(os.path.dirname(__file__))
geo_sam_tool_path = os.path.join(cwd, 'geo_sam_tool.svg')
encoder_tool_path = os.path.join(cwd, 'encoder_tool.svg')

QIcon_GeoSAMTool = QIcon(geo_sam_tool_path)
QIcon_GeoSAMEncoder = QIcon(encoder_tool_path)
