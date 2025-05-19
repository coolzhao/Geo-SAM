from qgis.core import QgsApplication
from qgis.gui import QgisInterface
from qgis.PyQt.QtCore import pyqtSignal, QObject
from qgis.PyQt.QtWidgets import (
    QAction,
    QToolBar,
)
import processing

from .tools.widgetTool import Selector, EncoderCopilot
from .ui.icons import QIcon_GeoSAMTool, QIcon_EncoderTool, QIcon_EncoderCopilot
from .geo_sam_provider import GeoSamProvider


class Geo_SAM(QObject):
    execute_SAM = pyqtSignal()

    def __init__(self, iface: QgisInterface, cwd: str):
        super().__init__()
        self.iface = iface
        self.cwd = cwd
        self.canvas = iface.mapCanvas()
        self.demo_img_name = "beiluhe_google_img_201211_clip"
        feature_dir = cwd + "/features/" + self.demo_img_name
        self.feature_dir = feature_dir

    def initProcessing(self):
        self.provider = GeoSamProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

        self.toolbar: QToolBar = self.iface.addToolBar('Geo SAM Toolbar')
        self.toolbar.setObjectName('mGeoSamToolbar')
        self.toolbar.setToolTip('Geo SAM Toolbar')

        self.actionSamTool = QAction(
            QIcon_GeoSAMTool,
            "Geo-SAM Segmentation",
            self.iface.mainWindow()
        )

        self.actionSamEncoder = QAction(
            QIcon_EncoderTool,
            "Geo-SAM Image Encoder",
            self.iface.mainWindow()
        )

        self.actionSamEncoderCopilot = QAction(
            QIcon_EncoderCopilot,
            "Geo-SAM Encoder Copilot",
            self.iface.mainWindow()
        )

        self.actionSamTool.setObjectName("mActionGeoSamTool")
        self.actionSamTool.setToolTip(
            "Geo-SAM Segmentation: Use it to label landforms")
        self.actionSamTool.triggered.connect(self.create_widget_selector)

        self.actionSamEncoder.setObjectName("mActionGeoSamEncoder")
        self.actionSamEncoder.setToolTip(
            "Geo-SAM Image Encoder: Use it to encode/preprocess image before labeling")
        self.actionSamEncoder.triggered.connect(self.encodeImage)

        self.actionSamEncoderCopilot.setObjectName(
            "mActionGeoSamEncoderCopilot")
        self.actionSamEncoderCopilot.setToolTip(
            "Encoder Copilot: Assist you in optimizing your Encoder Settings")
        self.actionSamEncoderCopilot.triggered.connect(
            self.create_widget_encoder_copilot)

        self.iface.addPluginToMenu('Geo-SAM Tools', self.actionSamTool)
        self.iface.addPluginToMenu('Geo-SAM Tools', self.actionSamEncoder)
        self.iface.addPluginToMenu(
            'Geo-SAM Tools', self.actionSamEncoderCopilot)

        # self.iface.addToolBarIcon(self.action)
        self.toolbar.addAction(self.actionSamTool)
        self.toolbar.addAction(self.actionSamEncoder)
        self.toolbar.addAction(self.actionSamEncoderCopilot)
        self.toolbar.setVisible(True)

    def create_widget_selector(self):
        '''Create widget for selecting landform by prompts'''
        if not hasattr(self, "wdg_select"):
            self.wdg_select = Selector(self, self.iface, self.cwd)
        self.wdg_select.open_widget()

    def create_widget_encoder_copilot(self):
        '''Create widget for co-piloting encoder settings'''
        if not hasattr(self, "wdg_copilot"):
            self.wdg_copilot = EncoderCopilot(self, self.iface, self.cwd)
        self.wdg_copilot.open_widget()

    def unload(self):
        '''Unload actions when plugin is closed'''
        if hasattr(self, "wdg_select"):
            self.wdg_select.unload()
            self.wdg_select.setParent(None)
        if hasattr(self, "wdg_copilot"):
            self.wdg_copilot.unload()
            self.wdg_copilot.setParent(None)

        # self.wdg_select.setVisible(False)
        self.iface.removeToolBarIcon(self.actionSamTool)
        self.iface.removeToolBarIcon(self.actionSamEncoder)
        self.iface.removeToolBarIcon(self.actionSamEncoderCopilot)
        self.iface.removePluginMenu('&Geo-SAM Tools', self.actionSamTool)
        self.iface.removePluginMenu('&Geo-SAM Tools', self.actionSamEncoder)
        self.iface.removePluginMenu(
            '&Geo-SAM Tools', self.actionSamEncoderCopilot)

        del self.actionSamTool
        del self.actionSamEncoder
        del self.actionSamEncoderCopilot
        del self.toolbar
        QgsApplication.processingRegistry().removeProvider(self.provider)

    def encodeImage(self):
        '''Convert layer containing a point x & y coordinate to a new point layer'''
        processing.execAlgorithmDialog('geo_sam:geo_sam_encoder', {})
