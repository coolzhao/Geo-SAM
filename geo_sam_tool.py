import os
import typing
from qgis.core import QgsProject
from qgis.gui import QgsMapToolPan
from qgis.core import QgsRasterLayer
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QFileDialog,
    QAction,
    QFileDialog,
)
from PyQt5.QtGui import QKeySequence, QIcon, QColor
from PyQt5 import uic

from .tools.geoTool import ImageCRSManager, LayerExtent
from .tools.SAMTool import SAM_Model
from .tools.canvasTool import RectangleMapTool, ClickTool, Canvas_Points, Canvas_Rectangle, SAM_PolygonFeature


class Geo_SAM(QObject):
    execute_SAM = pyqtSignal()
    activate_fg = pyqtSignal()

    def __init__(self, iface, cwd: str):
        super().__init__()
        self.iface = iface
        self.cwd = cwd
        self.canvas = iface.mapCanvas()
        # self.create_toolbar()
        self.demo_img_name = "beiluhe_google_img_201211_clip"
        feature_dir = cwd + "/features/" + self.demo_img_name
        self.feature_dir = feature_dir
        self.toolPan = QgsMapToolPan(self.canvas)

    def initGui(self):
        icon_path = os.path.join(self.cwd, "icons/geo_sam_tool.svg")

        self.action = QAction(
            QIcon(icon_path),
            "GeoSAM Tool",
            self.iface.mainWindow()
        )
        self.action.triggered.connect(self.create_widget_selector)
        # self.toolbar.addAction(self.action)
        self.iface.addPluginToMenu('&Geo SAM', self.action)
        self.iface.addToolBarIcon(self.action)

    def load_demo_img(self):
        layer_list = QgsProject.instance().mapLayersByName(self.demo_img_name)
        print(layer_list)
        if layer_list:
            rlayer = layer_list[0]
        else:
            img_path = os.path.join(
                self.cwd, "rasters", self.demo_img_name+'.tif')
            if os.path.exists(img_path):
                rlayer = QgsRasterLayer(img_path, self.demo_img_name)
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    canvas = self.iface.mapCanvas()
                    canvas.setExtent(rlayer.extent())
                    canvas.refresh()
                else:
                    print("Demo image layer failed to load!")
                # self.iface.addRasterLayer(img_path, self.demo_img_name)
            else:
                print(img_path, 'does not exist')

    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        del self.action

    def _init_feature_related(self):
        self.sam_model = SAM_Model(self.feature_dir, self.cwd)
        self.img_crs_manager = ImageCRSManager(self.sam_model.img_crs)
        self.canvas_points = Canvas_Points(self.canvas, self.img_crs_manager)
        self.canvas_rect = Canvas_Rectangle(self.canvas, self.img_crs_manager)

        self.canvas_points.clear()
        self.canvas_rect._init_rect_layer()

        # self.shortcut_undo = QShortcut(QKeySequence(Qt.ControlModifier + Qt.Key_Z), self.iface.mainWindow())
        # self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self.iface.mainWindow())
        # self.shortcut_undo.setContext(Qt.ApplicationShortcut)
        # self.shortcut_undo.activated.connect(self.execute_segmentation)

        self.execute_SAM.connect(self.execute_segmentation)
        self.activate_fg.connect(self.draw_foreground_point)

        # init tools
        self.tool_click_fg = ClickTool(
            self.canvas,
            self.canvas_points,
            'fgp',
            self.execute_SAM,
        )
        self.tool_click_bg = ClickTool(
            self.canvas,
            self.canvas_points,
            'bgp',
            self.execute_SAM,
        )
        self.tool_click_rect = RectangleMapTool(
            self.canvas_rect, self.execute_SAM, self.img_crs_manager
        )

    def create_widget_selector(self):
        '''Create widget selector'''

        # clear last layers if reloaded
        if hasattr(self, "canvas_points"):
            self.canvas_points.clear()
        if hasattr(self, "canvas_rect"):
            self.canvas_rect.clear()

        self._init_feature_related()
        self.load_demo_img()

        UI_Selector = uic.loadUi(os.path.join(self.cwd, "ui/Selector.ui"))
        # connect signals for buttons
        self.wdg_sel = UI_Selector
        self.wdg_sel.pushButton_fg.clicked.connect(self.draw_foreground_point)
        self.wdg_sel.pushButton_bg.clicked.connect(self.draw_background_point)
        self.wdg_sel.pushButton_rect.clicked.connect(self.draw_rect)
        self.wdg_sel.pushButton_clear.clicked.connect(self.clear_layers)
        self.wdg_sel.pushButton_find_file.clicked.connect(self.find_file)
        self.wdg_sel.pushButton_load_file.clicked.connect(self.load_shp_file)
        self.wdg_sel.pushButton_save.clicked.connect(self.save_shp_file)
        self.wdg_sel.pushButton_find_feature.clicked.connect(self.find_feature)
        self.wdg_sel.pushButton_load_feature.clicked.connect(self.load_feature)

        self.wdg_sel.radioButton_enable.setChecked(True)
        self.wdg_sel.radioButton_enable.toggled.connect(self.enable_disable)
        self.wdg_sel.pushButton_fg.setCheckable(True)
        self.wdg_sel.pushButton_bg.setCheckable(True)
        self.wdg_sel.pushButton_rect.setCheckable(True)

        # add widget to QGIS
        self.wdg_sel.setFloating(True)
        self.iface.addDockWidget(Qt.TopDockWidgetArea, self.wdg_sel)

        # start with fg
        self.draw_foreground_point()

    def enable_disable(self):
        radioButton = self.sender()
        if not radioButton.isChecked():
            # UI_Selector.setEnabled(False)
            self.canvas.setMapTool(self.toolPan)
            # findChildren(self, type, name: str = '', options: Union[Qt.FindChildOptions, Qt.FindChildOption] = Qt.FindChildrenRecursively): argument 2 has unexpected type 'FindChildOption'
            # for wdg in UI_Selector.findChildren(QPushButton, 'pushButton_fg'): # PyQt5.QtWidgets.QDockWidget
            # QApplication.restoreOverrideCursor()
            # print(wdg.objectName())
            self.wdg_sel.pushButton_fg.setEnabled(False)
            self.wdg_sel.pushButton_bg.setEnabled(False)
            self.wdg_sel.pushButton_rect.setEnabled(False)
            # self.tool_click_fg.deactivate()
            # self.tool_click_bg.deactivate()
            # self.tool_click_rect.deactivate()
        else:
            # for wdg in UI_Selector.findChildren(QWidget, 'pushButton', Qt.FindChildrenRecursively):
            #     print(wdg.objectName())
            #     wdg.setEnabled(True)
            self.wdg_sel.pushButton_fg.setEnabled(True)
            self.wdg_sel.pushButton_bg.setEnabled(True)
            self.wdg_sel.pushButton_rect.setEnabled(True)
            self.reset_label_tool()

            # UI_Selector.setEnabled(True)

        # self.wdg_sel.radioButton_enable.setEnabled(True)

    def execute_segmentation(self):
        if not hasattr(self, "polygon"):
            self.load_shp_file()
        self.sam_model.sam_predict(
            self.canvas_points, self.canvas_rect, self.polygon)

    # def _set_button_selected(self, button):
    #     buttons_map = {'foreground': self.wdg_sel.pushButton_fg,
    #                    'background': self.wdg_sel.pushButton_bg,
    #                    'rectangle': self.wdg_sel.pushButton_rect}
    #     button_selected = buttons_map.pop(button)
    #     button_selected.setStyleSheet(f"background-color: #c8c8c8")
    #     for button_other in buttons_map:
    #         buttons_map[button_other].setStyleSheet(f"background-color: #f0f0f0")

    def draw_foreground_point(self):
        self.canvas.setMapTool(self.tool_click_fg)
        button = self.wdg_sel.pushButton_fg  # self.sender()
        if button.isChecked():
            print(button.objectName(), "is checked")
        else:
            button.toggle()
        # self._set_button_selected('foreground')
        if self.wdg_sel.pushButton_bg.isChecked():
            self.wdg_sel.pushButton_bg.toggle()
        if self.wdg_sel.pushButton_rect.isChecked():
            self.wdg_sel.pushButton_rect.toggle()
        self.tool_click_bg.deactivate()
        self.prompt_type = 'fgpt'

    def draw_background_point(self):
        self.canvas.setMapTool(self.tool_click_bg)
        button = self.wdg_sel.pushButton_bg  # self.sender()
        if button.isChecked():
            print(button.objectName(), "is checked")
        else:
            button.toggle()
        # self._set_button_selected('background')
        if self.wdg_sel.pushButton_fg.isChecked():
            self.wdg_sel.pushButton_fg.toggle()
        if self.wdg_sel.pushButton_rect.isChecked():
            self.wdg_sel.pushButton_rect.toggle()
        self.tool_click_fg.deactivate()
        self.prompt_type = 'bgpt'

    def draw_rect(self):
        self.canvas.setMapTool(self.tool_click_rect)
        button = self.wdg_sel.pushButton_rect  # self.sender()
        if button.isChecked():
            print(button.objectName(), "is checked")
        else:
            button.toggle()
        # self._set_button_selected('rectangle')
        if self.wdg_sel.pushButton_fg.isChecked():
            self.wdg_sel.pushButton_fg.toggle()
        if self.wdg_sel.pushButton_bg.isChecked():
            self.wdg_sel.pushButton_bg.toggle()
        self.tool_click_fg.deactivate()
        self.tool_click_bg.deactivate()
        self.prompt_type = 'bbox'

    def find_file(self):
        path, _ = QFileDialog.getSaveFileName(None, "Save shapefile", "")
        self.wdg_sel.path_out.setText(path)

    def load_shp_file(self):
        text = self.wdg_sel.path_out.text()
        self.polygon = SAM_PolygonFeature(self.img_crs_manager, text)

    def find_feature(self):
        feature_dir_str = QFileDialog.getExistingDirectory(
            None, "get feature directory", "")
        self.wdg_sel.path_feature.setText(feature_dir_str)

    def load_feature(self):
        self.feature_dir = self.wdg_sel.path_feature.text()
        self._init_feature_related()
        self.load_shp_file()
        self.draw_foreground_point()

    def clear_layers(self):
        self.canvas_points.clear()
        self.canvas_rect.clear()
        if hasattr(self, "polygon"):
            self.polygon.rollback_changes()
        # self.activate_fg.emit()
        self.reset_label_tool()

    def save_shp_file(self):
        self.polygon.commit_changes()
        self.clear_layers()
        # self.activate_fg.emit()

    def reset_label_tool(self):
        if self.prompt_type == 'bbox':
            self.draw_rect()
        else:
            self.draw_foreground_point()
            self.prompt_type = 'fgpt'
