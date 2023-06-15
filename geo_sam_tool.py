import os
import typing
from qgis.core import QgsProject, Qgis, QgsMessageLog
from qgis.gui import QgsMapToolPan
from qgis.core import QgsRasterLayer
from qgis.PyQt.QtWidgets import QDockWidget
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
from .ui import UI_Selector


class Geo_SAM(QObject):
    execute_SAM = pyqtSignal()
    activate_fg = pyqtSignal()

    def __init__(self, iface, cwd: str):
        super().__init__()
        self.iface = iface
        self.cwd = cwd
        self.canvas = iface.mapCanvas()
        self.demo_img_name = "beiluhe_google_img_201211_clip"
        feature_dir = cwd + "/features/" + self.demo_img_name
        # feature_dir = r"D:\Data\sam_data\features\mosaic_scripts_wp_4.77_sub_114"
        self.feature_dir = feature_dir
        self.toolPan = QgsMapToolPan(self.canvas)

    def initGui(self):
        icon_path = os.path.join(self.cwd, "icons/geo_sam_tool.svg")

        self.action = QAction(
            QIcon(icon_path),
            "Geo-SAM Tool",
            self.iface.mainWindow()
        )
        self.action.triggered.connect(self.create_widget_selector)
        # self.toolbar.addAction(self.action)
        self.iface.addPluginToMenu('&Geo-SAM', self.action)
        self.iface.addToolBarIcon(self.action)

    def load_demo_img(self):
        layer_list = QgsProject.instance().mapLayersByName(self.demo_img_name)
        print(layer_list)
        if layer_list:
            rlayer = layer_list[0]
        else:
            img_path = os.path.join(
                self.cwd, "rasters", self.demo_img_name+'.tif')
            # img_path = r"D:\Data\sam_data\rasters\mosaic_scripts_wp_4.77_sub_114.tif"
            if os.path.exists(img_path):
                rlayer = QgsRasterLayer(img_path, self.demo_img_name)
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                else:
                    print("Demo image layer failed to load!")
                # self.iface.addRasterLayer(img_path, self.demo_img_name)
            else:
                print(img_path, 'does not exist')

    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu('&Geo-SAM', self.action)
        del self.action

    def topping_polygon_sam_layer(self):
        if hasattr(self, "polygon"):
            rg = self.iface.layerTreeCanvasBridge().rootGroup()
            order = rg.layerOrder()
            if self.polygon.layer in order and order[0] != self.polygon.layer:
                order.remove(self.polygon.layer)
                order.insert(0, self.polygon.layer)
            rg.setCustomLayerOrder(order)

    def clear_canvas_layers_safely(self):
        if hasattr(self, "canvas_points"):
            self.canvas_points.clear()
        if hasattr(self, "canvas_rect"):
            self.canvas_rect.clear()

    def _init_feature_related(self):
        '''Init or reload feature related objects'''

        # clear canvas layers when loading new image
        self.clear_canvas_layers_safely()

        # init feature related objects
        self.sam_model = SAM_Model(self.feature_dir, self.cwd)
        self.img_crs_manager = ImageCRSManager(self.sam_model.img_crs)
        self.canvas_points = Canvas_Points(self.canvas, self.img_crs_manager)
        self.canvas_rect = Canvas_Rectangle(self.canvas, self.img_crs_manager)

        # reset canvas extent
        canvas = self.iface.mapCanvas()
        extent_canvas = self.img_crs_manager.img_extent_to_crs(
            self.sam_model.extent, QgsProject.instance().crs())
        canvas.setExtent(extent_canvas)
        canvas.refresh()

        # self.shortcut_undo = QShortcut(QKeySequence(Qt.ControlModifier + Qt.Key_Z), self.iface.mainWindow())
        # self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self.iface.mainWindow())
        # self.shortcut_undo.setContext(Qt.ApplicationShortcut)
        # self.shortcut_undo.activated.connect(self.execute_segmentation)

        # init tools
        self.tool_click_fg = ClickTool(
            self.canvas,
            self.canvas_points,
            'fgpt',
            self.execute_SAM,
        )
        self.tool_click_bg = ClickTool(
            self.canvas,
            self.canvas_points,
            'bgpt',
            self.execute_SAM,
        )
        self.tool_click_rect = RectangleMapTool(
            self.canvas_rect, self.execute_SAM, self.img_crs_manager
        )

    def create_widget_selector(self):
        '''Create widget selector'''

        # clear canvas layers if reloaded
        self.clear_canvas_layers_safely()

        self._init_feature_related()
        self.load_demo_img()
        receiversCount = self.receivers(self.execute_SAM)
        if receiversCount == 0:
            self.execute_SAM.connect(self.execute_segmentation)
            self.activate_fg.connect(self.draw_foreground_point)

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
        # self.wdg_sel.closed.connect(self.destruct()) # qgis.gui.QgsDockWidget, actually QDockWidget no closed signal
        # self.wdg_sel.closeEvent = self.destruct
        # self.wdg_sel.visibilityChanged.connect(self.destruct)

        # add widget to QGIS
        self.wdg_sel.setFloating(True)
        self.iface.addDockWidget(Qt.TopDockWidgetArea, self.wdg_sel)

        # default is fgpt, but do not change when reloading feature folder
        self.reset_prompt_type()

    def destruct(self):
        self.save_shp_file()
        receiversCount = self.receivers(self.execute_SAM)
        if receiversCount > 0:
            self.execute_SAM.disconnect()
            self.activate_fg.disconnect()
        self.canvas.setMapTool(self.toolPan)
        # self.wdg_sel.destroy()

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
            self.wdg_sel.pushButton_clear.setEnabled(False)
            self.wdg_sel.pushButton_save.setEnabled(False)
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
            self.wdg_sel.pushButton_clear.setEnabled(True)
            self.wdg_sel.pushButton_save.setEnabled(True)
            self.reset_prompt_type()

            # UI_Selector.setEnabled(True)

        # self.wdg_sel.radioButton_enable.setEnabled(True)

    def execute_segmentation(self):
        if not hasattr(self, "polygon"):
            self.load_shp_file()
        self.sam_model.sam_predict(
            self.canvas_points, self.canvas_rect, self.polygon)
        self.topping_polygon_sam_layer()

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
        if self.feature_dir is not None and os.path.exists(self.feature_dir):
            self.clear_layers()
            self._init_feature_related()
            self.load_shp_file()
            # self.draw_foreground_point()
            self.reset_prompt_type()  # do not change tool
        else:
            self.iface.messageBar().pushMessage("Feature folder not exist",
                                                "choose a another folder", level=Qgis.Info)

    def clear_layers(self):
        self.clear_canvas_layers_safely()
        if hasattr(self, "polygon"):
            self.polygon.rollback_changes()
        # self.activate_fg.emit()
        self.reset_prompt_type()

    def save_shp_file(self):
        if hasattr(self, "polygon"):
            self.polygon.commit_changes()
        self.clear_layers()
        # self.activate_fg.emit()

    def reset_prompt_type(self):
        if hasattr(self, "prompt_type"):
            if self.prompt_type == 'bbox':
                self.draw_rect()
            else:
                self.draw_foreground_point()
        else:
            self.draw_foreground_point()
