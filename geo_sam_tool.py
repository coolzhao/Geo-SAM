import os
from typing import List
from qgis.core import QgsProject, Qgis, QgsMessageLog, QgsApplication
from qgis.gui import QgsMapToolPan, QgisInterface
from qgis.core import QgsRasterLayer
from qgis.PyQt.QtWidgets import QDockWidget
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QFileDialog,
    QAction,
    QFileDialog,
    QApplication,
    QShortcut

)
from PyQt5.QtGui import QKeySequence, QIcon, QColor
from PyQt5 import uic

from .tools.geoTool import ImageCRSManager, LayerExtent
from .tools.SAMTool import SAM_Model
from .tools.canvasTool import RectangleMapTool, ClickTool, Canvas_Points, Canvas_Rectangle, SAM_PolygonFeature
from .ui import UI_Selector


class Geo_SAM(QObject):
    execute_SAM = pyqtSignal()

    def __init__(self, iface: QgisInterface, cwd: str):
        super().__init__()
        self.iface = iface
        self.cwd = cwd
        self.canvas = iface.mapCanvas()
        self.demo_img_name = "beiluhe_google_img_201211_clip"
        feature_dir = cwd + "/features/" + self.demo_img_name
        # feature_dir = r"D:\Data\sam_data\features\mosaic_scripts_wp_4.77_sub_114"
        self.feature_dir = feature_dir
        self.toolPan = QgsMapToolPan(self.canvas)
        self.dockFirstOpen = True
        self.prompts: List = []

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
        self._clear_layers()

        # TODO: could use stortcut after reload plugin
        if hasattr(self, "shortcut_undo"):
            self.shortcut_undo.disconnect()
        if hasattr(self, "shortcut_save"):
            self.shortcut_save.disconnect()
        if hasattr(self, "shortcut_clear"):
            self.shortcut_clear.disconnect()

        del self.action

    def topping_polygon_sam_layer(self):
        if hasattr(self, "polygon"):
            root = QgsProject.instance().layerTreeRoot()
            tree_layer = root.findLayer(self.polygon.layer.id())
            tl_clone = tree_layer.clone()
            root.insertChildNode(0, tl_clone)
            root.removeChildNode(tree_layer)

    def clear_canvas_layers_safely(self):
        '''Clear canvas layers safely'''
        if hasattr(self, "canvas_points"):
            self.canvas_points.clear()
        if hasattr(self, "canvas_rect"):
            self.canvas_rect.clear()

    def _init_feature_related(self):
        '''Init or reload feature related objects'''

        # clear layers when loading new image
        # self.clear_layers()

        # init feature related objects
        self.sam_model = SAM_Model(self.feature_dir, self.cwd)
        self.img_crs_manager = ImageCRSManager(self.sam_model.img_crs)
        self.canvas_points = Canvas_Points(self.canvas, self.img_crs_manager)
        self.canvas_rect = Canvas_Rectangle(self.canvas, self.img_crs_manager)

        # reset canvas extent
        extent_canvas = self.img_crs_manager.img_extent_to_crs(
            self.sam_model.extent,
            QgsProject.instance().crs()
        )
        self.canvas.setExtent(extent_canvas)
        self.canvas.refresh()

        # init tools
        self.tool_click_fg = ClickTool(
            self.canvas,
            self.canvas_points,
            'fgpt',
            self.prompts,
            self.execute_SAM,
        )
        self.tool_click_bg = ClickTool(
            self.canvas,
            self.canvas_points,
            'bgpt',
            self.prompts,
            self.execute_SAM,
        )
        self.tool_click_rect = RectangleMapTool(
            self.canvas_rect, self.prompts, self.execute_SAM, self.img_crs_manager
        )

    def create_widget_selector(self):
        '''Create widget selector'''
        QgsMessageLog.logMessage(
            f'create widget selector: dockFirstOpen : {self.dockFirstOpen} ', 'Geo SAM', Qgis.Info)
        if self.dockFirstOpen:
            self._init_feature_related()
            self.load_demo_img()
            receiversCount = self.receivers(self.execute_SAM)
            if receiversCount == 0:
                self.execute_SAM.connect(self.execute_segmentation)

            self.wdg_sel = UI_Selector
            self.wdg_sel.pushButton_fg.clicked.connect(
                self.draw_foreground_point)
            self.wdg_sel.pushButton_bg.clicked.connect(
                self.draw_background_point)
            self.wdg_sel.pushButton_rect.clicked.connect(self.draw_rect)
            self.wdg_sel.pushButton_clear.clicked.connect(self.clear_layers)
            self.wdg_sel.pushButton_find_file.clicked.connect(self.find_file)
            self.wdg_sel.pushButton_load_file.clicked.connect(
                self.load_shp_file)
            self.wdg_sel.pushButton_save.clicked.connect(self.save_shp_file)
            self.wdg_sel.pushButton_find_feature.clicked.connect(
                self.find_feature)
            self.wdg_sel.pushButton_load_feature.clicked.connect(
                self.load_feature)

            self.wdg_sel.radioButton_enable.setChecked(True)
            self.wdg_sel.radioButton_enable.toggled.connect(
                self.enable_disable)
            self.wdg_sel.pushButton_fg.setCheckable(True)
            self.wdg_sel.pushButton_bg.setCheckable(True)
            self.wdg_sel.pushButton_rect.setCheckable(True)

            self.wdg_sel.setFloating(True)

            # qgis.gui.QgsDockWidget, actually QDockWidget no closed signal
            # If a signal is connected to several slots,
            # the slots are activated in the same order in which the connections were made, when the signal is emitted.
            self.wdg_sel.closed.connect(self.destruct)
            self.wdg_sel.closed.connect(self.iface.actionPan().trigger)
            # self.iface.actionAddFeature
            # self.wdg_sel.closeEvent = self.destruct
            # self.wdg_sel.visibilityChanged.connect(self.destruct)

            self.shortcut_save = QShortcut(
                QKeySequence("S"), self.iface.mainWindow())
            self.shortcut_save.activated.connect(self.save_shp_file)

            self.shortcut_undo = QShortcut(
                QKeySequence('Z'), self.iface.mainWindow())
            self.shortcut_undo.activated.connect(self.undo_last_prompt)

            self.shortcut_clear = QShortcut(
                QKeySequence('C'), self.iface.mainWindow())
            self.shortcut_clear.activated.connect(self.clear_layers)

            self.dockFirstOpen = False
        else:
            self.clear_layers()

        # add widget to QGIS
        self.iface.addDockWidget(Qt.TopDockWidgetArea, self.wdg_sel)

        # default is fgpt, but do not change when reloading feature folder
        self.reset_prompt_type()

    def register_shortcuts(self):
        # Unregister existing shortcuts, if any
        self.iface.unregisterMainWindowActions()

        # Register new shortcuts
        action = self.iface.registerMainWindowAction(
            "my_plugin:my_action", "My Action", self.myAction, QKeySequence("Ctrl+M"))
        self.iface.addPluginToMenu("&My Plugin", action)

    def undo_last_prompt(self):
        if len(self.prompts) > 0:
            prompt_last = self.prompts.pop()
            QgsMessageLog.logMessage(
                f'undo last prompt {prompt_last}', 'Geo SAM', Qgis.Info)
            if prompt_last == 'bbox':
                self.canvas_rect.clear()
            else:
                if len(self.canvas_points.markers) > 0:
                    self.canvas_points.popPoint()
            self.execute_SAM.emit()

    def destruct(self):
        '''Destruct actions when closed widget'''
        # TODO: make it work
        # self.iface.actionPan().trigger()
        # self.canvas.unsetMapTool(self.canvas.mapTool())
        # self.iface.mapCanvas().unsetMapTool(self.tool_click_bg)
        # self.iface.mapCanvas().unsetMapTool(self.tool_click_fg)
        # self.iface.mapCanvas().unsetMapTool(self.tool_click_rect)
        # self.canvas.setMapTool(self.toolPan)
        self.clear_layers()

    def enable_disable(self):
        '''Enable or disable the widget selector'''
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
        '''draw foreground point in canvas'''
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
        '''draw background point in canvas'''
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
        '''draw rectangle in canvas'''
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
        '''find shapefile path'''
        path, _ = QFileDialog.getSaveFileName(None, "Save shapefile", "")
        self.wdg_sel.path_out.setText(path)

    def load_shp_file(self):
        '''load shapefile'''
        text = self.wdg_sel.path_out.text()
        self.polygon = SAM_PolygonFeature(self.img_crs_manager, text)

    def find_feature(self):
        '''find feature directory'''
        feature_dir_str = QFileDialog.getExistingDirectory(
            None, "get feature directory", "")
        self.wdg_sel.path_feature.setText(feature_dir_str)

    def load_feature(self):
        '''load feature'''
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

    def _clear_layers(self):
        '''Clear all temporary layers (canvas and new sam result)'''
        self.clear_canvas_layers_safely()
        if hasattr(self, "polygon"):
            self.polygon.rollback_changes()

    def clear_layers(self):
        '''Clear all temporary layers (canvas and new sam result) and reset prompt type'''
        self.clear_canvas_layers_safely()
        if hasattr(self, "polygon"):
            self.polygon.rollback_changes()
        self.reset_prompt_type()
        self.prompts.clear()

    def save_shp_file(self):
        '''save sam result into shapefile layer'''
        if hasattr(self, "polygon"):
            self.polygon.commit_changes()
        self.clear_canvas_layers_safely()
        self.prompts.clear()

    def reset_prompt_type(self):
        '''reset prompt type'''
        if hasattr(self, "prompt_type"):
            if self.prompt_type == 'bbox':
                self.draw_rect()
            else:
                self.draw_foreground_point()
        else:
            self.draw_foreground_point()
