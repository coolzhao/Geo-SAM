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
from .ui.icons import QIcon_GeoSAMTool


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
        self.toolPan = QgsMapToolPan(self.canvas)
        self.dockFirstOpen = True
        self.prompt_history: List[str] = []
        self.sam_feature_history: List[List[int]] = []

    def initGui(self):
        self.action = QAction(
            QIcon_GeoSAMTool,
            "Geo-SAM Tool",
            self.iface.mainWindow()
        )
        self.action.triggered.connect(self.create_widget_selector)
        self.iface.addPluginToMenu('&Geo-SAM', self.action)
        self.iface.addToolBarIcon(self.action)

    def create_widget_selector(self):
        '''Create widget selector'''
        if self.dockFirstOpen:
            self._init_feature_related()
            self.load_demo_img()

            if self.receivers(self.execute_SAM) == 0:
                self.execute_SAM.connect(self.execute_segmentation)

            self.wdg_sel = UI_Selector
            # prompts
            self.wdg_sel.pushButton_fg.clicked.connect(
                self.draw_foreground_point)
            self.wdg_sel.pushButton_bg.clicked.connect(
                self.draw_background_point)
            self.wdg_sel.pushButton_rect.clicked.connect(self.draw_rect)

            # tools
            self.wdg_sel.pushButton_clear.clicked.connect(self.clear_layers)
            self.wdg_sel.pushButton_undo.clicked.connect(self.undo_last_prompt)
            self.wdg_sel.pushButton_save.clicked.connect(self.save_shp_file)

            self.wdg_sel.pushButton_find_file.clicked.connect(self.find_file)
            self.wdg_sel.pushButton_load_file.clicked.connect(
                self.load_shp_file)
            self.wdg_sel.pushButton_find_feature.clicked.connect(
                self.find_feature)
            self.wdg_sel.pushButton_load_feature.clicked.connect(
                self.load_feature)
            self.wdg_sel.radioButton_enable.toggled.connect(
                self.enable_disable)

            # set button checkable
            self.wdg_sel.pushButton_fg.setCheckable(True)
            self.wdg_sel.pushButton_bg.setCheckable(True)
            self.wdg_sel.pushButton_rect.setCheckable(True)

            # If a signal is connected to several slots,
            # the slots are activated in the same order in which the connections were made, when the signal is emitted.
            self.wdg_sel.closed.connect(self.destruct)
            self.wdg_sel.closed.connect(self.iface.actionPan().trigger)

            # shortcuts
            self.wdg_sel.pushButton_clear.setShortcut("C")
            self.wdg_sel.pushButton_undo.setShortcut("Z")
            self.wdg_sel.pushButton_save.setShortcut("S")

            self.shortcut_tab = QShortcut(
                QKeySequence(Qt.Key_Tab), self.wdg_sel)
            self.shortcut_tab.activated.connect(self.loop_prompt_type)
            self.shortcut_undo_sam_pg = QShortcut(
                QKeySequence(QKeySequence.Undo), self.wdg_sel)
            self.shortcut_undo_sam_pg.activated.connect(self.undo_sam_polygon)

            self.wdg_sel.setFloating(True)
            self.dockFirstOpen = False
        else:
            self.clear_layers()

        # add widget to QGIS
        self.iface.addDockWidget(Qt.TopDockWidgetArea, self.wdg_sel)

        # default is fgpt, but do not change when reloading feature folder
        self.reset_prompt_type()

    def destruct(self):
        '''Destruct actions when closed widget'''
        self.clear_layers()

    def unload(self):
        '''Unload actions when plugin is closed'''
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu('&Geo-SAM', self.action)
        self._clear_layers()

        if hasattr(self, "shortcut_tab"):
            self.shortcut_tab.disconnect()
        if hasattr(self, "shortcut_undo_sam_pg"):
            self.shortcut_undo_sam_pg.disconnect()
        del self.action

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
                else:
                    print("Demo image layer failed to load!")
                # self.iface.addRasterLayer(img_path, self.demo_img_name)
            else:
                print(img_path, 'does not exist')

    def topping_polygon_sam_layer(self):
        '''Topping polygon layer of SAM result to top of TOC'''
        root = QgsProject.instance().layerTreeRoot()
        tree_layer = root.findLayer(self.polygon.layer.id())

        if root.children()[0] == tree_layer:
            return None

        # move to top
        tl_clone = tree_layer.clone()
        root.insertChildNode(0, tl_clone)
        parent_tree_layer = tree_layer.parent()
        parent_tree_layer.removeChildNode(tree_layer)

    def clear_canvas_layers_safely(self):
        '''Clear canvas layers safely'''
        if hasattr(self, "canvas_points"):
            self.canvas_points.clear()
        if hasattr(self, "canvas_rect"):
            self.canvas_rect.clear()

    def _init_feature_related(self):
        '''Init or reload feature related objects'''

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
            self.prompt_history,
            self.execute_SAM,
        )
        self.tool_click_bg = ClickTool(
            self.canvas,
            self.canvas_points,
            'bgpt',
            self.prompt_history,
            self.execute_SAM,
        )
        self.tool_click_rect = RectangleMapTool(
            self.canvas_rect, self.prompt_history, self.execute_SAM, self.img_crs_manager
        )

    def loop_prompt_type(self):
        '''Loop prompt type'''
        if self.wdg_sel.pushButton_fg.isChecked():
            self.draw_background_point()
        elif self.wdg_sel.pushButton_bg.isChecked():
            self.draw_rect()
        elif self.wdg_sel.pushButton_rect.isChecked():
            self.draw_foreground_point()

    def undo_last_prompt(self):
        if len(self.prompt_history) > 0:
            prompt_last = self.prompt_history.pop()
            if prompt_last == 'bbox':
                self.canvas_rect.clear()
            else:
                if len(self.canvas_points.markers) > 0:
                    self.canvas_points.popPoint()
            self.execute_SAM.emit()

    def enable_disable(self):
        '''Enable or disable the widget selector'''
        radioButton = self.sender()
        if not radioButton.isChecked():
            # UI_Selector.setEnabled(False)
            self.canvas.setMapTool(self.toolPan)
            # findChildren(self, type, name: str = '', options: Union[Qt.FindChildOptions, Qt.FindChildOption] = Qt.FindChildrenRecursively): argument 2 has unexpected type 'FindChildOption'
            # for wdg in UI_Selector.findChildren(QPushButton, 'pushButton_fg'): # PyQt5.QtWidgets.QDockWidget
            self.wdg_sel.pushButton_fg.setEnabled(False)
            self.wdg_sel.pushButton_bg.setEnabled(False)
            self.wdg_sel.pushButton_rect.setEnabled(False)
            self.wdg_sel.pushButton_clear.setEnabled(False)
            self.wdg_sel.pushButton_undo.setEnabled(False)
            self.wdg_sel.pushButton_save.setEnabled(False)
        else:
            # for wdg in UI_Selector.findChildren(QWidget, 'pushButton', Qt.FindChildrenRecursively):
            #     print(wdg.objectName())
            #     wdg.setEnabled(True)
            self.wdg_sel.pushButton_fg.setEnabled(True)
            self.wdg_sel.pushButton_bg.setEnabled(True)
            self.wdg_sel.pushButton_rect.setEnabled(True)
            self.wdg_sel.pushButton_clear.setEnabled(True)
            self.wdg_sel.pushButton_undo.setEnabled(True)
            self.wdg_sel.pushButton_save.setEnabled(True)
            self.reset_prompt_type()

            # UI_Selector.setEnabled(True)

        # self.wdg_sel.radioButton_enable.setEnabled(True)

    def ensure_sam_feature_exist(self):
        layer_list = QgsProject.instance().mapLayersByName("polygon_sam")
        if len(layer_list) == 0 or not hasattr(self, "polygon"):
            self.load_shp_file()

    def execute_segmentation(self):
        self.ensure_sam_feature_exist()

        # add last id to history
        features = list(self.polygon.layer.getFeatures())
        if len(list(features)) == 0:
            last_id = 1
        else:
            last_id = features[-1].id() + 1

        if (len(self.sam_feature_history) >= 1 and
                len(self.sam_feature_history[-1]) == 1):
            self.sam_feature_history[-1][0] = last_id
        else:
            self.sam_feature_history.append([last_id])

        # execute segmentation
        self.sam_model.sam_predict(
            self.canvas_points, self.canvas_rect, self.polygon)
        self.topping_polygon_sam_layer()

    def draw_foreground_point(self):
        '''draw foreground point in canvas'''
        self.canvas.setMapTool(self.tool_click_fg)
        button = self.wdg_sel.pushButton_fg
        if not button.isChecked():
            button.toggle()

        if self.wdg_sel.pushButton_bg.isChecked():
            self.wdg_sel.pushButton_bg.toggle()
        if self.wdg_sel.pushButton_rect.isChecked():
            self.wdg_sel.pushButton_rect.toggle()
        self.prompt_type = 'fgpt'

    def draw_background_point(self):
        '''draw background point in canvas'''
        self.canvas.setMapTool(self.tool_click_bg)
        button = self.wdg_sel.pushButton_bg
        if not button.isChecked():
            button.toggle()

        if self.wdg_sel.pushButton_fg.isChecked():
            self.wdg_sel.pushButton_fg.toggle()
        if self.wdg_sel.pushButton_rect.isChecked():
            self.wdg_sel.pushButton_rect.toggle()
        self.prompt_type = 'bgpt'

    def draw_rect(self):
        '''draw rectangle in canvas'''
        self.canvas.setMapTool(self.tool_click_rect)
        button = self.wdg_sel.pushButton_rect  # self.sender()
        if not button.isChecked():
            button.toggle()

        if self.wdg_sel.pushButton_fg.isChecked():
            self.wdg_sel.pushButton_fg.toggle()
        if self.wdg_sel.pushButton_bg.isChecked():
            self.wdg_sel.pushButton_bg.toggle()
        self.prompt_type = 'bbox'

    def find_file(self):
        '''find shapefile path'''
        path, _ = QFileDialog.getSaveFileName(None, "Save shapefile", "")
        self.wdg_sel.path_out.setText(path)

    def load_shp_file(self):
        '''load shapefile'''
        text = self.wdg_sel.path_out.text()
        self.polygon = SAM_PolygonFeature(self.img_crs_manager, text)
        self.sam_feature_history = []

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
        '''Clear all temporary layers (canvas and new sam result) and reset prompt'''
        self.clear_canvas_layers_safely()
        if hasattr(self, "polygon"):
            self.polygon.rollback_changes()
        self.reset_prompt_type()
        self.prompt_history.clear()

    def save_shp_file(self):
        '''save sam result into shapefile layer'''
        self.clear_canvas_layers_safely()
        self.prompt_history.clear()
        if hasattr(self, "polygon"):
            self.polygon.commit_changes()

            # add last id of new features to history
            features = list(self.polygon.layer.getFeatures())
            if len(list(features)) == 0:
                return None
            last_id = features[-1].id()
            if self.sam_feature_history[-1][0] <= last_id:
                self.sam_feature_history[-1].append(last_id)

    def reset_prompt_type(self):
        '''reset prompt type'''
        if hasattr(self, "prompt_type"):
            if self.prompt_type == 'bbox':
                self.draw_rect()
            else:
                self.draw_foreground_point()
        else:
            self.draw_foreground_point()

    def undo_sam_polygon(self):
        '''undo last sam polygon'''
        if len(self.sam_feature_history) == 0:
            return None
        last_ids = self.sam_feature_history.pop(-1)
        if len(last_ids) == 1:
            self.clear_layers()
            return None
        rm_ids = list(range(last_ids[0], last_ids[1]+1))
        self.polygon.layer.dataProvider().deleteFeatures(rm_ids)

        # If caching is enabled, a simple canvas refresh might not be sufficient
        # to trigger a redraw and must clear the cached image for the layer
        if self.canvas.isCachingEnabled():
            self.polygon.layer.triggerRepaint()
        else:
            self.canvas.refresh()
