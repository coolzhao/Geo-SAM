import os
import time
import json
from typing import List
from pathlib import Path
from qgis.core import QgsProject, Qgis, QgsMessageLog, QgsApplication, QgsCoordinateReferenceSystem
from qgis.gui import QgsMapToolPan, QgisInterface, QgsFileWidget
from qgis.core import QgsRasterLayer, QgsRectangle
from qgis.utils import iface
from qgis.PyQt.QtWidgets import QDockWidget
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QFileDialog,
    QAction,
    QFileDialog,
    QApplication,
    QShortcut,
    QToolBar,
    QMessageBox,
)
from PyQt5.QtGui import QKeySequence, QIcon, QColor
from PyQt5 import uic
import processing

from .geoTool import ImageCRSManager, LayerExtent
from .SAMTool import SAM_Model
from .canvasTool import RectangleMapTool, ClickTool, Canvas_Points, Canvas_Rectangle, SAM_PolygonFeature, Canvas_Extent
from ..ui import UI_Selector, UI_EncoderCopilot
from ..ui.icons import QIcon_GeoSAMTool, QIcon_EncoderTool, QIcon_EncoderCopilot
from ..geo_sam_provider import GeoSamProvider


class Selector(QObject):
    execute_SAM = pyqtSignal()

    def __init__(self, parent, iface: QgisInterface, cwd: str):
        super().__init__()
        self.parent = parent
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

    def open_widget(self):
        '''Create widget selector'''
        self.parent.toolbar.setVisible(True)
        if self.dockFirstOpen:
            self._init_feature_related()
            self.load_demo_img()

            if self.receivers(self.execute_SAM) == 0:
                self.execute_SAM.connect(self.execute_segmentation)

            self.wdg_sel = UI_Selector
            ########## prompts table ##########
            self.wdg_sel.pushButton_fg.clicked.connect(
                self.draw_foreground_point)
            self.wdg_sel.pushButton_bg.clicked.connect(
                self.draw_background_point)
            self.wdg_sel.pushButton_rect.clicked.connect(self.draw_rect)

            # tools
            self.wdg_sel.pushButton_clear.clicked.connect(self.clear_layers)
            self.wdg_sel.pushButton_undo.clicked.connect(self.undo_last_prompt)
            self.wdg_sel.pushButton_save.clicked.connect(self.save_shp_file)

            self.wdg_sel.pushButton_load_file.clicked.connect(
                self.load_shp_file)
            self.wdg_sel.pushButton_load_feature.clicked.connect(
                self.load_feature)
            self.wdg_sel.radioButton_enable.setChecked(True)
            self.wdg_sel.radioButton_enable.toggled.connect(
                self.enable_disable_edit_mode)

            self.wdg_sel.radioButton_exe_move.setChecked(False)
            self.wdg_sel.radioButton_exe_move.toggled.connect(
                self.execute_sam_move_mode)

            # set filter for file dialog
            self.wdg_sel.QgsFile_shapefile.setFilter("*.shp")
            self.wdg_sel.QgsFile_shapefile.setStorageMode(
                QgsFileWidget.SaveFile)
            self.wdg_sel.QgsFile_feature.setStorageMode(
                QgsFileWidget.GetDirectory)

            # set button checkable
            self.wdg_sel.pushButton_fg.setCheckable(True)
            self.wdg_sel.pushButton_bg.setCheckable(True)
            self.wdg_sel.pushButton_rect.setCheckable(True)

            ######### Setting table #########
            self.wdg_sel.Box_min_area.valueChanged.connect(
                self.filter_feature_by_area)
            self.wdg_sel.radioButton_show_extent.setChecked(True)
            self.wdg_sel.radioButton_show_extent.toggled.connect(
                self.show_hide_sam_feature_extent)

            # If a signal is connected to several slots,
            # the slots are activated in the same order in which the connections were made, when the signal is emitted.
            self.wdg_sel.closed.connect(self.destruct)
            self.wdg_sel.closed.connect(self.iface.actionPan().trigger)

            # shortcuts
            self.wdg_sel.pushButton_clear.setShortcut("C")
            self.wdg_sel.pushButton_undo.setShortcut("Z")
            self.wdg_sel.pushButton_save.setShortcut("S")
            self.wdg_sel.radioButton_exe_move.setShortcut("M")

            self.shortcut_tab = QShortcut(
                QKeySequence(Qt.Key_Tab), self.wdg_sel)
            self.shortcut_tab.activated.connect(self.loop_prompt_type)
            self.shortcut_undo_sam_pg = QShortcut(
                QKeySequence(QKeySequence.Undo), self.wdg_sel)
            self.shortcut_undo_sam_pg.activated.connect(self.undo_sam_polygon)
            # self.wdg_sel.setFloating(True)

            # default is fgpt, but do not change when reloading feature folder
            # self.reset_prompt_type()
            self.dockFirstOpen = False
            # add widget to QGIS
            self.iface.addDockWidget(Qt.TopDockWidgetArea, self.wdg_sel)
        else:
            self.clear_layers(clear_extent=True)

        self.enable_disable_edit_mode()
        self.show_hide_sam_feature_extent()

        if not self.wdg_sel.isUserVisible():
            self.wdg_sel.setUserVisible(True)

    def destruct(self):
        '''Destruct actions when closed widget'''
        self.clear_layers(clear_extent=True)

    def unload(self):
        '''Unload actions when plugin is closed'''
        self.clear_layers(clear_extent=True)

        if hasattr(self, "shortcut_tab"):
            self.shortcut_tab.disconnect()
        if hasattr(self, "shortcut_undo_sam_pg"):
            self.shortcut_undo_sam_pg.disconnect()

        self.iface.removeDockWidget(self.wdg_sel)

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

        if tree_layer is None:
            return None
        if not tree_layer.isVisible():
            tree_layer.setItemVisibilityChecked(True)
        if root.children()[0] == tree_layer:
            return None

        # move to top
        tl_clone = tree_layer.clone()
        root.insertChildNode(0, tl_clone)
        parent_tree_layer = tree_layer.parent()
        parent_tree_layer.removeChildNode(tree_layer)

    def clear_canvas_layers_safely(self, clear_extent: bool = False):
        '''Clear canvas layers safely'''
        if hasattr(self, "canvas_points"):
            self.canvas_points.clear()
        if hasattr(self, "canvas_rect"):
            self.canvas_rect.clear()
        if hasattr(self, "canvas_extent") and clear_extent:
            self.canvas_extent.clear()
        # if hasattr(self, "polygon_temp"):
        #     self.polygon_temp.rollback_changes()

    def _init_feature_related(self):
        '''Init or reload feature related objects'''

        # init feature related objects
        self.sam_model = SAM_Model(self.feature_dir, self.cwd)
        self.iface.messageBar().pushMessage("Great",
                                            (f"SAM Features with {self.sam_model.feature_size} patches in '{Path(self.feature_dir).name}' have been loaded, "
                                             "you can start labeling now"), level=Qgis.Info, duration=10)

        self.img_crs_manager = ImageCRSManager(self.sam_model.img_crs)
        self.canvas_points = Canvas_Points(self.canvas, self.img_crs_manager)
        self.canvas_rect = Canvas_Rectangle(self.canvas, self.img_crs_manager)
        self.canvas_extent = Canvas_Extent(self.canvas, self.img_crs_manager)

        # reset canvas extent
        self.sam_extent_canvas_crs = self.img_crs_manager.img_extent_to_crs(
            self.sam_model.extent,
            QgsProject.instance().crs()
        )
        self.canvas.setExtent(self.sam_extent_canvas_crs)
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
            self.canvas_rect,
            self.prompt_history,
            self.execute_SAM,
            self.img_crs_manager
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
                # self.canvas_rect.clear()
                self.canvas_rect.popRect()
            else:
                self.canvas_points.popPoint()
            self.execute_SAM.emit()

    def enable_disable_edit_mode(self):
        '''Enable or disable the widget selector'''
        # radioButton = self.sender()
        radioButton = self.wdg_sel.radioButton_enable
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

    def show_hide_sam_feature_extent(self):
        '''Show or hide extent of SAM encoded feature'''
        if self.wdg_sel.radioButton_show_extent.isChecked():
            if hasattr(self, "sam_extent_canvas_crs"):
                self.canvas_extent.add_extent(self.sam_extent_canvas_crs)
            else:
                self.iface.messageBar().pushMessage("Oops",
                                                    ("No sam feature loaded"), level=Qgis.Info, duration=10)
        else:
            self.canvas_extent.clear()

    def execute_sam_move_mode(self):
        if self.wdg_sel.radioButton_exe_move.isChecked():
            self.tool_click_fg.move_mode = True
            self.tool_click_bg.move_mode = True
            self.tool_click_rect.move_mode = True
        else:
            self.tool_click_fg.move_mode = False
            self.tool_click_bg.move_mode = False
            self.tool_click_rect.move_mode = False

    def filter_feature_by_area(self):
        t_area = self.wdg_sel.Box_min_area.value()
        self.polygon.t_area = t_area

    def ensure_polygon_sam_exist(self):
        if hasattr(self, "polygon"):
            layer = QgsProject.instance().mapLayer(self.polygon.layer_id)
            if layer:
                return None
        self.load_shp_file()

    def execute_segmentation(self) -> bool:
        # check last prompt inside feature extent
        if len(self.prompt_history) > 0:
            prompt_last = self.prompt_history[-1]
            if prompt_last == 'bbox':
                last_rect = self.canvas_rect.extent
                last_prompt = QgsRectangle(
                    last_rect[0], last_rect[2], last_rect[1], last_rect[3])
            else:
                last_point = self.canvas_points.points_img_crs[-1]
                last_prompt = QgsRectangle(last_point, last_point)
            if not last_prompt.intersects(self.sam_model.extent):
                if not self.message_box_outside():
                    self.undo_last_prompt()
                return False

        self.ensure_polygon_sam_exist()

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
        if not self.sam_model.sam_predict(
                self.canvas_points, self.canvas_rect, self.polygon, self.prompt_history):
            self.undo_last_prompt()
        self.topping_polygon_sam_layer()

        return True

    # def execute_segmentation_temp(self) -> bool:
    #     '''Execute segmentation for temporary polygon when mouse move'''
    #     # check last prompt inside feature extent
    #     if len(self.prompt_history) > 0:
    #         prompt_last = self.prompt_history[-1]
    #         if prompt_last == 'bbox':
    #             last_rect = self.canvas_rect.extent
    #             last_prompt = QgsRectangle(
    #                 last_rect[0], last_rect[2], last_rect[1], last_rect[3])
    #         else:
    #             last_point = self.canvas_points.points_img_crs[-1]
    #             last_prompt = QgsRectangle(last_point, last_point)
    #         if not last_prompt.intersects(self.sam_model.extent):
    #             if not self.message_box_outside():
    #                 self.undo_last_prompt()
    #             return False

    #     self.ensure_polygon_sam_exist()

    #     # execute segmentation
    #     if self.sam_model.sam_predict(
    #             self.canvas_points, self.canvas_rect, self.polygon_temp, self.prompt_history):
    #         return True

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

    def load_shp_file(self):
        '''load shapefile'''
        file_path = self.wdg_sel.QgsFile_shapefile.filePath()
        self.sam_feature_history = []
        if hasattr(self, "polygon"):
            layer = QgsProject.instance().mapLayer(self.polygon.layer_id)
            if layer and layer.source() == file_path:
                return None

        self.polygon = SAM_PolygonFeature(
            self.img_crs_manager, file_path)
        # self.polygon_temp = SAM_PolygonFeature(
        #     self.img_crs_manager, default_name='polygon_sam_temp')  # for drawing when mouse move

    def load_feature(self):
        '''load feature'''
        self.feature_dir = self.wdg_sel.QgsFile_feature.filePath()
        if self.feature_dir is not None and os.path.exists(self.feature_dir):
            self.clear_layers(clear_extent=True)
            self._init_feature_related()
            # self.load_shp_file()
            # self.draw_foreground_point()
            # if self.wdg_sel.radioButton_enable.isChecked():
            #     self.reset_prompt_type()  # do not change tool
            self.enable_disable_edit_mode()
            self.show_hide_sam_feature_extent()
        else:
            self.iface.messageBar().pushMessage("Feature folder not exist",
                                                "choose a another folder", level=Qgis.Info)

    def clear_layers(self, clear_extent: bool = False):
        '''Clear all temporary layers (canvas and new sam result) and reset prompt'''
        self.clear_canvas_layers_safely(clear_extent=clear_extent)
        if hasattr(self, "polygon"):
            self.polygon.rollback_changes()
        # if hasattr(self, "polygon_temp"):
        #     self.polygon_temp.rollback_changes()
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
            self.clear_layers(clear_extent=False)
            return None
        rm_ids = list(range(last_ids[0], last_ids[1]+1))
        self.polygon.layer.dataProvider().deleteFeatures(rm_ids)

        # If caching is enabled, a simple canvas refresh might not be sufficient
        # to trigger a redraw and must clear the cached image for the layer
        if self.canvas.isCachingEnabled():
            self.polygon.layer.triggerRepaint()
        else:
            self.canvas.refresh()

    def message_box_outside(self):
        mb = QMessageBox()
        mb.setText(
            'Point/rectangle is located outside of the feature boundary, click OK to undo last prompt.')
        mb.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        return_value = mb.exec()
        # TODO: Clear last point falls outside the boundary
        if return_value == QMessageBox.Ok:
            return False
        elif return_value == QMessageBox.Cancel:
            return True


class EncoderCopilot(QObject):
    # TODO: support encoding process in this widget
    def __init__(self, parent, iface: QgisInterface, cwd: str):
        super().__init__()
        self.parent = parent
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.toolPan = QgsMapToolPan(self.canvas)
        self.dockFirstOpen: bool = True
        # init crs
        self.project: QgsProject = QgsProject.instance()
        self.crs_project: QgsCoordinateReferenceSystem = self.project.crs()
        self.crs_layer: QgsCoordinateReferenceSystem = self.crs_project

    def open_widget(self):
        '''Create widget selector'''
        self.parent.toolbar.setVisible(True)
        if self.dockFirstOpen:
            self.wdg_copilot = UI_EncoderCopilot

            ########## prompts table ##########
            self.wdg_copilot.mMapLayerComboBox.layerChanged.connect(
                self.set_band)
            self.wdg_copilot.mExtentGroupBox.setMapCanvas(self.canvas)
            self.wdg_copilot.mExtentGroupBox.extentChanged.connect(
                self.show_extent)
            self.wdg_copilot.pushButton_CopySetting.clicked.connect(
                self.json_setting_to_clipboard)
            # If a signal is connected to several slots,
            # the slots are activated in the same order in which the connections were made, when the signal is emitted.
            self.wdg_copilot.closed.connect(self.destruct)
            self.wdg_copilot.closed.connect(self.iface.actionPan().trigger)
            self.wdg_copilot.closed.connect(self.reset_to_project_crs)

            self.dockFirstOpen = False
            # add widget to QGIS
            self.iface.addDockWidget(Qt.BottomDockWidgetArea, self.wdg_copilot)
        else:
            pass

        if not self.wdg_copilot.isUserVisible():
            self.wdg_copilot.setUserVisible(True)

    def set_band(self):
        '''init band field by raster layer band and set project crs to layer crs'''
        # set raster layer band to band field
        self.raster_layer = self.wdg_copilot.mMapLayerComboBox.currentLayer()
        self.wdg_copilot.mRasterBandComboBox_R.setLayer(self.raster_layer)
        self.wdg_copilot.mRasterBandComboBox_G.setLayer(self.raster_layer)
        self.wdg_copilot.mRasterBandComboBox_B.setLayer(self.raster_layer)

        # set crs to layer crs
        self.crs_project = self.project.crs()
        self.crs_layer = self.raster_layer.crs()

        if self.crs_layer != self.crs_project:
            self.project.setCrs(self.crs_layer)
            self.iface.messageBar().pushMessage(
                "Note:",
                "Project crs has been changed to the layer crs temporarily. "
                "It will be reset to the original crs when this widget is closed.",
                level=Qgis.Info,
                duration=30)
        if not hasattr(self, "canvas_extent"):
            self.canvas_extent = Canvas_Extent(self.canvas, self.crs_layer)

    def get_resolutions(self):
        '''Get x, y resolution from resolution group box'''
        scale = self.wdg_copilot.mBoxResolutionScale.value()
        resolution_layer = (
            self.raster_layer.rasterUnitsPerPixelX(),
            self.raster_layer.rasterUnitsPerPixelY()
        )
        resolution_scaled = (
            resolution_layer[0] * scale,
            resolution_layer[1] * scale
        )
        return resolution_scaled

    def get_stride(self):
        '''Get stride from overlap group box'''
        overlaps = self.wdg_copilot.mBoxOverlap.value()
        stride = int((100 - overlaps)/100 * 1024)
        return stride

    def get_extent_str(self):
        '''Get extent string from extent group box'''
        extent = self.wdg_copilot.mExtentGroupBox.outputExtent()
        xMin, yMin, xMax, yMax = (extent.xMinimum(),
                                  extent.yMinimum(),
                                  extent.xMaximum(),
                                  extent.yMaximum())
        return f'{xMin}, {xMax}, {yMin}, {yMax}'

    def check_extent_set(self):
        '''Check if extent has been set. If not, alert user to set extent'''
        extent = self.get_extent_str().split('[')[0]
        vals = extent.split(',')
        if (float(vals[0]) == float(vals[1])
                or float(vals[2]) == float(vals[3])):
            # alert user to set extent
            mb = QMessageBox()
            mb.setText(
                "Oops: Extent has not been set. Please set extent first.")
            mb.setStandardButtons(QMessageBox.Ok)
            mb.exec()
            return False
        else:
            return True

    def check_raster_selected(self):
        '''Check if raster layer has been selected. If not, alert user to select a raster layer'''
        if hasattr(self, "canvas_extent"):
            return True
        else:
            # alert user to select a raster layer
            mb = QMessageBox()
            mb.setText(
                "Oops: "
                "Raster Layer has not been selected. "
                "Please set Raster Layer and Bands first."
            )
            mb.setStandardButtons(QMessageBox.Ok)
            mb.exec()
            return False

    def check_setting_available(self):
        '''Check if setting is available. If not, alert user to set setting'''
        if not self.check_raster_selected():
            return False
        elif not self.check_extent_set():
            return False
        else:
            return True

    def json_setting_to_clipboard(self):
        if not self.check_setting_available():
            return None
        stride = self.get_stride()
        resolution = self.get_resolutions()
        bands = [self.wdg_copilot.mRasterBandComboBox_R.currentBand(),
                 self.wdg_copilot.mRasterBandComboBox_G.currentBand(),
                 self.wdg_copilot.mRasterBandComboBox_B.currentBand()]
        crs = self.crs_layer.authid()
        extent = f'{self.get_extent_str()} [{crs}]'
        json_dict = {
            "inputs":
                {"BANDS": bands,
                 "CRS": crs,
                 "EXTENT": extent,
                 "INPUT": self.raster_layer.source(),
                 # TODO: show image with range interactively
                 "RANGE": "None,None",
                 # TODO: support different resolution for x and y
                 "RESOLUTION": resolution[0],
                 "STRIDE": stride
                 }
        }
        json_str = json.dumps(json_dict, indent=4)

        QApplication.clipboard().setText(json_str)
        self.iface.messageBar().pushMessage(
            "Note:",
            "Settings have been copied to clipboard. "
            "You can paste it to Geo-SAM Image Encoder now.",
            level=Qgis.Info,
            duration=10
        )

    def show_extent(self):
        if self.check_raster_selected():
            self.canvas_extent.clear()
            extent = self.wdg_copilot.mExtentGroupBox.outputExtent()
            self.canvas_extent.add_extent(extent)

    def reset_to_project_crs(self):
        self.project.setCrs(self.crs_project)

    def reset_canvas(self):
        self.reset_to_project_crs()
        if hasattr(self, "canvas_extent"):
            self.canvas_extent.clear()

    def destruct(self):
        '''Destruct actions when closed widget'''
        self.reset_canvas()

    def unload(self):
        '''Unload actions when plugin is closed'''
        self.reset_canvas()
