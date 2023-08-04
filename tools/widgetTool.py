import os
import json
from typing import List, Tuple, Any, Dict
from pathlib import Path
from qgis.core import QgsProject, QgsCoordinateReferenceSystem, QgsMapLayerProxyModel
from qgis.gui import QgsMapToolPan, QgisInterface, QgsFileWidget
from qgis.core import QgsRasterLayer, QgsRectangle, QgsRasterBandStats
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QShortcut,
    QFileDialog,
    QDockWidget,
)
from PyQt5.QtGui import QKeySequence
from torchgeo.samplers import Units
from torchgeo.datasets import BoundingBox

from .geoTool import ImageCRSManager
from .SAMTool import SAM_Model
from .canvasTool import RectangleMapTool, ClickTool, Canvas_Points, Canvas_Rectangle, SAM_PolygonFeature, Canvas_Extent
from ..ui import UI_Selector, UI_EncoderCopilot
from .torchgeo_sam import SamTestGridGeoSampler, SamTestRasterDataset
from .messageTool import MessageTool

SAM_Model_Types_Full: List[str] = ["vit_h (huge)",
                                   "vit_l (large)",
                                   "vit_b (base)"]
SAM_Model_Types = [i.split(' ')[0].strip() for i in SAM_Model_Types_Full]


class Selector(QDockWidget):
    execute_SAM = pyqtSignal()

    def __init__(self, parent, iface: QgisInterface, cwd: str):
        # super().__init__()
        QDockWidget.__init__(self)
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
        self.hover_mode: bool = False
        self.t_area: float = 0.0
        self.t_area_default: float = 0.0

    def open_widget(self):
        '''Create widget selector'''
        self.parent.toolbar.setVisible(True)
        if self.dockFirstOpen:
            self._init_feature_related()
            self.load_demo_img()

            if self.receivers(self.execute_SAM) == 0:
                self.execute_SAM.connect(self.execute_segmentation)

            self.wdg_sel = UI_Selector
            ########## connect function to widget items ##########
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
                self.load_vector_file)

            self.wdg_sel.pushButton_load_feature.clicked.connect(
                self.load_feature)
            self.wdg_sel.radioButton_enable.setChecked(True)
            self.wdg_sel.radioButton_enable.toggled.connect(
                self.toggle_edit_mode)

            self.wdg_sel.radioButton_exe_hover.setChecked(False)
            self.wdg_sel.radioButton_exe_hover.toggled.connect(
                self.toggle_sam_hover_mode)

            self.wdg_sel.Box_min_area.valueChanged.connect(
                self.filter_feature_by_area)
            self.wdg_sel.Box_min_area_default.valueChanged.connect(
                self.load_default_t_area)
            self.wdg_sel.radioButton_show_extent.toggled.connect(
                self.show_hide_sam_feature_extent)

            ######### Setting default parameters for items #########
            # set filter for file dialog
            # self.wdg_sel.QgsFile_shapefile.setFilter("*.shp")
            # self.wdg_sel.QgsFile_shapefile.setStorageMode(
            #     QgsFileWidget.SaveFile)
            self.wdg_sel.MapLayerComboBox.setFilters(
                QgsMapLayerProxyModel.PolygonLayer
                | QgsMapLayerProxyModel.VectorLayer
            )
            self.wdg_sel.MapLayerComboBox.setAllowEmptyLayer(True)
            self.wdg_sel.MapLayerComboBox.setLayer(None)
            self.wdg_sel.MapLayerComboBox.layerChanged.connect(
                self.set_vector_layer)

            self.wdg_sel.QgsFile_feature.setStorageMode(
                QgsFileWidget.GetDirectory)

            # set button checkable
            self.wdg_sel.pushButton_fg.setCheckable(True)
            self.wdg_sel.pushButton_bg.setCheckable(True)
            self.wdg_sel.pushButton_rect.setCheckable(True)

            # set show extent checked
            self.wdg_sel.radioButton_show_extent.setChecked(True)

            # If a signal is connected to several slots,
            # the slots are activated in the same order in which the connections were made, when the signal is emitted.
            self.wdg_sel.closed.connect(self.destruct)
            self.wdg_sel.closed.connect(self.iface.actionPan().trigger)

            ########### shortcuts ############
            # create shortcuts
            self.shortcut_clear = QShortcut(
                QKeySequence(Qt.Key_C), self.wdg_sel)
            self.shortcut_undo = QShortcut(
                QKeySequence(Qt.Key_Z), self.wdg_sel)
            self.shortcut_save = QShortcut(
                QKeySequence(Qt.Key_S), self.wdg_sel)
            self.shortcut_hover_mode = QShortcut(
                QKeySequence(Qt.Key_H), self.wdg_sel)
            self.shortcut_tab = QShortcut(
                QKeySequence(Qt.Key_Tab), self.wdg_sel)
            self.shortcut_undo_sam_pg = QShortcut(
                QKeySequence(QKeySequence.Undo), self.wdg_sel)

            # connect shortcuts
            self.shortcut_clear.activated.connect(self.clear_layers)
            self.shortcut_undo.activated.connect(self.undo_last_prompt)
            self.shortcut_save.activated.connect(self.save_shp_file)
            self.shortcut_hover_mode.activated.connect(self.toggle_hover_mode)
            self.shortcut_tab.activated.connect(self.loop_prompt_type)
            self.shortcut_undo_sam_pg.activated.connect(self.undo_sam_polygon)

            # set context for shortcuts to application
            # this will make shortcuts work even if the widget is not focused
            self.shortcut_clear.setContext(Qt.ApplicationShortcut)
            self.shortcut_undo.setContext(Qt.ApplicationShortcut)
            self.shortcut_save.setContext(Qt.ApplicationShortcut)
            self.shortcut_hover_mode.setContext(Qt.ApplicationShortcut)
            self.shortcut_tab.setContext(Qt.ApplicationShortcut)
            self.shortcut_undo_sam_pg.setContext(Qt.ApplicationShortcut)

            ########## set dock ##########
            self.wdg_sel.setFloating(True)
            self.wdg_sel.setFocusPolicy(Qt.StrongFocus)

            # default is fgpt, but do not change when reloading feature folder
            # self.reset_prompt_type()
            self.dockFirstOpen = False
            # add widget to QGIS
            self.iface.addDockWidget(Qt.TopDockWidgetArea, self.wdg_sel)
        else:
            self.clear_layers(clear_extent=True)

        self.toggle_edit_mode()
        self.show_hide_sam_feature_extent()

        if not self.wdg_sel.isUserVisible():
            self.wdg_sel.setUserVisible(True)

    def destruct(self):
        '''Destruct actions when closed widget'''
        self.clear_layers(clear_extent=True)

        if hasattr(self, "shortcut_tab"):
            self.shortcut_tab.disconnect()
        if hasattr(self, "shortcut_undo_sam_pg"):
            self.shortcut_undo_sam_pg.disconnect()
        if hasattr(self, "shortcut_clear"):
            self.shortcut_clear.activated.disconnect()
        if hasattr(self, "shortcut_undo"):
            self.shortcut_undo.activated.disconnect()
        if hasattr(self, "shortcut_save"):
            self.shortcut_save.activated.disconnect()
        if hasattr(self, "shortcut_hover_mode"):
            self.shortcut_hover_mode.activated.disconnect()
        if hasattr(self, "wdg_sel"):
            self.wdg_sel.MapLayerComboBox.layerChanged.disconnect()

    def unload(self):
        '''Unload actions when plugin is closed'''
        self.destruct()
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
        MessageTool.MessageBar(
            "Great",
            f"SAM Features with {self.sam_model.feature_size} patches "
            f"in '{Path(self.feature_dir).name}' have been loaded, "
            "you can start labeling now"
        )

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

    def toggle_edit_mode(self):
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
                MessageTool.MessageBar(
                    "Oops",
                    "No sam feature loaded"
                )
        else:
            self.canvas_extent.clear()

    def toggle_hover_mode(self):
        '''Toggle move mode in widget selector. For shortcut only'''
        if self.wdg_sel.radioButton_exe_hover.isChecked():
            self.wdg_sel.radioButton_exe_hover.setChecked(False)
        else:
            self.wdg_sel.radioButton_exe_hover.setChecked(True)
        # toggle move mode in sam model
        self.toggle_sam_hover_mode()

    def toggle_sam_hover_mode(self):
        '''Toggle move mode in sam model'''
        if self.wdg_sel.radioButton_exe_hover.isChecked():
            self.hover_mode = True
            self.tool_click_fg.hover_mode = True
            self.tool_click_bg.hover_mode = True
            self.tool_click_rect.hover_mode = True
        else:
            self.hover_mode = False
            self.tool_click_fg.hover_mode = False
            self.tool_click_bg.hover_mode = False
            self.tool_click_rect.hover_mode = False
            # clear hover prompts
            self.tool_click_fg.clear_hover_prompt()
            self.tool_click_bg.clear_hover_prompt()
            self.tool_click_rect.clear_hover_prompt()

            self.execute_SAM.emit()

    def filter_feature_by_area(self):
        t_area = self.wdg_sel.Box_min_area.value()
        if not hasattr(self, "polygon"):
            return None

        self.polygon.canvas_polygon.clear()
        self.polygon.add_geojson_feature_to_canvas(
            self.polygon.geojson, t_area)
        self.t_area = t_area

    def load_default_t_area(self):
        self.t_area_default = self.wdg_sel.Box_min_area_default.value()
        self.wdg_sel.Box_min_area.setValue(self.t_area_default)

    def ensure_polygon_sam_exist(self):
        if hasattr(self, "polygon"):
            layer = QgsProject.instance().mapLayer(self.polygon.layer_id)
            if layer:
                return None
        self.set_vector_layer()

    def execute_segmentation(self) -> bool:
        # check last prompt inside feature extent
        if len(self.prompt_history) > 0 and not self.hover_mode:
            prompt_last = self.prompt_history[-1]
            if prompt_last == 'bbox':
                last_rect = self.canvas_rect.extent
                last_prompt = QgsRectangle(
                    last_rect[0], last_rect[2], last_rect[1], last_rect[3])
            else:
                last_point = self.canvas_points.points_img_crs[-1]
                last_prompt = QgsRectangle(last_point, last_point)
            if not last_prompt.intersects(self.sam_model.extent):
                self.message_box_outside()
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

        self.ensure_polygon_sam_exist()
        # execute segmentation
        if not self.sam_model.sam_predict(
                self.canvas_points,
                self.canvas_rect,
                self.polygon,
                self.prompt_history,
                self.hover_mode,
                self.t_area
        ):
            self.undo_last_prompt()
        self.topping_polygon_sam_layer()

        return True

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

    def set_vector_layer(self):
        '''set sam output vector layer'''

        new_layer = self.wdg_sel.MapLayerComboBox.currentLayer()

        # parse whether the new selected layer is same as current layer
        if hasattr(self, "polygon"):
            old_layer = QgsProject.instance().mapLayer(self.polygon.layer_id)
            if (old_layer and new_layer and
                    old_layer.id() == new_layer.id()):
                return None
            else:
                if not self.polygon.reset_layer(new_layer):
                    self.MapLayerComboBox.setLayer(None)
        else:
            self.polygon = SAM_PolygonFeature(
                self.img_crs_manager, layer=new_layer)

        # clear layer history
        self.sam_feature_history = []
        self.wdg_sel.MapLayerComboBox.setLayer(self.polygon.layer)

    def load_vector_file(self) -> None:
        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix('shp')
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_path, _ = file_dialog.getSaveFileName(
            None, "QFileDialog.getOpenFileName()",
            "",
            "Shapefile (*.shp)",
            options=QFileDialog.DontConfirmOverwrite
        )

        if file_path is None or file_path == '':
            return None

        file_path = Path(file_path)
        if not file_path.parent.is_dir():
            MessageTool.MessageBoxOK(
                "Oops: "
                "Failed to open file, please choose a existing folder"
            )
            return None
        else:
            if file_path.suffix.lower() != '.shp':
                file_path.with_suffix('.shp')

            layer_list = QgsProject.instance().mapLayersByName(file_path.stem)
            if len(layer_list) > 0:
                self.polygon = SAM_PolygonFeature(
                    self.img_crs_manager, layer=layer_list[0])
                if not hasattr(self.polygon, "layer"):
                    return None
                MessageTool.MessageBar(
                    "Attention",
                    f"Layer '{file_path.name}' has already been in the project, "
                    "you can start labeling now"
                )
                self.wdg_sel.MapLayerComboBox.setLayer(self.polygon.layer)

            else:
                self.polygon = SAM_PolygonFeature(
                    self.img_crs_manager, shapefile=file_path)
                if not hasattr(self.polygon, "layer"):
                    return None
            # clear layer history
            self.sam_feature_history = []
            self.wdg_sel.MapLayerComboBox.setLayer(self.polygon.layer)

    def load_feature(self):
        '''load feature'''
        self.feature_dir = self.wdg_sel.QgsFile_feature.filePath()
        if self.feature_dir is not None and os.path.exists(self.feature_dir):
            self.clear_layers(clear_extent=True)
            self._init_feature_related()
            self.toggle_edit_mode()
            self.show_hide_sam_feature_extent()
        else:
            MessageTool.MessageBar(
                'Oops',
                "Feature folder not exist, please choose a another folder"
            )

    def clear_layers(self, clear_extent: bool = False):
        '''Clear all temporary layers (canvas and new sam result) and reset prompt'''
        self.clear_canvas_layers_safely(clear_extent=clear_extent)
        if hasattr(self, "polygon"):
            # self.polygon.rollback_changes()
            self.polygon.canvas_polygon.clear()
        # if hasattr(self, "polygon_temp"):
        #     self.polygon_temp.rollback_changes()
        self.prompt_history.clear()

    def save_shp_file(self):
        '''save sam result into shapefile layer'''
        self.clear_canvas_layers_safely()
        self.prompt_history.clear()
        if hasattr(self, "polygon"):
            self.polygon.add_feature_to_layer(self.prompt_history, self.t_area)
            self.polygon.commit_changes()
            self.polygon.canvas_polygon.clear()

            # add last id of new features to history
            features = list(self.polygon.layer.getFeatures())
            if len(list(features)) == 0:
                return None
            last_id = features[-1].id()
            if self.sam_feature_history[-1][0] <= last_id:
                self.sam_feature_history[-1].append(last_id)
            self.wdg_sel.Box_min_area.setValue(self.t_area_default)

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
        if self.hover_mode:
            return True
        else:
            return MessageTool.MessageBoxOK('Point/rectangle is located outside of the feature boundary, click OK to undo last prompt.')


class EncoderCopilot(QDockWidget):
    # TODO: support encoding process in this widget
    def __init__(self, parent, iface: QgisInterface, cwd: str):
        QDockWidget.__init__(self)
        self.parent = parent
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.toolPan = QgsMapToolPan(self.canvas)
        self.dockFirstOpen: bool = True

        # init crs
        self.project: QgsProject = QgsProject.instance()
        self.crs_project: QgsCoordinateReferenceSystem = self.project.crs()
        self.crs_layer: QgsCoordinateReferenceSystem = self.crs_project
        # init raster layer
        self.raster_layer: QgsRasterLayer = None

    def open_widget(self):
        '''Create widget selector'''
        self.parent.toolbar.setVisible(True)
        if self.dockFirstOpen:
            self.wdg_copilot = UI_EncoderCopilot

            ########## connect functions to widget items ##########
            # upper part
            self.wdg_copilot.MapLayerComboBox.layerChanged.connect(
                self.parse_raster_info)
            self.wdg_copilot.ExtentGroupBox.setMapCanvas(self.canvas)
            self.wdg_copilot.ExtentGroupBox.extentChanged.connect(
                self.show_extents)
            self.wdg_copilot.BoxResolutionScale.valueChanged.connect(
                self.show_extents)
            self.wdg_copilot.BoxOverlap.valueChanged.connect(
                self.show_extents)
            self.wdg_copilot.pushButton_CopySetting.clicked.connect(
                self.json_setting_to_clipboard)
            self.wdg_copilot.pushButton_ExportSetting.clicked.connect(
                self.json_setting_to_file)

            # bottom part
            self.wdg_copilot.CheckpointFileWidget.fileChanged.connect(
                self.parse_model_type)

            # If a signal is connected to several slots,
            # the slots are activated in the same order in which the connections were made, when the signal is emitted.
            self.wdg_copilot.closed.connect(self.destruct)
            self.wdg_copilot.closed.connect(self.iface.actionPan().trigger)
            self.wdg_copilot.closed.connect(self.reset_to_project_crs)

            ########## set default values ##########
            # collapse group boxes
            self.wdg_copilot.AdvancedParameterGroupBox.setCollapsed(True)
            # checkpoint
            self.wdg_copilot.CheckpointFileWidget.setFilter("*.pth")
            self.wdg_copilot.CheckpointFileWidget.setStorageMode(
                QgsFileWidget.GetFile)
            self.wdg_copilot.CheckpointFileWidget.setConfirmOverwrite(False)

            # model types
            if self.wdg_copilot.SAMModelComboBox.count() == 0:
                self.wdg_copilot.SAMModelComboBox.addItems(
                    SAM_Model_Types_Full)

            self.dockFirstOpen = False
            # add widget to QGIS
            # self.iface.addDockWidget(Qt.BottomDockWidgetArea, self.wdg_copilot)
        else:
            pass

        if not self.wdg_copilot.isUserVisible():
            self.wdg_copilot.setUserVisible(True)

    def parse_raster_info(self):
        '''Parse raster info and set to widget items'''
        # set raster layer band to band field
        # TODO: support multi-threading
        if not self.valid_raster_layer():
            self.clear_bands()
            return None
        self.wdg_copilot.RasterBandComboBox_R.setLayer(self.raster_layer)
        self.wdg_copilot.RasterBandComboBox_G.setLayer(self.raster_layer)
        self.wdg_copilot.RasterBandComboBox_B.setLayer(self.raster_layer)

        # set crs to layer crs
        self.crs_project = self.project.crs()
        self.crs_layer = self.raster_layer.crs()

        if self.crs_layer != self.crs_project:
            self.project.setCrs(self.crs_layer)
            MessageTool.MessageBar(
                "Note:",
                "Project crs has been changed to the layer crs temporarily. "
                "It will be reset to the original crs when this widget is closed.",
                duration=30
            )

        # set data value range
        stats = self.raster_layer.dataProvider().bandStatistics(1, QgsRasterBandStats.All)
        self.wdg_copilot.MinValueBox.setValue(stats.minimumValue)
        self.wdg_copilot.MaxValueBox.setValue(stats.maximumValue)

        if not hasattr(self, "canvas_extent"):
            self.canvas_extent = Canvas_Extent(self.canvas, self.crs_layer)

    def clear_bands(self):
        self.wdg_copilot.RasterBandComboBox_R.setBand(-1)
        self.wdg_copilot.RasterBandComboBox_G.setBand(-1)
        self.wdg_copilot.RasterBandComboBox_B.setBand(-1)

    def valid_raster_layer(self) -> bool:
        '''Check if raster layer is valid. If not, alert user to select a valid raster layer'''
        layer = self.wdg_copilot.MapLayerComboBox.currentLayer()
        if isinstance(layer, QgsRasterLayer):
            self.raster_layer = self.wdg_copilot.MapLayerComboBox.currentLayer()
            return True
        else:
            MessageTool.MessageBoxOK(
                "Oops: Invalid Raster Layer. Please select a valid raster layer!"
            )
            self.raster_layer = None
            return False

    def get_bands(self) -> List[int]:
        bands = [self.wdg_copilot.RasterBandComboBox_R.currentBand(),
                 self.wdg_copilot.RasterBandComboBox_G.currentBand(),
                 self.wdg_copilot.RasterBandComboBox_B.currentBand()]
        return bands

    def get_resolutions(self) -> Tuple[float, float]:
        '''Get x, y resolution from resolution group box'''
        scale = self.wdg_copilot.BoxResolutionScale.value()
        resolution_layer = (
            self.raster_layer.rasterUnitsPerPixelX(),
            self.raster_layer.rasterUnitsPerPixelY()
        )
        resolution_scaled = (
            resolution_layer[0] * scale,
            resolution_layer[1] * scale
        )
        return resolution_scaled

    def get_stride(self) -> int:
        '''Get stride from overlap group box'''
        overlaps = self.wdg_copilot.BoxOverlap.value()
        stride = int((100 - overlaps)/100 * 1024)
        return stride

    def get_extent(self) -> QgsRectangle:
        extent = self.wdg_copilot.ExtentGroupBox.outputExtent()
        return extent

    def get_extent_str(self) -> str:
        '''Get extent string from extent group box'''
        extent = self.get_extent()
        xMin, yMin, xMax, yMax = (extent.xMinimum(),
                                  extent.yMinimum(),
                                  extent.xMaximum(),
                                  extent.yMaximum())
        return f'{xMin}, {xMax}, {yMin}, {yMax}'

    def get_checkpoint_path(self) -> str:
        '''Get checkpoint path from file widget'''
        checkpoint_path = self.wdg_copilot.CheckpointFileWidget.filePath()
        return checkpoint_path

    def parse_model_type(self) -> None:
        checkpoint_path = Path(self.get_checkpoint_path())
        for model_type in SAM_Model_Types:
            if model_type in checkpoint_path.name:
                self.wdg_copilot.SAMModelComboBox.setCurrentIndex(
                    SAM_Model_Types.index(model_type))
                break

    def get_model_type(self) -> int:
        '''Get SAM model type from combo box'''
        model_type = self.wdg_copilot.SAMModelComboBox.currentIndex()
        return model_type

    def get_max_value(self) -> float:
        return self.wdg_copilot.MaxValueBox.value()

    def get_min_value(self) -> float:
        return self.wdg_copilot.MinValueBox.value()

    def get_GPU_ID(self) -> int:
        return self.wdg_copilot.DeviceIDBox.value()

    def get_batch_size(self) -> int:
        return self.wdg_copilot.BatchSizeBox.value()

    def check_extent_set(self) -> bool:
        '''Check if extent has been set. If not, alert user to set extent'''
        extent = self.get_extent_str()
        vals = extent.split(',')
        if (float(vals[0]) == float(vals[1])
                or float(vals[2]) == float(vals[3])):
            # alert user to set extent
            MessageTool.MessageBoxOK(
                "Oops: Extent has not been set. Please set extent first."
            )
            return False
        else:
            return True

    def check_raster_selected(self) -> bool:
        '''Check if raster layer has been selected. If not, alert user to select a raster layer'''
        if self.raster_layer is not None:
            return True
        else:
            # alert user to select a raster layer
            MessageTool.MessageBoxOK(
                "Oops: "
                "Raster Layer has not been selected/detected. "
                "Please set/reset Raster Layer first!"
            )
            return False

    def check_setting_available(self) -> bool:
        '''Check if setting is available. If not, alert user to set setting'''
        if not self.check_raster_selected():
            return False
        elif not self.check_extent_set():
            return False
        else:
            return True

    def retrieve_setting(self) -> str:
        '''Retrieve setting from widget items'''
        if not self.check_setting_available():
            return None
        stride = self.get_stride()
        resolution = self.get_resolutions()
        bands = self.get_bands()
        crs = self.crs_layer.authid()
        extent = f'{self.get_extent_str()} [{crs}]'
        checkpoint_path = self.get_checkpoint_path()
        model_type = self.get_model_type()
        max_value = self.get_max_value()
        min_value = self.get_min_value()
        batch_size = self.get_batch_size()
        gpu_id = self.get_GPU_ID()

        json_dict = {
            "inputs":
                {"INPUT": self.raster_layer.source(),
                 "BANDS": bands,
                 # TODO: show image with range interactively
                 "RANGE": f"{min_value},{max_value}",
                 "CRS": crs,
                 "EXTENT": extent,
                 "RESOLUTION": resolution[0],
                 "STRIDE": stride,
                 "CKPT": checkpoint_path,
                 "MODEL_TYPE": model_type,
                 "BATCH_SIZE": batch_size,
                 "CUDA_ID": gpu_id,
                 }
        }
        json_str = json.dumps(json_dict, indent=4)
        return json_str

    def json_setting_to_clipboard(self) -> None:
        '''Copy setting to clipboard'''
        json_str = self.retrieve_setting()
        if json_str is None:
            return None

        QApplication.clipboard().setText(json_str)
        MessageTool.MessageBar(
            "Note:",
            "Setting has been copied to clipboard. "
            "You can paste it to Geo-SAM Image Encoder or a json file now.",
            duration=30
        )

    def json_setting_to_file(self) -> None:
        json_str = self.retrieve_setting()
        if json_str is None:
            return None

        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix('json')
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_path, _ = file_dialog.getSaveFileName(
            None, "QFileDialog.getOpenFileName()",
            "",
            "Json Files (*.json)")
        file_path = Path(file_path)
        try:
            if not file_path.parent.is_dir():
                MessageTool.MessageBoxOK(
                    "Oops: "
                    "Failed to save setting to file. "
                    "Please choose a valid directory first."
                )
                return None
            else:
                if file_path.suffix != '.json':
                    file_path.with_suffix('.json')

                with open(file_path, 'w') as f:
                    f.write(json_str)
        except Exception as e:
            MessageTool.MessageLog(
                f"Failed to save setting to file. Error: {e}",
                level='critical'
            )
            return None

    def show_extents(self):
        self.show_bbox_extent()
        self.show_batch_extent()

    def show_bbox_extent(self):
        '''Show bbox extent in canvas'''
        if hasattr(self, "canvas_extent"):
            self.canvas_extent.clear()
            extent = self.get_extent()
            self.canvas_extent.add_extent(extent)

    def show_batch_extent(self):
        '''Show all batch extents in canvas'''
        if not self.check_setting_available():
            return None

        input_bands = [self.raster_layer.bandName(i_band)
                       for i_band in self.get_bands()]

        raster_file = Path(self.raster_layer.source())

        SamTestRasterDataset.filename_glob = raster_file.name
        SamTestRasterDataset.all_bands = [
            self.raster_layer.bandName(i_band) for i_band in range(1, self.raster_layer.bandCount()+1)
        ]
        layer_ds = SamTestRasterDataset(
            root=str(raster_file.parent),
            crs=None,
            res=self.get_resolutions()[0],
            bands=input_bands,
            cache=False
        )
        extent = self.get_extent()
        extent_bbox = BoundingBox(
            minx=extent.xMinimum(),
            maxx=extent.xMaximum(),
            miny=extent.yMinimum(),
            maxy=extent.yMaximum(),
            mint=layer_ds.index.bounds[4],
            maxt=layer_ds.index.bounds[5]
        )

        ds_sampler = SamTestGridGeoSampler(
            layer_ds,
            size=1024,  # Currently, only 1024 considered (SAM default)
            stride=self.get_stride(),
            roi=extent_bbox,
            units=Units.PIXELS  # Units.CRS or Units.PIXELS
        )
        if len(ds_sampler) == 0:
            MessageTool.MessageBar(
                'Oops!!!',
                'No available patch sample inside the chosen extent!!! '
                'Please choose another extent.',
                duration=30
            )
            return None

        for i, patch in enumerate(ds_sampler):
            extent = QgsRectangle(
                patch['bbox'].minx,
                patch['bbox'].miny,
                patch['bbox'].maxx,
                patch['bbox'].maxy
            )
            alpha = ((i+1) / len(ds_sampler) * 255)
            if alpha == 255:
                print(alpha)
            self.canvas_extent.add_extent(
                extent,
                use_type='batch_extent',
                alpha=((i+1) / len(ds_sampler) * 255)
            )

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
