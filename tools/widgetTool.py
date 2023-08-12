import os
import json
from typing import List, Tuple, Any, Dict
from pathlib import Path
import rasterio
import numpy as np
from rasterio.windows import from_bounds as window_from_bounds
from qgis.core import QgsProject, QgsCoordinateReferenceSystem, QgsMapLayerProxyModel
from qgis.gui import QgsMapToolPan, QgisInterface, QgsFileWidget, QgsDoubleSpinBox
from qgis.core import QgsRasterLayer, QgsRectangle
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QApplication,
    QShortcut,
    QFileDialog,
    QDockWidget,
)
from PyQt5.QtGui import QKeySequence, QColor
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


class ParseRangeThread(QThread):
    def __init__(
            self,
            retrieve_range: pyqtSignal,
            raster_path: str,
            extent: List[float],
            bands: List[int],
    ):
        super().__init__()
        self.retrieve_range = retrieve_range
        self.raster_path = raster_path
        self.extent = extent
        self.bands = bands

    def run(self):
        with rasterio.open(self.raster_path) as src:
            # if image is too large, downsample it
            width = src.width
            height = src.height

            scale = width * height / 100000000
            if scale >= 2:
                width = int(width / scale)
                height = int(height / scale)
            if self.extent is None:
                window = None
            else:
                window = window_from_bounds(*self.extent, src.transform)

            arr = src.read(
                self.bands,
                out_shape=(len(self.bands), height, width),
                window=window
            )
            if src.meta['nodata'] is not None:
                arr = np.ma.masked_equal(arr, src.meta['nodata'])

        self.retrieve_range.emit(f"{np.nanmin(arr)}, {np.nanmax(arr)}")


class ShowBatchExtentThread(QThread):
    def __init__(self, retrieve_batch, ds_sampler):
        super().__init__()
        self.retrieve_batch = retrieve_batch
        self.ds_sampler = ds_sampler

    def run(self):
        extents = []
        for patch in self.ds_sampler:
            extents.append(
                [patch['bbox'].minx,
                 patch['bbox'].miny,
                 patch['bbox'].maxx,
                 patch['bbox'].maxy]
            )
        self.retrieve_batch.emit(f"{extents}")


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

            ######### Setting default parameters for items #########
            self.wdg_sel.MapLayerComboBox.setFilters(
                QgsMapLayerProxyModel.PolygonLayer
                | QgsMapLayerProxyModel.VectorLayer
            )
            self.wdg_sel.MapLayerComboBox.setAllowEmptyLayer(True)
            self.wdg_sel.MapLayerComboBox.setLayer(None)

            self.wdg_sel.QgsFile_feature.setStorageMode(
                QgsFileWidget.GetDirectory)

            # set button checkable
            self.wdg_sel.pushButton_fg.setCheckable(True)
            self.wdg_sel.pushButton_bg.setCheckable(True)
            self.wdg_sel.pushButton_rect.setCheckable(True)

            # toggle show extent
            self.wdg_sel.radioButton_show_extent.toggled.connect(
                self.toggle_encoding_extent)
            # set show extent checked
            self.wdg_sel.radioButton_show_extent.setChecked(True)

            # set default color
            self.wdg_sel.ColorButton_bgpt.setColor(Qt.red)
            self.wdg_sel.ColorButton_fgpt.setColor(Qt.blue)
            self.wdg_sel.ColorButton_bbox.setColor(Qt.blue)
            self.wdg_sel.ColorButton_extent.setColor(Qt.red)

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

            self.wdg_sel.MapLayerComboBox.layerChanged.connect(
                self.set_vector_layer)
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

            # threshold of area
            self.wdg_sel.Box_min_pixel.valueChanged.connect(
                self.filter_feature_by_area)
            self.wdg_sel.Box_min_pixel_default.valueChanged.connect(
                self.load_default_t_area)

            self.wdg_sel.ColorButton_bgpt.colorChanged.connect(
                self.reset_background_color)
            self.wdg_sel.ColorButton_fgpt.colorChanged.connect(
                self.reset_foreground_color)
            self.wdg_sel.ColorButton_bbox.colorChanged.connect(
                self.reset_rectangular_color)
            self.wdg_sel.ColorButton_extent.colorChanged.connect(
                self.reset_extent_color)

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
                QKeySequence(Qt.Key_P), self.wdg_sel)
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
        self.toggle_encoding_extent()

        if not self.wdg_sel.isUserVisible():
            self.wdg_sel.setUserVisible(True)

    def disconnect_safely(self, item):
        try:
            item.disconnect()
        except:
            pass

    def destruct(self):
        '''Destruct actions when closed widget'''
        self.clear_layers(clear_extent=True)
        self.iface.actionPan().trigger()

        # set context for shortcuts to application
        # this will make shortcuts work even if the widget is not focused
        # self.shortcut_clear.setContext(Qt.ApplicationShortcut)
        # self.shortcut_undo.setContext(Qt.ApplicationShortcut)
        # self.shortcut_save.setContext(Qt.ApplicationShortcut)
        # self.shortcut_hover_mode.setContext(Qt.ApplicationShortcut)
        # self.shortcut_tab.setContext(Qt.ApplicationShortcut)
        # self.shortcut_undo_sam_pg.setContext(Qt.ApplicationShortcut)
        self.disconnect_safely(self.shortcut_tab)
        self.disconnect_safely(self.shortcut_undo_sam_pg)
        self.disconnect_safely(self.shortcut_clear)
        self.disconnect_safely(self.shortcut_undo)
        self.disconnect_safely(self.shortcut_save)
        self.disconnect_safely(self.shortcut_hover_mode)
        self.disconnect_safely(self.wdg_sel.MapLayerComboBox.layerChanged)

    def unload(self):
        '''Unload actions when plugin is closed'''
        self.clear_layers(clear_extent=True)
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
        if hasattr(self, "polygon"):
            self.polygon.rollback_changes()

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
        self.res = float(
            (self.sam_model.test_features.index_df.loc[:, 'res']/16).mean())
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

    def toggle_encoding_extent(self):
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
        '''Toggle move mode in widget selector.'''
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

    def is_pressed_prompt(self):
        '''Check if the prompt is clicked or hovered'''
        if (self.tool_click_fg.pressed or
            self.tool_click_bg.pressed or
                self.tool_click_rect.pressed):
            self.tool_click_rect.pressed = False
            return True
        return False

    def filter_feature_by_area(self):
        t_area = self.wdg_sel.Box_min_pixel.value() * self.res ** 2
        if not hasattr(self, "polygon"):
            return None

        if self.hover_mode:
            self.polygon.canvas_polygon.clear()
            self.polygon.add_geojson_feature_to_canvas(
                self.polygon.geojson_layer,  # only need to use geojson_layer
                t_area
            )

        # layer refresh for all mode
        self.polygon.rollback_changes()
        self.polygon.add_geojson_feature_to_layer(
            self.polygon.geojson_layer,
            t_area,
            self.prompt_history
        )
        self.t_area = t_area

    def load_default_t_area(self):
        self.t_area_default = self.wdg_sel.Box_min_pixel_default.value() * self.res ** 2
        self.wdg_sel.Box_min_pixel.setValue(self.t_area_default)

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
                MessageTool.MessageLog(
                    f"canvas_points.points_img_crs: {self.canvas_points.points_img_crs}")
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

        # show pressed prompt result in hover mode
        if self.hover_mode and self.is_pressed_prompt():
            self.polygon.rollback_changes()
            self.polygon.add_geojson_feature_to_layer(
                self.polygon.geojson_canvas,  # update with canvas polygon
                self.t_area,
                self.prompt_history,
                overwrite_geojson=True
            )
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
            self.toggle_enpcoding_extent()
        else:
            MessageTool.MessageBar(
                'Oops',
                "Feature folder not exist, please choose a another folder"
            )

    def clear_layers(self, clear_extent: bool = False):
        '''Clear all temporary layers (canvas and new sam result) and reset prompt'''
        self.clear_canvas_layers_safely(clear_extent=clear_extent)
        if hasattr(self, "polygon"):
            self.polygon.rollback_changes()
            self.polygon.canvas_polygon.clear()
        self.prompt_history.clear()

    def save_shp_file(self):
        '''save sam result into shapefile layer'''

        need_toggle = False
        if self.hover_mode:
            need_toggle = True
            self.toggle_hover_mode()
            if len(self.prompt_history) == 0:
                MessageTool.MessageBoxOK(
                    "Preview mode only shows the preview of prompts. Click first to apply the prompt."
                )
                self.toggle_hover_mode()
                return False

        if hasattr(self, "polygon"):
            self.polygon.add_geojson_feature_to_layer(
                self.polygon.geojson_layer,
                self.t_area,
                self.prompt_history
            )
            self.polygon.commit_changes()
            self.polygon.canvas_polygon.clear()

            # add last id of new features to history
            features = list(self.polygon.layer.getFeatures())
            if len(list(features)) == 0:
                return None
            last_id = features[-1].id()
            MessageTool.MessageLog(
                f"sam_feature_history: {self.sam_feature_history}", 'critical')
            if len(self.sam_feature_history) > 0:
                if self.sam_feature_history[-1][0] <= last_id:
                    self.sam_feature_history[-1].append(last_id)

        self.clear_canvas_layers_safely()
        self.prompt_history.clear()
        self.wdg_sel.Box_min_pixel.setValue(self.t_area_default)
        # reenable hover mode
        if need_toggle:
            self.toggle_hover_mode()

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

    def reset_background_color(self):
        '''Reset background color'''
        self.canvas_points.background_color = self.wdg_sel.ColorButton_bgpt.color()
        self.canvas_points.flush_points_color()
        self.tool_click_bg.reset_cursor_color(
            self.wdg_sel.ColorButton_bgpt.color().name()
        )

    def reset_foreground_color(self):
        '''Reset foreground color'''
        self.canvas_points.foreground_color = self.wdg_sel.ColorButton_fgpt.color()
        self.canvas_points.flush_points_color()
        self.tool_click_fg.reset_cursor_color(
            self.wdg_sel.ColorButton_fgpt.color().name()
        )

    def reset_rectangular_color(self):
        '''Reset rectangular color'''
        color = self.wdg_sel.ColorButton_bbox.color()
        color_fill = list(color.getRgb())
        color_fill[-1] = 10
        color_fill = QColor(*color_fill)
        self.canvas_rect.set_line_color(color)
        self.canvas_rect.set_fill_color(color_fill)

        self.tool_click_rect.reset_cursor_color(
            self.wdg_sel.ColorButton_bbox.color().name()
        )

    def reset_extent_color(self):
        '''Reset extent color'''
        self.canvas_extent.set_color(
            self.wdg_sel.ColorButton_extent.color()
        )

        if self.wdg_sel.radioButton_show_extent.isChecked():
            self.canvas_extent.clear()
            if hasattr(self, "sam_extent_canvas_crs"):
                self.canvas_extent.add_extent(self.sam_extent_canvas_crs)


class EncoderCopilot(QDockWidget):
    # TODO: support encoding process in this widget
    retrieve_range = pyqtSignal(str)
    retrieve_batch = pyqtSignal(str)

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
            self.wdg_copilot.MapLayerComboBox.setFilters(
                QgsMapLayerProxyModel.RasterLayer)
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
            self.wdg_copilot.pushButton_parse_raster.clicked.connect(
                self.parse_min_max_value)

            # bottom part
            self.wdg_copilot.CheckpointFileWidget.fileChanged.connect(
                self.parse_model_type)

            # signal
            self.retrieve_range.connect(self.set_range_to_widget)
            self.retrieve_batch.connect(self.show_batch_extent_in_canvas)

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

    def set_range_to_widget(self, range: str):
        '''Set range to widget'''
        range = eval(range)
        self.wdg_copilot.MinValueBox.setValue(range[0])
        self.wdg_copilot.MaxValueBox.setValue(range[1])
        self.wdg_copilot.label_range_status.setText("Done!")

    def show_batch_extent_in_canvas(self, extents: str):
        extents = eval(extents)
        num_batch = len(extents)
        idx = np.random.randint(0, num_batch, size=(int(num_batch/10)))
        alphas = np.full(num_batch, 100)
        line_widths = np.full(num_batch, 2)
        alphas[idx] = 254
        line_widths[idx] = 5

        for i, extent in enumerate(extents):
            if i == num_batch - 1:
                alpha = 255
                line_width = 5
            else:
                alpha = alphas[i]

            line_width = line_widths[i]

            self.canvas_extent.add_extent(
                QgsRectangle(*extent),
                use_type='batch_extent',
                alpha=alpha,
                line_width=line_width
            )
        self.wdg_copilot.label_batch_settings.setText(
            f"Done! {len(extents)} batch")

    def parse_raster_info(self):
        '''Parse raster info and set to widget items'''
        # clear widget items
        self.wdg_copilot.label_range_status.setText("")
        self.wdg_copilot.label_batch_settings.setText("")
        self.wdg_copilot.MaxValueBox.setClearValue(
            0, "Not set")
        self.wdg_copilot.MinValueBox.setClearValue(
            0, "Not set")
        self.wdg_copilot.MinValueBox.clear()
        self.wdg_copilot.MaxValueBox.clear()
        self.wdg_copilot.BoxResolutionScale.setClearValue(
            1, "Not set")
        self.wdg_copilot.BoxOverlap.setClearValue(
            50, "Not set")

        if not self.valid_raster_layer():
            self.clear_bands()
            return None

        # set raster layer band to band field
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

        if not hasattr(self, "canvas_extent"):
            self.canvas_extent = Canvas_Extent(self.canvas, self.crs_layer)

    def parse_min_max_value(self):
        '''Parse min and max value from raster layer'''
        if not self.valid_raster_layer():
            return None
        extent = self.get_extent()
        if (extent.xMinimum() == extent.xMaximum() or
                extent.yMinimum() == extent.yMaximum()):
            extent = None
        else:
            extent = [extent.xMinimum(),
                      extent.yMinimum(),
                      extent.xMaximum(),
                      extent.yMaximum()]

        bands = list(set(self.get_bands()))

        self.wdg_copilot.label_range_status.setText("Parse ...")
        self.ParseThread = ParseRangeThread(
            self.retrieve_range,
            self.raster_layer.source(),
            extent,
            bands
        )
        self.ParseThread.start()

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

    def get_range_value(self) -> float:
        max_value = self.wdg_copilot.MaxValueBox.value()
        min_value = self.wdg_copilot.MinValueBox.value()
        if max_value == 0 and min_value == 0:
            return None
        return min_value, max_value

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
        range = self.get_range_value()
        batch_size = self.get_batch_size()
        gpu_id = self.get_GPU_ID()

        json_dict = {
            "inputs":
                {"INPUT": self.raster_layer.source(),
                 "BANDS": bands,
                 # TODO: show image with range interactively
                 "RANGE": range,
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

        self.ds_sampler = SamTestGridGeoSampler(
            layer_ds,
            size=1024,  # Currently, only 1024 considered (SAM default)
            stride=self.get_stride(),
            roi=extent_bbox,
            units=Units.PIXELS  # Units.CRS or Units.PIXELS
        )
        if len(self.ds_sampler) == 0:
            MessageTool.MessageBar(
                'Oops!!!',
                'No available patch sample inside the chosen extent!!! '
                'Please choose another extent.',
                duration=30
            )
            return None

        self.wdg_copilot.label_batch_settings.setText("Computing ...")
        self.show_batch_extent_thread = ShowBatchExtentThread(
            self.retrieve_batch, self.ds_sampler)
        self.show_batch_extent_thread.start()

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
