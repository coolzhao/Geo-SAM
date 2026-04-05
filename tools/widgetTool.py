import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QShortcut,
)
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsMapLayerProxyModel,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
)
from qgis.gui import QgisInterface, QgsFileWidget, QgsMapToolPan
from rasterio.windows import from_bounds as window_from_bounds

from ..ui import (
    DefaultSettings,
    Settings,
    UI_EncoderCopilot,
    UI_Selector,
    save_user_settings,
)
from .canvasTool import (
    Canvas_Extent,
    Canvas_Points,
    Canvas_Rectangle,
    ClickTool,
    RectangleMapTool,
    SAM_PolygonFeature,
)
from .geosam_runtime import (
    chip_extent_rectangles_for_source,
    describe_feature_source,
    get_model_checkpoint_path,
    get_model_display_items,
    infer_model_id_from_checkpoint_path,
    layer_extent_rectangle,
    layer_pixel_area,
    load_plugin_settings,
    query_feature_source,
    query_raster_layer,
    query_result_to_geojson_features,
    save_plugin_settings,
)
from .geoTool import ImageCRSManager
from .messageTool import MessageTool


class ParseRangeThread(QThread):
    def __init__(
            self,
            retrieve_range: pyqtSignal,
            raster_path: str,
            extent: list[float],
            bands: list[int],
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
            if src.meta["nodata"] is not None:
                arr = np.ma.masked_equal(arr, src.meta["nodata"])

        self.retrieve_range.emit(f"{np.nanmin(arr)}, {np.nanmax(arr)}")


class ShowPatchExtentThread(QThread):
    def __init__(self, retrieve_patch, ds_sampler):
        super().__init__()
        self.retrieve_patch = retrieve_patch
        self.ds_sampler = ds_sampler

    def run(self):
        extents = []
        for patch in self.ds_sampler:
            bbox = patch["bbox"]
            if isinstance(bbox, QgsRectangle):
                extents.append(
                    [
                        bbox.xMinimum(),
                        bbox.yMinimum(),
                        bbox.xMaximum(),
                        bbox.yMaximum(),
                    ]
                )
                continue
            extents.append(
                [bbox.minx,
                 bbox.miny,
                 bbox.maxx,
                 bbox.maxy]
            )
        self.retrieve_patch.emit(f"{extents}")


class Selector(QDockWidget):
    execute_SAM = pyqtSignal()

    def __init__(self, parent, iface: QgisInterface, cwd: str, load_demo: bool = True):
        # super().__init__()
        QDockWidget.__init__(self)
        self.parent = parent
        self.iface = iface
        self.cwd = Path(cwd)
        self.canvas = iface.mapCanvas()
        self.demo_img_name = "beiluhe_google_img_201211_clip"
        self.feature_dir = str(self.cwd / "features" / self.demo_img_name)
        self.load_demo = load_demo
        self.project: QgsProject = QgsProject.instance()
        self.toolPan = QgsMapToolPan(self.canvas)
        self.dockFirstOpen = True
        self.prompt_history: list[str] = []
        self.sam_feature_history: list[list[int]] = []
        self.preview_mode: bool = False
        self.pixel_area: float = 1.0
        self.t_area: float = 0.0
        self.t_area_default: float = 0.0
        self.need_execute_sam_toggle_mode: bool = True
        self.need_execute_sam_filter_area: bool = True
        self.runtime_source_kind: str | None = None
        self.runtime_feature_summary = None
        self.runtime_layer: QgsRasterLayer | None = None
        self.runtime_extent: QgsRectangle | None = None
        self.runtime_crs: QgsCoordinateReferenceSystem | None = None
        # colors
        self.style_preview_polygon: dict[str, Any] = {
            "line_color": Settings["preview_color"],
            "fill_color": self.alpha_color(Settings["preview_color"], 10),
            "line_width": 2,
        }
        self.style_prompt_polygon: dict[str, Any] = {
            "line_color": Settings["prompt_color"],
            "fill_color": self.alpha_color(Settings["prompt_color"], 10),
            "line_width": 3,
        }

    def open_widget(self):
        """Create widget selector"""
        self.parent.toolbar.setVisible(True)
        if self.dockFirstOpen:
            self.crs_project: QgsCoordinateReferenceSystem = self.project.crs()

            if self.receivers(self.execute_SAM) == 0:
                self.execute_SAM.connect(self.execute_segmentation)

            self.wdg_sel = UI_Selector
            self._setup_runtime_widgets()

            ######### Setting default parameters for items #########
            self.wdg_sel.MapLayerComboBox.setFilters(
                QgsMapLayerProxyModel.PolygonLayer
                | QgsMapLayerProxyModel.VectorLayer
            )
            self.wdg_sel.MapLayerComboBox.setAllowEmptyLayer(True)
            self.wdg_sel.MapLayerComboBox.setAdditionalLayers([None])

            self.wdg_sel.QgsFile_feature.setStorageMode(
                QgsFileWidget.GetDirectory)
            self.wdg_sel.QgsFile_feature.setFilePath("")

            # set button checkable
            self.wdg_sel.pushButton_fg.setCheckable(True)
            self.wdg_sel.pushButton_bg.setCheckable(True)
            self.wdg_sel.pushButton_rect.setCheckable(True)

            self.wdg_sel.pushButton_reset_settings.clicked.connect(
                self.reset_default_settings)

            # toggle show extent
            self.wdg_sel.radioButton_show_extent.toggled.connect(
                self.toggle_encoding_extent)

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
            self.wdg_sel.RealTimeLayerComboBox.layerChanged.connect(
                self.on_realtime_layer_changed)
            self.wdg_sel.ModelComboBox.currentIndexChanged.connect(
                self.on_model_changed)
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
            self.wdg_sel.ColorButton_prompt.colorChanged.connect(
                self.reset_prompt_polygon_color)
            self.wdg_sel.ColorButton_preview.colorChanged.connect(
                self.reset_preview_polygon_color)
            self.wdg_sel.radioButton_load_demo.toggled.connect(
                self.toggle_load_demo_img)

            # Slots fire in the same connection order when a signal is emitted.
            self.wdg_sel.closed.connect(self.destruct)
            self.wdg_sel.closed.connect(self.iface.actionPan().trigger)
            self.wdg_sel.closed.connect(self.reset_to_project_crs)

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

            ########## set default Settings ##########
            self.set_user_settings()
            self.refresh_runtime_controls()

            ########## set dock ##########
            self.wdg_sel.setFloating(True)
            self.wdg_sel.setFocusPolicy(Qt.StrongFocus)

            # default is fgpt, but do not change when reloading feature folder
            # self.reset_prompt_type()
            self.dockFirstOpen = False
        else:
            self.clear_layers(clear_extent=True)

        # add widget to QGIS
        self.iface.addDockWidget(Qt.TopDockWidgetArea, self.wdg_sel)

        self.toggle_edit_mode()
        self.toggle_encoding_extent()
        self._ensure_feature_crs()

        # if not self.wdg_sel.isUserVisible():
        #     self.wdg_sel.setUserVisible(True)

    def _setup_runtime_widgets(self):
        """Create model and realtime-layer controls on the selector widget."""
        if hasattr(self.wdg_sel, "ModelComboBox"):
            return

        prompt_layout = self.wdg_sel.Top.layout()
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout(model_group)
        model_layout.setContentsMargins(9, 9, 9, 9)
        model_combo = QComboBox(model_group)
        model_combo.addItem("", "")
        for model_id, label in get_model_display_items():
            model_combo.addItem(label, model_id)
        model_layout.addWidget(model_combo)
        self.wdg_sel.ModelComboBox = model_combo
        prompt_layout.insertWidget(0, model_group)

        input_layout = self.wdg_sel.FeatureFolderGroupBox.layout()
        realtime_group = QGroupBox("RealTime Layer")
        realtime_layout = QHBoxLayout(realtime_group)
        realtime_layout.setContentsMargins(9, 9, 9, 9)
        self.wdg_sel.RealTimeLayerComboBox = self._build_realtime_layer_combo()
        realtime_layout.addWidget(self.wdg_sel.RealTimeLayerComboBox)
        input_layout.insertWidget(0, realtime_group)

        self.wdg_sel.FeatureFolderGroupBox.setTitle("Input")
        self.wdg_sel.OutShapefileGroupBox.setTitle("Output Feature")
        self.wdg_sel.groupBox_14.hide()
        self.wdg_sel.groupBox_11.setTitle("Minimum Pixels")
        self.wdg_sel.groupBox_15.hide()
        self.wdg_sel.radioButton_load_demo.hide()

        middle_splitter = self.wdg_sel.splitter
        middle_splitter.insertWidget(0, self.wdg_sel.FeatureFolderGroupBox)
        middle_splitter.insertWidget(1, self.wdg_sel.OutShapefileGroupBox)
        return

    def _build_realtime_layer_combo(self):
        if hasattr(self.wdg_sel, "RealTimeLayerComboBox"):
            return self.wdg_sel.RealTimeLayerComboBox
        from qgis.gui import QgsMapLayerComboBox

        combo = QgsMapLayerComboBox(self.wdg_sel)
        combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        combo.setAllowEmptyLayer(True)
        combo.setAdditionalLayers([None])
        return combo

    def refresh_runtime_controls(self):
        """Refresh model and source selectors from persisted settings."""
        settings = load_plugin_settings()
        model_index = self.wdg_sel.ModelComboBox.findData(
            settings.get("selected_model_id", "")
        )
        if model_index >= 0:
            self.wdg_sel.ModelComboBox.setCurrentIndex(model_index)
        else:
            self.wdg_sel.ModelComboBox.setCurrentIndex(0)

    def on_model_changed(self):
        """Persist the selected model and validate checkpoint availability."""
        model_id = self.selected_model_id()
        save_plugin_settings({"selected_model_id": model_id or ""})
        if model_id is None:
            return
        checkpoint_path = get_model_checkpoint_path(model_id)
        if checkpoint_path.exists():
            return
        MessageTool.MessageBoxOK(
            "The selected model is not downloaded yet.\n"
            "1. Open Geo-SAM Settings and download the checkpoint.\n"
            "2. Documentation: https://geo-sam.readthedocs.io/en/latest/",
            title="Model Not Downloaded",
        )

    def on_realtime_layer_changed(self):
        """React to realtime layer changes and rebuild the runtime context."""
        if self.wdg_sel.RealTimeLayerComboBox.currentLayer() is None:
            if self.wdg_sel.QgsFile_feature.filePath().strip():
                self.load_feature()
            return
        self.clear_layers(clear_extent=True)
        self._activate_runtime_from_inputs()

    def alpha_color(self, color: QColor, alpha: float) -> QColor:
        return QColor(color.red(), color.green(), color.blue(), alpha)

    def set_user_settings_color(self):
        self.wdg_sel.ColorButton_bgpt.setColor(Settings["bg_color"])
        self.wdg_sel.ColorButton_fgpt.setColor(Settings["fg_color"])
        self.wdg_sel.ColorButton_bbox.setColor(Settings["bbox_color"])
        self.wdg_sel.ColorButton_extent.setColor(Settings["extent_color"])
        self.wdg_sel.ColorButton_prompt.setColor(Settings["prompt_color"])
        self.wdg_sel.ColorButton_preview.setColor(Settings["preview_color"])

    def set_user_settings(self):
        if Settings["show_boundary"]:
            self.wdg_sel.radioButton_show_extent.setChecked(True)

        self.set_user_settings_color()
        self.wdg_sel.Box_min_pixel_default.setValue(
            DefaultSettings["default_minimum_pixels"]
        )

    def reset_default_settings(self):
        save_user_settings({}, mode="overwrite")
        save_plugin_settings({
            "selected_model_id": "",
            "model_repo_url": str(load_plugin_settings()["model_repo_url"]),
            "model_store_dir": str(load_plugin_settings()["model_store_dir"]),
            "cache_enabled": bool(load_plugin_settings()["cache_enabled"]),
            "cache_dir": str(load_plugin_settings()["cache_dir"]),
            "cache_max_size_mb": int(load_plugin_settings()["cache_max_size_mb"]),
        })

        if DefaultSettings["show_boundary"]:
            self.wdg_sel.radioButton_show_extent.setChecked(True)

        self.wdg_sel.Box_min_pixel_default.setValue(
            DefaultSettings["default_minimum_pixels"])

        # set default color
        self.wdg_sel.ColorButton_bgpt.setColor(DefaultSettings["bg_color"])
        self.wdg_sel.ColorButton_fgpt.setColor(DefaultSettings["fg_color"])
        self.wdg_sel.ColorButton_bbox.setColor(DefaultSettings["bbox_color"])
        self.wdg_sel.ColorButton_extent.setColor(
            DefaultSettings["extent_color"])
        self.wdg_sel.ColorButton_prompt.setColor(
            DefaultSettings["prompt_color"])
        self.wdg_sel.ColorButton_preview.setColor(
            DefaultSettings["preview_color"])

    def disconnect_safely(self, item):
        try:
            item.disconnect()
        except (TypeError, RuntimeError):
            pass

    def reset_to_project_crs(self):
        if self.crs_project != self.project.crs():
            MessageTool.MessageBar(
                "Note:",
                "Project CRS has been reset to original CRS."
            )
            self.project.setCrs(self.crs_project)

    def destruct(self):
        """Destruct actions when closed widget"""
        self.clear_layers(clear_extent=True)
        self.reset_to_project_crs()
        self.iface.actionPan().trigger()

    def unload(self):
        """Unload actions when plugin is closed"""
        self.clear_layers(clear_extent=True)
        if hasattr(self, "shortcut_tab"):
            self.disconnect_safely(self.shortcut_tab)
        if hasattr(self, "shortcut_undo_sam_pg"):
            self.disconnect_safely(self.shortcut_undo_sam_pg)
        if hasattr(self, "shortcut_clear"):
            self.disconnect_safely(self.shortcut_clear)
        if hasattr(self, "shortcut_undo"):
            self.disconnect_safely(self.shortcut_undo)
        if hasattr(self, "shortcut_save"):
            self.disconnect_safely(self.shortcut_save)
        if hasattr(self, "shortcut_hover_mode"):
            self.disconnect_safely(self.shortcut_hover_mode)
        if hasattr(self, "wdg_sel"):
            self.disconnect_safely(self.wdg_sel.MapLayerComboBox.layerChanged)
            self.iface.removeDockWidget(self.wdg_sel)
        self.destruct()

    def load_demo_img(self):
        return None

    def topping_polygon_sam_layer(self):
        """Topping polygon layer of SAM result to top of TOC"""
        root = QgsProject.instance().layerTreeRoot()
        tree_layer = root.findLayer(self.polygon.layer.id())

        if tree_layer is None:
            return
        if not tree_layer.isVisible():
            tree_layer.setItemVisibilityChecked(True)
        if root.children()[0] == tree_layer:
            return

        # move to top
        tl_clone = tree_layer.clone()
        root.insertChildNode(0, tl_clone)
        parent_tree_layer = tree_layer.parent()
        parent_tree_layer.removeChildNode(tree_layer)

    def clear_canvas_layers_safely(self, clear_extent: bool = False):
        """Clear canvas layers safely"""
        self.canvas.refresh()
        if hasattr(self, "canvas_points"):
            self.canvas_points.clear()
        if hasattr(self, "canvas_rect"):
            self.canvas_rect.clear()
        if hasattr(self, "canvas_extent") and clear_extent:
            self.canvas_extent.clear()
        if hasattr(self, "polygon"):
            self.polygon.clear_canvas_polygons()
        self.canvas.refresh()

    def _ensure_feature_crs(self):
        if self.runtime_crs is None:
            return
        if self.runtime_crs != self.crs_project:
            self.project.setCrs(self.runtime_crs)
            MessageTool.MessageBar(
                "Note:",
                "Project CRS has been changed to the active source CRS temporarily, "
                "and will be reset to original CRS when this widget is closed.",
                duration=30
            )

    def _activate_runtime_from_inputs(self):
        """Load the current runtime source from the selector inputs."""
        realtime_layer = self.wdg_sel.RealTimeLayerComboBox.currentLayer()
        feature_path = self.wdg_sel.QgsFile_feature.filePath().strip()
        if realtime_layer is not None and feature_path:
            MessageTool.MessageBar(
                "Geo-SAM",
                "RealTime Layer takes precedence. Clear it to use the feature folder.",
                level="warning",
                duration=10,
            )
        if realtime_layer is not None:
            return self._set_runtime_for_layer(realtime_layer)
        if feature_path:
            return self._set_runtime_for_feature_folder(feature_path)
        return None

    def _set_runtime_context(
        self,
        *,
        source_kind: str,
        crs_text: str,
        extent: QgsRectangle,
        pixel_area: float,
    ):
        """Initialize shared canvas tools for the active runtime source."""
        self.runtime_source_kind = source_kind
        self.runtime_extent = extent
        self.runtime_crs = QgsCoordinateReferenceSystem(crs_text)
        if not self.runtime_crs.isValid():
            msg = (
                f"Failed to resolve a valid CRS from the active {source_kind} "
                f"source: {crs_text!r}"
            )
            raise ValueError(msg)
        self.pixel_area = pixel_area if pixel_area > 0 else 1.0
        self.img_crs_manager = ImageCRSManager(crs_text)
        self.canvas_points = Canvas_Points(self.canvas, self.img_crs_manager)
        self.canvas_rect = Canvas_Rectangle(self.canvas, self.img_crs_manager)
        self.canvas_extent = Canvas_Extent(self.canvas, self.img_crs_manager)

        self.tool_click_fg = ClickTool(
            self.canvas,
            self.canvas_points,
            "fgpt",
            self.prompt_history,
            self.execute_SAM,
        )
        self.tool_click_bg = ClickTool(
            self.canvas,
            self.canvas_points,
            "bgpt",
            self.prompt_history,
            self.execute_SAM,
        )
        self.tool_click_rect = RectangleMapTool(
            self.canvas_rect,
            self.prompt_history,
            self.execute_SAM,
            self.img_crs_manager
        )

        self._ensure_feature_crs()
        self.source_extent_canvas_crs = self.img_crs_manager.img_extent_to_crs(
            extent,
            QgsProject.instance().crs(),
        )
        self.canvas.setExtent(self.source_extent_canvas_crs)
        self.canvas.refresh()
        self.load_default_t_area()
        self.toggle_encoding_extent()

    def _set_runtime_for_feature_folder(self, feature_dir: str):
        """Load feature-folder runtime metadata and canvas tools."""
        summary = describe_feature_source(feature_dir)
        self.feature_dir = feature_dir
        self.runtime_feature_summary = summary
        self.runtime_layer = None
        extent = QgsRectangle(*summary.extent)
        self._set_runtime_context(
            source_kind="feature",
            crs_text=summary.crs_text,
            extent=extent,
            pixel_area=summary.pixel_area,
        )
        MessageTool.MessageBar(
            "Geo-SAM",
            f"Loaded feature folder '{Path(feature_dir).name}' with "
            f"{summary.chip_count} cached chips.",
            level="success",
        )
        return self.runtime_source_kind

    def _set_runtime_for_layer(self, layer: QgsRasterLayer):
        """Load realtime raster-layer runtime metadata and canvas tools."""
        self.runtime_layer = layer
        self.runtime_feature_summary = None
        self._set_runtime_context(
            source_kind="realtime",
            crs_text=layer.crs().authid() or layer.crs().toWkt(),
            extent=layer_extent_rectangle(layer),
            pixel_area=layer_pixel_area(layer),
        )
        MessageTool.MessageBar(
            "Geo-SAM",
            f"Loaded realtime layer '{layer.name()}'.",
            level="success",
        )
        return self.runtime_source_kind

    def _require_runtime_source(self) -> bool:
        """Ensure one input source is available before prompting."""
        if hasattr(self, "tool_click_fg") and self.runtime_source_kind is not None:
            return True
        if self._activate_runtime_from_inputs() is not None:
            return True
        MessageTool.MessageBoxOK(
            "Please choose a RealTime Layer or a Feature folder first.",
            title="Input Required",
        )
        return False

    def selected_model_id(self) -> str | None:
        """Return the current model identifier or None."""
        model_id = self.wdg_sel.ModelComboBox.currentData()
        if model_id in (None, ""):
            return None
        return str(model_id)

    def _require_selected_model(self) -> str | None:
        """Ensure a valid downloaded model is selected."""
        model_id = self.selected_model_id()
        if model_id is None:
            MessageTool.MessageBoxOK(
                "Please choose a SAM model first.",
                title="Model Required",
            )
            return None
        checkpoint_path = get_model_checkpoint_path(model_id)
        if checkpoint_path.exists():
            return model_id
        MessageTool.MessageBoxOK(
            "The selected model is not downloaded yet.\n"
            "Open Geo-SAM Settings to download it.",
            title="Model Not Downloaded",
        )
        return None

    def _build_geosam_query(self):
        """Build a GeoSAM query object from the current canvas prompts."""
        from geosam import BoundingBox as GeoBoundingBox
        from geosam import Points as GeoPoints
        from geosam import PromptSet

        if self.runtime_crs is None or not self.runtime_crs.isValid():
            msg = (
                "The active input source does not have a valid CRS. "
                "Reload the feature folder or realtime layer and try again."
            )
            raise ValueError(msg)
        crs_text = self.runtime_crs.authid() or self.runtime_crs.toWkt()
        if not crs_text:
            msg = (
                "The active input source CRS could not be converted to a "
                "GeoSAM query CRS string."
            )
            raise ValueError(msg)
        points = None
        if (
            hasattr(self, "canvas_points")
            and len(self.canvas_points.img_crs_points) > 0
        ):
            points = GeoPoints(
                [[point.x(), point.y()] for point in self.canvas_points.img_crs_points],
                labels=[1 if label else 0 for label in self.canvas_points.labels],
                crs=crs_text,
            )

        bbox = None
        if hasattr(self, "canvas_rect") and self.canvas_rect.extent is not None:
            min_x, max_x, min_y, max_y = self.canvas_rect.extent
            bbox = GeoBoundingBox(min_x, min_y, max_x, max_y, crs=crs_text)

        if points is not None and bbox is not None:
            return PromptSet(points=points, bbox=bbox)
        return points or bbox

    def loop_prompt_type(self):
        """Loop prompt type"""
        if not hasattr(self, "tool_click_fg"):
            return
        # reset pressed to False before loop
        self.tool_click_fg.pressed = False
        self.tool_click_bg.pressed = False
        self.tool_click_rect.pressed = False

        if self.wdg_sel.pushButton_fg.isChecked():
            self.draw_background_point()
        elif self.wdg_sel.pushButton_bg.isChecked():
            self.draw_rect()
        elif self.wdg_sel.pushButton_rect.isChecked():
            self.draw_foreground_point()

    def undo_last_prompt(self):
        if len(self.prompt_history) > 0:
            prompt_last = self.prompt_history.pop()
            if prompt_last == "bbox":
                # self.canvas_rect.clear()
                self.canvas_rect.popRect()
            else:
                self.canvas_points.popPoint()
            self.execute_SAM.emit()

    def toggle_edit_mode(self):
        """Enable or disable the widget selector"""
        # radioButton = self.sender()
        radioButton = self.wdg_sel.radioButton_enable
        if not radioButton.isChecked():
            self.canvas.setMapTool(self.toolPan)
            self.wdg_sel.pushButton_fg.setEnabled(False)
            self.wdg_sel.pushButton_bg.setEnabled(False)
            self.wdg_sel.pushButton_rect.setEnabled(False)
            self.wdg_sel.pushButton_clear.setEnabled(False)
            self.wdg_sel.pushButton_undo.setEnabled(False)
            self.wdg_sel.pushButton_save.setEnabled(False)
        else:
            self.wdg_sel.pushButton_fg.setEnabled(True)
            self.wdg_sel.pushButton_bg.setEnabled(True)
            self.wdg_sel.pushButton_rect.setEnabled(True)
            self.wdg_sel.pushButton_clear.setEnabled(True)
            self.wdg_sel.pushButton_undo.setEnabled(True)
            self.wdg_sel.pushButton_save.setEnabled(True)

    def toggle_encoding_extent(self):
        """Show or hide extent of SAM encoded feature"""
        if self.wdg_sel.radioButton_show_extent.isChecked():
            if hasattr(self, "source_extent_canvas_crs"):
                self.canvas_extent.add_extent(self.source_extent_canvas_crs)
            else:
                MessageTool.MessageBar(
                    "Geo-SAM",
                    "No input source loaded.",
                )
            show_extent = True
        else:
            if not hasattr(self, "canvas_extent"):
                return
            self.canvas_extent.clear()
            show_extent = False
        save_user_settings({"show_boundary": show_extent}, mode="update")

    def toggle_hover_mode(self):
        """Toggle move mode in widget selector."""
        if self.wdg_sel.radioButton_exe_hover.isChecked():
            self.wdg_sel.radioButton_exe_hover.setChecked(False)
            self.need_execute_sam_toggle_mode = True
        else:
            self.wdg_sel.radioButton_exe_hover.setChecked(True)
            self.need_execute_sam_toggle_mode = False
        # toggle move mode in sam model
        self.toggle_sam_hover_mode()

    def toggle_sam_hover_mode(self):
        """Toggle move mode in sam model"""
        if not hasattr(self, "tool_click_fg"):
            self.preview_mode = self.wdg_sel.radioButton_exe_hover.isChecked()
            return
        if self.wdg_sel.radioButton_exe_hover.isChecked():
            self.preview_mode = True
            self.tool_click_fg.preview_mode = True
            self.tool_click_bg.preview_mode = True
            self.tool_click_rect.preview_mode = True
        else:
            self.preview_mode = False
            self.tool_click_fg.preview_mode = False
            self.tool_click_bg.preview_mode = False
            self.tool_click_rect.preview_mode = False
            # clear hover prompts
            self.tool_click_fg.clear_hover_prompt()
            self.tool_click_bg.clear_hover_prompt()
            self.tool_click_rect.clear_hover_prompt()

        if self.need_execute_sam_toggle_mode:
            self.execute_SAM.emit()

    def is_pressed_prompt(self):
        """Check if the prompt is clicked or hovered"""
        if (self.tool_click_fg.pressed or
            self.tool_click_bg.pressed or
                self.tool_click_rect.pressed):
            return True
        return False

    def filter_feature_by_area(self):
        """Filter feature by area"""
        if not self.need_execute_sam_filter_area:
            return

        t_area = self.wdg_sel.Box_min_pixel.value()
        if not hasattr(self, "polygon"):
            return

        # clear SAM canvas result
        self.polygon.canvas_prompt_polygon.clear()
        if self.preview_mode:
            self.polygon.canvas_preview_polygon.clear()

        # filter feature by new area, only show in prompt canvas
        self.polygon.add_geojson_feature_to_canvas(
            self.polygon.geojson_canvas_prompt,
            t_area,
            target="prompt"
        )

        self.t_area = t_area

    def load_default_t_area(self):
        min_pixel = self.wdg_sel.Box_min_pixel_default.value()
        self.t_area_default = min_pixel * self.pixel_area
        self.wdg_sel.Box_min_pixel.setValue(self.t_area_default)
        save_user_settings(
            {"default_minimum_pixels": min_pixel}, mode="update")

    def ensure_polygon_sam_exist(self):
        if hasattr(self, "polygon"):
            layer = QgsProject.instance().mapLayer(self.polygon.layer_id)
            if layer:
                return
        self.set_vector_layer()

    def execute_segmentation(self) -> bool:
        if not self._require_runtime_source():
            return False
        model_id = self._require_selected_model()
        if model_id is None:
            return False

        # check prompt inside feature extent and add last id to history for new prompt
        if len(self.prompt_history) > 0 and self.is_pressed_prompt():
            prompt_last = self.prompt_history[-1]
            if prompt_last == "bbox":
                last_rect = self.canvas_rect.extent
                last_prompt = QgsRectangle(
                    last_rect[0], last_rect[2], last_rect[1], last_rect[3])
            else:
                last_point = self.canvas_points.img_crs_points[-1]
                last_prompt = QgsRectangle(last_point, last_point)
            if (
                self.runtime_extent is not None
                and not last_prompt.intersects(self.runtime_extent)
            ):
                self.check_message_box_outside()
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

        # clear canvas prompt polygon for new prompt
        if self.is_pressed_prompt():
            self.polygon.canvas_prompt_polygon.clear()

        query = self._build_geosam_query()
        if query is None:
            return True
        try:
            if (
                self.runtime_source_kind == "realtime"
                and self.runtime_layer is not None
            ):
                result = query_raster_layer(self.runtime_layer, model_id, query)
            elif self.runtime_source_kind == "feature":
                result = query_feature_source(self.feature_dir, model_id, query)
            else:
                MessageTool.MessageBoxOK(
                    "Please choose a RealTime Layer or a Feature folder first.",
                    title="Input Required",
                )
                return False
        except Exception as exc:
            MessageTool.MessageBoxOK(str(exc), title="Geo-SAM Query Failed")
            return False

        geojson_features = query_result_to_geojson_features(result)
        self.polygon.canvas_preview_polygon.clear()
        self.polygon.canvas_prompt_polygon.clear()
        if self.preview_mode:
            self.polygon.add_geojson_feature_to_canvas(
                geojson_features,
                self.t_area,
                overwrite_geojson=True
            )
        else:
            self.polygon.add_geojson_feature_to_canvas(
                geojson_features,
                self.t_area,
                target="prompt",
                overwrite_geojson=True
            )

        # show pressed prompt result in preview mode
        if self.preview_mode and self.is_pressed_prompt():
            self.polygon.add_geojson_feature_to_canvas(
                self.polygon.geojson_canvas_preview,  # update with canvas polygon
                self.t_area,
                target="prompt",
                overwrite_geojson=True
            )
        self.topping_polygon_sam_layer()

        return True

    def draw_foreground_point(self):
        """Draw foreground point in canvas"""
        if not hasattr(self, "tool_click_fg"):
            self._require_runtime_source()
            return
        self.canvas.setMapTool(self.tool_click_fg)
        button = self.wdg_sel.pushButton_fg
        if not button.isChecked():
            button.toggle()

        if self.wdg_sel.pushButton_bg.isChecked():
            self.wdg_sel.pushButton_bg.toggle()
        if self.wdg_sel.pushButton_rect.isChecked():
            self.wdg_sel.pushButton_rect.toggle()
        self.prompt_type = "fgpt"

    def draw_background_point(self):
        """Draw background point in canvas"""
        if not hasattr(self, "tool_click_bg"):
            self._require_runtime_source()
            return
        self.canvas.setMapTool(self.tool_click_bg)
        button = self.wdg_sel.pushButton_bg
        if not button.isChecked():
            button.toggle()

        if self.wdg_sel.pushButton_fg.isChecked():
            self.wdg_sel.pushButton_fg.toggle()
        if self.wdg_sel.pushButton_rect.isChecked():
            self.wdg_sel.pushButton_rect.toggle()
        self.prompt_type = "bgpt"

    def draw_rect(self):
        """Draw rectangle in canvas"""
        if not hasattr(self, "tool_click_rect"):
            self._require_runtime_source()
            return
        self.canvas.setMapTool(self.tool_click_rect)
        button = self.wdg_sel.pushButton_rect  # self.sender()
        if not button.isChecked():
            button.toggle()

        if self.wdg_sel.pushButton_fg.isChecked():
            self.wdg_sel.pushButton_fg.toggle()
        if self.wdg_sel.pushButton_bg.isChecked():
            self.wdg_sel.pushButton_bg.toggle()
        self.prompt_type = "bbox"

    def set_vector_layer(self):
        """Set sam output vector layer"""
        if not hasattr(self, "img_crs_manager"):
            MessageTool.MessageBar(
                "Geo-SAM",
                "Choose an input source before configuring the output layer.",
                level="warning",
            )
            return
        new_layer = self.wdg_sel.MapLayerComboBox.currentLayer()

        # parse whether the new selected layer is same as current layer
        if hasattr(self, "polygon"):
            old_layer = QgsProject.instance().mapLayer(self.polygon.layer_id)
            if (old_layer and new_layer and
                    old_layer.id() == new_layer.id()):
                return
            if not self.polygon.reset_layer(new_layer):
                self.MapLayerComboBox.setLayer(None)
        else:
            self.polygon = SAM_PolygonFeature(
                self.img_crs_manager, layer=new_layer,
                kwargs_preview_polygon=self.style_preview_polygon,
                kwargs_prompt_polygon=self.style_prompt_polygon
            )

        # clear layer history
        self.sam_feature_history = []
        self.wdg_sel.MapLayerComboBox.setLayer(self.polygon.layer)
        # self.set_user_settings_color()

    def load_vector_file(self) -> None:
        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix("shp")
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_path, _ = file_dialog.getSaveFileName(
            None, "QFileDialog.getOpenFileName()",
            "",
            "Shapefile (*.shp)",
            options=QFileDialog.DontConfirmOverwrite
        )

        if file_path is None or file_path == "":
            return

        file_path = Path(file_path)
        if not file_path.parent.is_dir():
            MessageTool.MessageBoxOK(
                "Oops: "
                "Failed to open file, please choose a existing folder"
            )
            return
        if file_path.suffix.lower() != ".shp":
            file_path.with_suffix(".shp")

        layer_list = QgsProject.instance().mapLayersByName(file_path.stem)
        if len(layer_list) > 0:
            self.polygon = SAM_PolygonFeature(
                self.img_crs_manager, layer=layer_list[0],
                kwargs_preview_polygon=self.style_preview_polygon,
                kwargs_prompt_polygon=self.style_prompt_polygon
            )
            if not hasattr(self.polygon, "layer"):
                return
            MessageTool.MessageBar(
                "Attention",
                f"Layer '{file_path.name}' has already been in the project, "
                "you can start labeling now"
            )
            self.wdg_sel.MapLayerComboBox.setLayer(self.polygon.layer)

        else:
            self.polygon = SAM_PolygonFeature(
                self.img_crs_manager, shapefile=file_path,
                kwargs_preview_polygon=self.style_preview_polygon,
                kwargs_prompt_polygon=self.style_prompt_polygon
            )
            if not hasattr(self.polygon, "layer"):
                return
        # clear layer history
        self.sam_feature_history = []
        self.wdg_sel.MapLayerComboBox.setLayer(self.polygon.layer)
            # self.set_user_settings_color()

    def load_feature(self):
        """Load feature"""
        self.feature_dir = self.wdg_sel.QgsFile_feature.filePath()
        if self.feature_dir is not None and os.path.exists(self.feature_dir):
            self.clear_layers(clear_extent=True)
            try:
                self._activate_runtime_from_inputs()
            except Exception as exc:
                MessageTool.MessageBoxOK(
                    str(exc),
                    title="Feature Folder Failed",
                )
        else:
            MessageTool.MessageBoxOK(
                "Feature folder does not exist.",
                title="Invalid Feature Folder",
            )

    def clear_layers(self, clear_extent: bool = False):
        """Clear all temporary layers (canvas and new sam result) and reset prompt"""
        self.clear_canvas_layers_safely(clear_extent=clear_extent)
        if hasattr(self, "polygon"):
            self.polygon.clear_canvas_polygons()
        self.prompt_history.clear()

    def save_shp_file(self):
        """Save sam result into shapefile layer"""
        need_toggle = False
        if self.preview_mode:
            need_toggle = True
            self.toggle_hover_mode()
            if len(self.prompt_history) == 0:
                MessageTool.MessageBoxOK(
                    "Preview mode only shows prompt previews. "
                    "Click first to apply the prompt."
                )
                self.toggle_hover_mode()
                return False

        if hasattr(self, "polygon"):
            self.polygon.add_geojson_feature_to_layer(
                self.polygon.geojson_canvas_prompt,
                self.t_area,
                self.prompt_history
            )

            self.polygon.commit_changes()
            self.polygon.canvas_preview_polygon.clear()
            self.polygon.canvas_prompt_polygon.clear()

            # add last id of new features to history
            features = list(self.polygon.layer.getFeatures())
            if len(list(features)) == 0:
                return None
            last_id = features[-1].id()

            if len(self.sam_feature_history) > 0:
                if self.sam_feature_history[-1][0] > last_id:
                    MessageTool.MessageLog(
                        "New features id is smaller than last id in history",
                        level="warning"
                    )
                self.sam_feature_history[-1].append(last_id)

        # reenable preview mode
        if need_toggle:
            self.toggle_hover_mode()
        self.clear_canvas_layers_safely()
        self.prompt_history.clear()
        self.polygon.reset_geojson()

        # avoid execute sam when reset min pixel to default value
        self.need_execute_sam_filter_area = False
        self.wdg_sel.Box_min_pixel.setValue(self.t_area_default)
        self.need_execute_sam_filter_area = True
        return None

    def reset_prompt_type(self):
        """Reset prompt type"""
        if hasattr(self, "prompt_type"):
            if self.prompt_type == "bbox":
                self.draw_rect()
            else:
                self.draw_foreground_point()
        else:
            self.draw_foreground_point()

    def undo_sam_polygon(self):
        """Undo last sam polygon"""
        if len(self.sam_feature_history) == 0:
            return
        last_ids = self.sam_feature_history.pop(-1)
        if len(last_ids) == 1:
            self.clear_layers(clear_extent=False)
            return
        rm_ids = list(range(last_ids[0], last_ids[1]+1))
        self.polygon.layer.dataProvider().deleteFeatures(rm_ids)

        # If caching is enabled, a simple canvas refresh might not be sufficient
        # to trigger a redraw and must clear the cached image for the layer
        if self.canvas.isCachingEnabled():
            self.polygon.layer.triggerRepaint()
        else:
            self.canvas.refresh()

    def check_message_box_outside(self):
        if self.preview_mode:
            return True
        return MessageTool.MessageBoxOK(
            "Point or rectangle is outside the source boundary. "
            "Click OK to undo the last prompt."
        )

    def reset_background_color(self):
        """Reset background color"""
        color = self.wdg_sel.ColorButton_bgpt.color()
        save_user_settings(
            {"bg_color": color.name()},
            mode="update"
        )
        if not hasattr(self, "canvas_points"):
            return
        self.canvas_points.background_color = color
        self.canvas_points.flush_points_color()
        self.tool_click_bg.reset_cursor_color(color.name())

    def reset_foreground_color(self):
        """Reset foreground color"""
        color = self.wdg_sel.ColorButton_fgpt.color()
        save_user_settings(
            {"fg_color": color.name()},
            mode="update"
        )
        if not hasattr(self, "canvas_points"):
            return
        self.canvas_points.foreground_color = color
        self.canvas_points.flush_points_color()
        self.tool_click_fg.reset_cursor_color(color.name())

    def reset_rectangular_color(self):
        """Reset rectangular color"""
        color = self.wdg_sel.ColorButton_bbox.color()
        save_user_settings(
            {"bbox_color": color.name()},
            mode="update"
        )
        if not hasattr(self, "canvas_rect"):
            return

        color_fill = list(color.getRgb())
        color_fill[-1] = 10
        color_fill = QColor(*color_fill)
        self.canvas_rect.set_line_color(color)
        self.canvas_rect.set_fill_color(color_fill)

        self.tool_click_rect.reset_cursor_color(color.name())

    def reset_extent_color(self):
        """Reset extent color"""
        color = self.wdg_sel.ColorButton_extent.color()
        if not hasattr(self, "canvas_extent"):
            return
        self.canvas_extent.set_color(color)
        save_user_settings(
            {"extent_color": color.name()},
            mode="update"
        )

        if self.wdg_sel.radioButton_show_extent.isChecked():
            self.canvas_extent.clear()
            if hasattr(self, "source_extent_canvas_crs"):
                self.canvas_extent.add_extent(self.source_extent_canvas_crs)

    def reset_prompt_polygon_color(self):
        """Reset prompt polygon color"""
        if not hasattr(self, "polygon"):
            return
        color = self.wdg_sel.ColorButton_prompt.color()
        save_user_settings(
            {"prompt_color": color.name()},
            mode="update"
        )
        self.polygon.canvas_prompt_polygon.set_line_style(color)

    def reset_preview_polygon_color(self):
        """Reset preview polygon color"""
        if not hasattr(self, "polygon"):
            return
        color = self.wdg_sel.ColorButton_preview.color()
        save_user_settings(
            {"preview_color": color.name()},
            mode="update"
        )

        self.polygon.canvas_preview_polygon.set_line_style(color)

    def toggle_load_demo_img(self):
        """Toggle whether load demo image"""
        return


class EncoderCopilot(QDockWidget):
    # TODO: support encoding process in this widget
    retrieve_range = pyqtSignal(str)
    retrieve_patch = pyqtSignal(str)

    def __init__(self, parent, iface: QgisInterface, cwd: str):
        QDockWidget.__init__(self)
        self.parent = parent
        self.iface = iface
        self.cwd = Path(cwd)
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
        """Create widget selector"""
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
            self.retrieve_patch.connect(self.show_patch_extent_in_canvas)

            # Slots fire in the same connection order when a signal is emitted.
            self.wdg_copilot.closed.connect(self.destruct)
            self.wdg_copilot.closed.connect(self.iface.actionPan().trigger)
            self.wdg_copilot.closed.connect(self.reset_to_project_crs)

            ########## set default values ##########
            # collapse group boxes
            self.wdg_copilot.AdvancedParameterGroupBox.setCollapsed(True)
            # checkpoint
            self.wdg_copilot.CheckpointFileWidget.setFilter("*.pt")
            self.wdg_copilot.CheckpointFileWidget.setStorageMode(
                QgsFileWidget.GetFile)
            self.wdg_copilot.CheckpointFileWidget.setConfirmOverwrite(False)

            if self.wdg_copilot.SAMModelComboBox.count() == 0:
                self.wdg_copilot.SAMModelComboBox.addItem("", "")
                for model_id, label in get_model_display_items():
                    self.wdg_copilot.SAMModelComboBox.addItem(label, model_id)

            self.wdg_copilot.SAMModelComboBox.setCurrentIndex(0)

            self.dockFirstOpen = False
            # add widget to QGIS
            # self.iface.addDockWidget(Qt.BottomDockWidgetArea, self.wdg_copilot)
        else:
            pass

        if not self.wdg_copilot.isUserVisible():
            self.wdg_copilot.setUserVisible(True)

    def set_range_to_widget(self, range: str):
        """Set range to widget"""
        range = eval(range)
        self.wdg_copilot.MinValueBox.setValue(range[0])
        self.wdg_copilot.MaxValueBox.setValue(range[1])
        self.wdg_copilot.label_range_status.setText("Done!")

    def show_patch_extent_in_canvas(self, extents: str):
        extents = eval(extents)
        num_patch = len(extents)
        idx = np.random.randint(0, num_patch, size=(int(num_patch/10)))
        alphas = np.full(num_patch, 100)
        line_widths = np.full(num_patch, 2)
        alphas[idx] = 254
        line_widths[idx] = 5

        for i, extent in enumerate(extents):
            if i == num_patch - 1:
                alpha = 255
                line_width = 5
            else:
                alpha = alphas[i]

            line_width = line_widths[i]

            self.canvas_extent.add_extent(
                QgsRectangle(*extent),
                use_type="patch_extent",
                alpha=alpha,
                line_width=line_width
            )
        self.wdg_copilot.label_patch_settings.setText(
            f"Done! {len(extents)} patches")

    def parse_raster_info(self):
        """Parse raster info and set to widget items"""
        # clear widget items
        self.wdg_copilot.label_range_status.setText("")
        self.wdg_copilot.label_patch_settings.setText("")
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
            return

        # set raster layer band to band field
        self.wdg_copilot.RasterBandComboBox_R.setLayer(self.raster_layer)
        self.wdg_copilot.RasterBandComboBox_G.setLayer(self.raster_layer)
        self.wdg_copilot.RasterBandComboBox_B.setLayer(self.raster_layer)

        # set crs to layer crs
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
        """Parse min and max value from raster layer"""
        if not self.valid_raster_layer():
            return
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
        """Check whether a valid raster layer is selected."""
        layer = self.wdg_copilot.MapLayerComboBox.currentLayer()
        if isinstance(layer, QgsRasterLayer):
            self.raster_layer = self.wdg_copilot.MapLayerComboBox.currentLayer()
            return True
        MessageTool.MessageBoxOK(
            "Oops: Invalid Raster Layer. Please select a valid raster layer!"
        )
        self.raster_layer = None
        return False

    def get_bands(self) -> list[int]:
        return [
            self.wdg_copilot.RasterBandComboBox_R.currentBand(),
            self.wdg_copilot.RasterBandComboBox_G.currentBand(),
            self.wdg_copilot.RasterBandComboBox_B.currentBand(),
        ]

    def get_resolutions(self) -> tuple[float, float]:
        """Get x, y resolution from resolution group box"""
        scale = self.wdg_copilot.BoxResolutionScale.value()
        resolution_layer = (
            self.raster_layer.rasterUnitsPerPixelX(),
            self.raster_layer.rasterUnitsPerPixelY()
        )
        return (
            resolution_layer[0] * scale,
            resolution_layer[1] * scale
        )

    def get_stride(self) -> int:
        """Get stride from overlap group box"""
        overlaps = self.wdg_copilot.BoxOverlap.value()
        return int((100 - overlaps)/100 * 1024)

    def get_extent(self) -> QgsRectangle:
        return self.wdg_copilot.ExtentGroupBox.outputExtent()

    def get_extent_str(self) -> str:
        """Get extent string from extent group box"""
        extent = self.get_extent()
        xMin, yMin, xMax, yMax = (extent.xMinimum(),
                                  extent.yMinimum(),
                                  extent.xMaximum(),
                                  extent.yMaximum())
        return f"{xMin}, {xMax}, {yMin}, {yMax}"

    def get_checkpoint_path(self) -> str:
        """Get checkpoint path from file widget"""
        return self.wdg_copilot.CheckpointFileWidget.filePath()

    def parse_model_type(self) -> None:
        checkpoint_path = self.get_checkpoint_path()
        if not checkpoint_path:
            self.wdg_copilot.SAMModelComboBox.setCurrentIndex(0)
            return
        try:
            model_id = infer_model_id_from_checkpoint_path(checkpoint_path)
        except ValueError:
            return
        model_index = self.wdg_copilot.SAMModelComboBox.findData(model_id)
        if model_index >= 0:
            self.wdg_copilot.SAMModelComboBox.setCurrentIndex(model_index)

    def get_model_type(self) -> str | None:
        """Get the GeoSAM model id from the combo box."""
        model_type = self.wdg_copilot.SAMModelComboBox.currentData()
        if model_type in ("", None):
            return None
        return str(model_type)

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
        """Check if extent has been set. If not, alert user to set extent"""
        extent = self.get_extent_str()
        vals = extent.split(",")
        if (float(vals[0]) == float(vals[1])
                or float(vals[2]) == float(vals[3])):
            # alert user to set extent
            MessageTool.MessageBoxOK(
                "Oops: Extent has not been set. Please set extent first."
            )
            return False
        return True

    def check_raster_selected(self) -> bool:
        """Check whether a raster layer has been selected."""
        if self.raster_layer is not None:
            return True
        # alert user to select a raster layer
        MessageTool.MessageBoxOK(
            "Oops: "
            "Raster Layer has not been selected/detected. "
            "Please set/reset Raster Layer first!"
        )
        return False

    def check_setting_available(self) -> bool:
        """Check if setting is available. If not, alert user to set setting"""
        if not self.check_raster_selected() or not self.check_extent_set():
            return False
        return True

    def retrieve_setting(self) -> str:
        """Retrieve setting from widget items"""
        if not self.check_setting_available():
            return None
        stride = self.get_stride()
        resolution = self.get_resolutions()
        bands = self.get_bands()
        crs = self.crs_layer.authid()
        extent = f"{self.get_extent_str()} [{crs}]"
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
        return json.dumps(json_dict, indent=4)

    def json_setting_to_clipboard(self) -> None:
        """Copy setting to clipboard"""
        json_str = self.retrieve_setting()
        if json_str is None:
            return

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
            return

        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix("json")
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
                return
            if file_path.suffix != ".json":
                file_path.with_suffix(".json")

            with open(file_path, "w") as f:
                f.write(json_str)
        except Exception as e:
            MessageTool.MessageLog(
                f"Failed to save setting to file. Error: {e}",
                level="critical"
            )
            return

    def show_extents(self):
        self.show_bbox_extent()
        self.show_patch_extent()

    def show_bbox_extent(self):
        """Show bbox extent in canvas"""
        if hasattr(self, "canvas_extent"):
            self.canvas_extent.clear()
            extent = self.get_extent()
            self.canvas_extent.add_extent(extent)

    def show_patch_extent(self):
        """Show all patch extents in canvas"""
        if not self.check_setting_available():
            return

        extent = self.get_extent()
        source_path = self.raster_layer.dataProvider().dataSourceUri()
        try:
            patch_extents = chip_extent_rectangles_for_source(
                source_path,
                bands=self.get_bands(),
                crs=None,
                res=self.get_resolutions()[0],
                extent=(
                    extent.xMinimum(),
                    extent.yMinimum(),
                    extent.xMaximum(),
                    extent.yMaximum(),
                ),
                extent_crs=self.crs_layer.authid() or self.crs_layer.toWkt(),
                chip_size=1024,
                stride=self.get_stride(),
            )
        except Exception as exc:
            MessageTool.MessageBoxOK(
                f"Failed to compute patch extents: {exc}",
                title="Patch Preview Failed",
            )
            return

        if len(patch_extents) == 0:
            MessageTool.MessageBar(
                "Oops!!!",
                "No available patch sample inside the chosen extent!!! "
                "Please choose another extent.",
                duration=30
            )
            return

        self.wdg_copilot.label_patch_settings.setText("Computing ...")
        self.show_patch_extent_thread = ShowPatchExtentThread(
            self.retrieve_patch,
            [{"bbox": QgsRectangle(*patch_extent)} for patch_extent in patch_extents],
        )
        self.show_patch_extent_thread.start()

    def reset_to_project_crs(self):
        self.project.setCrs(self.crs_project)

    def reset_canvas(self):
        self.reset_to_project_crs()
        if hasattr(self, "canvas_extent"):
            self.canvas_extent.clear()

    def destruct(self):
        """Destruct actions when closed widget"""
        self.reset_canvas()

    def unload(self):
        """Unload actions when plugin is closed"""
        self.reset_canvas()
