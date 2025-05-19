from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
from qgis._gui import QgsMapMouseEvent
from qgis.core import (
    Qgis,
    QgsField,
    QgsFields,
    QgsFillSymbol,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsVectorLayerUtils,
    QgsWkbTypes,
)
from qgis.gui import (
    QgsMapCanvas,
    QgsMapTool,
    QgsMapToolEmitPoint,
    QgsRubberBand,
    QgsVertexMarker,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.utils import iface
from rasterio.transform import Affine, rowcol

if TYPE_CHECKING:
    from .widgetTool import Selector

from ..ui.cursors import (
    UI_SCALE,
    CursorPointBG,
    CursorPointFG,
    CursorRect,
    customize_bbox_cursor,
    customize_bg_point_cursor,
    customize_fg_point_cursor,
)
from .geoTool import ImageCRSManager, LayerExtent
from .messageTool import MessageTool
from .ulid import GroupId

qgis_version = Qgis.QGIS_VERSION_INT

# QVariant has been deprecated in version 3.38, use QMetaType instead
if qgis_version < 33800:
    from qgis.PyQt.QtCore import QVariant as QMetaType

    QMetaType.QString = QMetaType.String
else:
    from qgis.PyQt.QtCore import QMetaType

SAM_Feature_Fields = [
    QgsField("group_ulid", QMetaType.QString),
    QgsField("N_GM", QMetaType.Int),
    QgsField("id", QMetaType.Int),
    QgsField("Area", QMetaType.Double),
    QgsField("N_FG", QMetaType.Int),
    QgsField("N_BG", QMetaType.Int),
    QgsField("BBox", QMetaType.Bool),
]
SAM_Feature_QgsFields = QgsFields()
new_fields = QgsFields()
for field in SAM_Feature_Fields:
    SAM_Feature_QgsFields.append(field)

SuffixDriverMap = {
    ".shp": "ESRI Shapefile",
    ".gpkg": "GPKG",
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
    ".sqlite": "SQLite",
}


class Canvas_Rectangle:
    """A class to manage Rectangle on canvas."""

    def __init__(
        self,
        canvas: QgsMapCanvas,
        img_crs_manager: ImageCRSManager,
        use_type: str = "bbox",
        alpha: int = 255,
        line_width=None,
    ):
        self.canvas = canvas
        self.qgis_project = QgsProject.instance()
        self.rect_list = []
        # self.box_geo = None
        self.img_crs_manager = img_crs_manager
        self.alpha = alpha
        self.line_width = line_width
        self.rubberBand = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)

        self.colors_bbox = {"fill_color": QColor(0, 0, 255, 10), "line_color": Qt.blue}
        self.colors_extent = {
            "fill_color": QColor(0, 0, 0, 0),
            "line_color": QColor(255, 0, 0),
        }

        if use_type == "bbox":
            self._init_bbox_layer()
        elif use_type == "extent":
            self._init_extent_layer()
        elif use_type == "patch_extent":
            self._init_batch_extent_layer()

    def flush_rect_color(self):
        """Flush the color of rectangle"""
        if len(self.rect_list) == 0:
            return None
        else:
            startPoint, endPoint = self.rect_list[-1]
            self.showRect(startPoint, endPoint)

    def _init_bbox_layer(self):
        """Initialize the rectangle layer for bbox prompt"""
        line_width = 1
        self.set_layer_style(
            self.colors_bbox["fill_color"], self.colors_bbox["line_color"], line_width
        )

    def _init_extent_layer(self):
        """Initialize the rectangle layer for extent of features"""
        # line_color2 = QColor(255, 255, 255)
        line_color2 = None  # not set secondary color currently
        line_width = 3
        self.set_layer_style(
            self.colors_extent["fill_color"],
            self.colors_extent["line_color"],
            line_width,
            line_color2,
        )

    def _init_batch_extent_layer(self):
        """Initialize the rectangle layer for batch extent"""
        color_random = np.random.randint(0, 255, size=3).tolist()
        fill_color = QColor(0, 0, 0, 0)
        line_color = QColor(
            color_random[0], color_random[1], color_random[2], self.alpha
        )
        # line_color2 = QColor(255, 255, 255)
        if self.line_width is None:
            line_width = 2
        else:
            line_width = self.line_width
        line_color2 = None

        if self.alpha == 255:
            line_color = QColor(255, 0, 0)
            line_color2 = QColor(255, 0, 0)

        self.set_layer_style(fill_color, line_color, line_width, line_color2)

    def set_fill_color(self, fill_color: QColor):
        if fill_color is not None:
            self.rubberBand.setFillColor(fill_color)

    def set_line_color(self, line_color: QColor):
        if line_color is not None:
            self.rubberBand.setStrokeColor(line_color)

    def set_line_width(self, line_width: int):
        if line_width is not None:
            self.rubberBand.setWidth(line_width)

    def set_line_color_2(self, line_color_2: QColor):
        if line_color_2 is not None:
            self.rubberBand.setSecondaryStrokeColor(line_color_2)

    def set_layer_style(self, fill_color, line_color, line_width, line_color_2=None):
        """Set the style of the rectangle layer"""
        self.set_fill_color(fill_color)
        self.set_line_color(line_color)
        self.set_line_color_2(line_color_2)
        self.set_line_width(line_width)

    def clear(self):
        """Clear the rectangle on canvas"""
        self.rubberBand.reset(QgsWkbTypes.PolygonGeometry)
        self.canvas.refresh()
        self.rect_list.clear()

    def addRect(self, startPoint: QgsPointXY, endPoint: QgsPointXY):
        self.rect_list.append((startPoint, endPoint))

    def popRect(self, show_rect: bool = True, clear_canvas: bool = True):
        if len(self.rect_list) > 0:
            self.rect_list.pop()
            if len(self.rect_list) > 0:
                startPoint, endPoint = self.rect_list[-1]
                if show_rect:
                    self.showRect(startPoint, endPoint)
            else:
                if clear_canvas:
                    self.clear()

    def showRect(self, startPoint, endPoint):
        self.rubberBand.reset(QgsWkbTypes.PolygonGeometry)
        if startPoint.x() == endPoint.x() or startPoint.y() == endPoint.y():
            return None

        point1 = QgsPointXY(startPoint.x(), startPoint.y())
        point2 = QgsPointXY(startPoint.x(), endPoint.y())
        point3 = QgsPointXY(endPoint.x(), endPoint.y())
        point4 = QgsPointXY(endPoint.x(), startPoint.y())

        self.rubberBand.addPoint(point1, False)
        self.rubberBand.addPoint(point2, False)
        self.rubberBand.addPoint(point3, False)
        self.rubberBand.addPoint(point4, True)  # true to update canvas
        self.rubberBand.show()

    @property
    def box_geo(self):
        """Returns a rectangle from two points with img crs"""
        if len(self.rect_list) == 0:
            return None
        else:
            # startPoint endPoint transform
            startPoint, endPoint = self.rect_list[-1]
            startPoint = self.img_crs_manager.point_to_img_crs(
                startPoint, self.qgis_project.crs()
            )
            endPoint = self.img_crs_manager.point_to_img_crs(
                endPoint, self.qgis_project.crs()
            )
            return [startPoint.x(), startPoint.y(), endPoint.x(), endPoint.y()]

    @property
    def extent(self):
        """Return the extent of the rectangle (minX, maxX. minY, maxY)"""
        if self.box_geo is not None:
            extent = [
                min(self.box_geo[0], self.box_geo[2]),
                max(self.box_geo[0], self.box_geo[2]),
                min(self.box_geo[1], self.box_geo[3]),
                max(self.box_geo[1], self.box_geo[3]),
            ]
            return extent
        else:
            return None

    def get_img_box(self, transform):
        """Return the box for SAM image"""
        if self.box_geo is not None:
            rowcol1 = rowcol(transform, self.box_geo[0], self.box_geo[1])
            rowcol2 = rowcol(transform, self.box_geo[2], self.box_geo[3])
            box = [
                min(rowcol1[1], rowcol2[1]),
                min(rowcol1[0], rowcol2[0]),
                max(rowcol1[1], rowcol2[1]),
                max(rowcol1[0], rowcol2[0]),
            ]
            return np.array(box)
        else:
            return None


class RectangleMapTool(QgsMapToolEmitPoint):
    """A map tool to draw a rectangle on canvas"""

    def __init__(
        self,
        canvas_rect: Canvas_Rectangle,
        prompt_history: List[Any],
        execute_SAM: pyqtSignal,
        img_crs_manager: ImageCRSManager,
    ):
        self.qgis_project = QgsProject.instance()
        self.canvas_rect = canvas_rect
        self.prompt_history = prompt_history
        self.execute_SAM = execute_SAM
        self.img_crs_manager = img_crs_manager
        self.preview_mode: bool = False
        self.have_added_for_moving: bool = False
        self.pressed: bool = False

        QgsMapToolEmitPoint.__init__(self, self.canvas_rect.canvas)
        self.setCursor(CursorRect)

        self.reset()

    def reset_cursor_color(self, color):
        Cursor_User = customize_bbox_cursor(color)
        self.setCursor(Cursor_User)

    def reset(self):
        self.startPoint = self.endPoint = None
        self.isEmittingPoint = False
        self.have_added_for_moving = False
        self.canvas_rect.rubberBand.reset(QgsWkbTypes.PolygonGeometry)

    def canvasPressEvent(self, e):
        self.pressed = False
        self.startPoint = self.toMapCoordinates(e.pos())
        self.endPoint = self.startPoint
        self.isEmittingPoint = True
        self.canvas_rect.showRect(self.startPoint, self.endPoint)

    def canvasReleaseEvent(self, e):
        self.pressed = True
        self.isEmittingPoint = False
        self.clear_hover_prompt()
        if self.startPoint is None or self.endPoint is None:
            return None
        elif (
            self.startPoint.x() == self.endPoint.x()
            or self.startPoint.y() == self.endPoint.y()
        ):
            return None
        else:
            self.canvas_rect.addRect(self.startPoint, self.endPoint)
            self.prompt_history.append("bbox")
            self.execute_SAM.emit()
            self.have_added_for_moving = False  # reset to False

    def clear_hover_prompt(self):
        # remove the last rectangle if have added a rectangle when mouse move
        if self.have_added_for_moving:
            self.canvas_rect.popRect(show_rect=False, clear_canvas=False)
            self.have_added_for_moving = False  # reset to False

    def canvasMoveEvent(self, e):
        if not self.isEmittingPoint:
            return

        self.pressed = False
        # update the rectangle as the mouse moves
        self.endPoint = self.toMapCoordinates(e.pos())
        self.canvas_rect.showRect(self.startPoint, self.endPoint)

        # execute SAM when mouse move
        if not self.preview_mode:
            return
        if self.startPoint is None or self.endPoint is None:
            return None
        elif (
            self.startPoint.x() == self.endPoint.x()
            or self.startPoint.y() == self.endPoint.y()
        ):
            return None
        else:
            self.canvas_rect.popRect(show_rect=False, clear_canvas=False)
            self.canvas_rect.addRect(self.startPoint, self.endPoint)
            self.execute_SAM.emit()
            self.have_added_for_moving = True

    def deactivate(self):
        QgsMapTool.deactivate(self)
        self.deactivated.emit()


class Canvas_Extent:
    """A class to manage feature Extent on canvas."""

    def __init__(self, canvas: QgsMapCanvas, img_crs_manager: ImageCRSManager) -> None:
        self.canvas = canvas
        self.img_crs_manager = img_crs_manager
        self.canvas_rect_list: List[Canvas_Rectangle] = []
        self.color = None

    def clear(self):
        """Clear all extents on canvas"""
        for canvas_rect in self.canvas_rect_list:
            canvas_rect.clear()
        self.canvas_rect_list = []

    def set_color(self, color: QColor):
        """Set the color of the extent"""
        self.color = color
        for canvas_rect in self.canvas_rect_list:
            canvas_rect.set_line_color(self.color)

    def add_extent(
        self,
        extent: QgsRectangle,
        use_type: str = "extent",
        alpha: int = 255,
        line_width=None,
    ):
        """Add a extent on canvas"""
        xMin, yMin, xMax, yMax = (
            extent.xMinimum(),
            extent.yMinimum(),
            extent.xMaximum(),
            extent.yMaximum(),
        )
        canvas_rect = Canvas_Rectangle(
            self.canvas,
            self.img_crs_manager,
            use_type=use_type,
            alpha=alpha,
            line_width=line_width,
        )
        if self.color is not None:
            canvas_rect.set_line_color(self.color)

        point1 = QgsPointXY(xMin, yMax)  # left top
        point2 = QgsPointXY(xMin, yMin)  # left bottom
        point3 = QgsPointXY(xMax, yMin)  # right bottom
        point4 = QgsPointXY(xMax, yMax)  # right top

        canvas_rect.rubberBand.addPoint(point1, False)
        canvas_rect.rubberBand.addPoint(point2, False)
        canvas_rect.rubberBand.addPoint(point3, False)
        # true to update canvas
        canvas_rect.rubberBand.addPoint(point4, True)
        canvas_rect.rubberBand.show()

        self.canvas_rect_list.append(canvas_rect)


class Canvas_Points:
    """
    A class to manage points on canvas.
    """

    def __init__(self, canvas: QgsMapCanvas, img_crs_manager: ImageCRSManager):
        """
        Parameters:
        ----------
        canvas: QgsMapCanvas
            canvas to add points
        img_crs_manager: ImageCRSManager
            The manager to transform points between image crs and other crs
        """
        self.canvas = canvas
        self.extent = None
        self.img_crs_manager = img_crs_manager
        self.markers: List[QgsVertexMarker] = []
        self.img_crs_points: List[QgsPointXY] = []
        self.labels: List[bool] = []
        self.foreground_color = QColor(0, 0, 255)
        self.background_color = QColor(255, 0, 0)
        self.point_size = 1
        # enum type: circle = 4, https://api.qgis.org/api/qgsvertexmarker_8h_source.html
        self.icon_type = QgsVertexMarker.ICON_CIRCLE

    @property
    def project_crs(self):
        return QgsProject.instance().crs()

    def addPoint(self, point: QgsPointXY, foreground: bool, show: bool = True):
        """
        Parameters:
        ----------
        point: QgsPointXY
            point to add marker
        foreground: bool
            True for foreground, False for background
        """
        m = QgsVertexMarker(self.canvas)
        m.setCenter(point)
        if show:
            if foreground:
                m.setColor(self.foreground_color)
                m.setFillColor(self.foreground_color)
            else:
                m.setColor(self.background_color)
                m.setFillColor(self.background_color)
            # m.setIconSize(12)
            m.setIconSize(round(UI_SCALE * self.point_size / 3))
            m.setIconType(self.icon_type)

        # add to markers and labels
        self.markers.append(m)
        point_img_crs = self.img_crs_manager.point_to_img_crs(point, self.project_crs)
        self.img_crs_points.append(point_img_crs)
        self.labels.append(foreground)

        self._update_extent()

    def flush_points_style(self):
        """Flush the color of points"""
        for i, m in enumerate(self.markers):
            if self.labels[i]:
                m.setColor(self.foreground_color)
                m.setFillColor(self.foreground_color)
                m.setIconSize(round(UI_SCALE * self.point_size / 3))
                m.setIconType(self.icon_type)
            else:
                m.setColor(self.background_color)
                m.setFillColor(self.background_color)
                m.setIconSize(round(UI_SCALE * self.point_size / 3))
                m.setIconType(self.icon_type)

    def popPoint(self):
        """remove the last marker"""
        if len(self.markers) > 0:
            m = self.markers.pop()
            self.canvas.scene().removeItem(m)
            self.img_crs_points.pop()
            self.labels.pop()

            self._update_extent()

    def clear(self):
        """remove all markers"""
        for m in self.markers:
            self.canvas.scene().removeItem(m)
        self.canvas.refresh()

        self.markers = []
        self.img_crs_points = []
        self.labels = []
        self._update_extent()

    def _update_extent(self):
        """update extent of markers with image crs"""
        points = []
        if len(self.markers) == 0:
            self.extent = None
        else:
            for m in self.markers:
                points.append(m.center())
            extent = QgsGeometry.fromMultiPointXY(points).boundingBox()
            if self.project_crs != self.img_crs_manager.img_crs:
                extent = self.img_crs_manager.extent_to_img_crs(
                    extent, self.project_crs
                )
            self.extent = LayerExtent.from_qgis_extent(extent)

    def get_points_and_labels(self, tf: Affine):
        """Returns points and labels for SAM image"""
        if len(self.markers) == 0:
            return None, None
        else:
            points = []
            for point in self.img_crs_points:
                row_point, col_point = rowcol(tf, point.x(), point.y())
                points.append((col_point, row_point))

            points = np.array(points)
            labels = np.array(self.labels).astype(np.uint8)
            return points, labels


class ClickTool(QgsMapToolEmitPoint):
    """A tool to add points to canvas_points"""

    def __init__(
        self,
        canvas,
        canvas_points: Canvas_Points,
        prompt_type: str,
        prompt_history: List,
        execute_SAM: pyqtSignal,
    ):
        self.canvas = canvas
        self.canvas_points = canvas_points
        self.prompt_history = prompt_history
        if prompt_type not in ["fgpt", "bgpt", "bbox"]:
            raise ValueError(
                f"prompt_type must be one of ['fgpt', 'bgpt', 'bbox'], not {prompt_type}"
            )
        self.prompt_type = prompt_type
        self.execute_SAM = execute_SAM
        self.preview_mode: bool = False
        self.pressed: bool = False
        QgsMapToolEmitPoint.__init__(self, self.canvas)

        self.have_added_for_moving = False  # whether have added a point when mouse move
        if prompt_type == "fgpt":
            self.setCursor(CursorPointFG)
        elif prompt_type == "bgpt":
            self.setCursor(CursorPointBG)

    def reset_cursor_color(self, color):
        if self.prompt_type == "fgpt":
            Cursor_User = customize_fg_point_cursor(color)
        elif self.prompt_type == "bgpt":
            Cursor_User = customize_bg_point_cursor(color)
        self.setCursor(Cursor_User)

    def clear_hover_prompt(self):
        # remove the last rectangle if have added a rectangle when mouse move
        if self.have_added_for_moving:
            self.canvas_points.popPoint()
            self.have_added_for_moving = False  # reset to False

    def canvasPressEvent(self, e: QgsMapMouseEvent):
        self.pressed = True
        self.clear_hover_prompt()

        self.have_added_for_moving = False

        # add a point when mouse press
        point = self.toMapCoordinates(e.pos())
        if self.prompt_type == "fgpt":
            self.canvas_points.addPoint(point, foreground=True)
        elif self.prompt_type == "bgpt":
            self.canvas_points.addPoint(point, foreground=False)
        self.prompt_history.append(self.prompt_type)
        self.execute_SAM.emit()

    def canvasMoveEvent(self, e: QgsMapMouseEvent) -> None:
        if not self.preview_mode:
            return

        self.pressed = False
        self.clear_hover_prompt()

        # add a point when mouse move
        point = self.toMapCoordinates(e.pos())
        if self.prompt_type == "fgpt":
            self.canvas_points.addPoint(point, foreground=True, show=False)
        elif self.prompt_type == "bgpt":
            self.canvas_points.addPoint(point, foreground=False, show=False)
        self.execute_SAM.emit()
        self.have_added_for_moving = True

    def activate(self):
        QgsMapToolEmitPoint.activate(self)

    def deactivate(self):
        QgsMapToolEmitPoint.deactivate(self)

    def isZoomTool(self):
        return False

    def isTransient(self):
        return False

    def isEditTool(self):
        return True

    def edit(self):
        pass


class Canvas_SAM_Polygon:
    """A class to manage Rectangle on canvas."""

    def __init__(
        self,
        canvas: QgsMapCanvas,
        line_color=QColor(0, 255, 0),
        fill_color=QColor(0, 255, 0, 10),
        line_width=2,
    ):
        self.canvas = canvas
        self.geometry_list: List[QgsGeometry] = []
        self.rubber_band_list: List[QgsRubberBand] = []
        self.line_color = line_color
        self.fill_color = fill_color
        self.line_width = line_width

    def new_rubber_band(self):
        rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
        self.set_layer_style(
            rubber_band, self.fill_color, self.line_color, self.line_width
        )
        return rubber_band

    def set_layer_style(
        self, rubber_band, fill_color, line_color, line_width, line_color_2=None
    ):
        """Set the style of the rectangle layer"""
        if fill_color is not None:
            rubber_band.setFillColor(fill_color)
        if line_color is not None:
            rubber_band.setStrokeColor(line_color)
        if line_color_2 is not None:
            rubber_band.setSecondaryStrokeColor(line_color_2)
        if line_width is not None:
            rubber_band.setWidth(line_width)

    def clear(self):
        """Clear the rectangle on canvas"""
        while True:
            if len(self.rubber_band_list) > 0:
                self.popPolygon()
            else:
                break

    def addPolygon(self, geometry: QgsGeometry):
        rubber_band = self.new_rubber_band()
        rubber_band.setToGeometry(geometry, None)
        self.rubber_band_list.append(rubber_band)
        self.geometry_list.append(geometry)

    def popPolygon(self):
        if len(self.rubber_band_list) > 0:
            rubber_band = self.rubber_band_list.pop()
            self.canvas.scene().removeItem(rubber_band)
            self.canvas.refresh()
            self.geometry_list.pop()

    def set_line_style(self, color: QColor, line_width: int = 2):
        if color is None:
            return None

        self.line_color = color
        color_fill = list(color.getRgb())
        color_fill[-1] = 10
        # self.color_fill = QColor(*color_fill)
        self.color_fill = QColor(color.red(), color.green(), color.blue(), 10)
        for rubber_band in self.rubber_band_list:
            rubber_band.setStrokeColor(self.line_color)
            rubber_band.setFillColor(self.color_fill)
            rubber_band.setWidth(line_width)


class SAM_PolygonFeature:
    """A polygon feature for SAM output"""

    def __init__(
        self,
        img_crs_manager: ImageCRSManager,
        shapefile: str | Path | None = None,
        layer: QgsVectorLayer | None = None,
        default_name: str = "polygon_sam",
        kwargs_preview_polygon: Dict = {},
        kwargs_prompt_polygon: Dict = {},
        overwrite: bool = False,
    ):
        self.qgis_project = QgsProject.instance()
        self.img_crs_manager = img_crs_manager
        self.shapefile = shapefile
        self.default_name = default_name
        self.canvas = iface.mapCanvas()
        self.canvas_preview_polygon = Canvas_SAM_Polygon(
            self.canvas, **kwargs_preview_polygon
        )
        self.canvas_prompt_polygon = Canvas_SAM_Polygon(
            self.canvas, **kwargs_prompt_polygon
        )
        self.overwrite = overwrite
        self.geojson_canvas_preview: Dict = {}
        self.geojson_canvas_prompt: Dict = {}
        self.geojson_layer: Dict = {}
        if layer is not None:
            self.reset_layer(layer)
        else:
            self.init_layer()

    def init_layer(self):
        if self.shapefile:
            self._load_shapefile(self.shapefile)
        else:
            self._init_layer()

    @property
    def layer_name(self):
        try:
            return self.layer.name()
        except:
            return self.default_name

    @property
    def layer_id(self):
        try:
            return self.layer.id()
        except:
            return None

    def reset_geojson(self):
        self.geojson_canvas_preview = {}
        self.geojson_canvas_prompt = {}

    def reset_layer(self, layer: QgsVectorLayer) -> bool:
        """Reset the layer to a new layer

        Parameters:
        ----------
        layer: QgsVectorLayer
            the new layer to reset

        Returns:
        -------
        bool: whether reset successfully
        """
        if layer:
            self.layer = layer
        else:
            self._init_layer()

        self.reset_geojson()
        # User may want to keep their own style, so not rerender the layer
        # self.render_layer()
        self.ensure_edit_mode()
        return True

    def _load_shapefile(self, shapefile):
        """Load the shapefile to the layer."""
        if isinstance(shapefile, Path):
            shapefile = str(shapefile)
        # if default name without suffix, using the shapefile suffix
        if Path(shapefile).stem == Path(shapefile).name:
            shapefile = shapefile + ".shp"

        # if file not exists, create a new one into disk
        if not os.path.exists(shapefile) or self.overwrite:
            save_options = QgsVectorFileWriter.SaveVectorOptions()
            suffix = Path(shapefile).suffix
            save_options.driverName = SuffixDriverMap[suffix]
            save_options.fileEncoding = "UTF-8"
            transform_context = QgsProject.instance().transformContext()

            writer = QgsVectorFileWriter.create(
                shapefile,
                SAM_Feature_QgsFields,
                QgsWkbTypes.Polygon,
                self.img_crs_manager.img_crs,
                transform_context,
                save_options,
            )

            # delete the writer to flush features to disk
            del writer

        layer = QgsVectorLayer(shapefile, Path(shapefile).stem, "ogr")

        self.layer = layer
        # self.layer.updateFields()
        self.render_layer()
        self.ensure_edit_mode()

    def _init_layer(
        self,
    ):
        """Initialize the layer. If the layer exists, load it. If not, create a new one on memory"""
        layer_list = QgsProject.instance().mapLayersByName(self.default_name)
        if layer_list:
            self.layer = layer_list[0]
            self.commit_changes()
            MessageTool.MessageBar(
                "Note:",
                f"Using vector layer: '{self.layer_name}' to store the output polygons.",
                duration=30,
            )
        else:
            self.layer = QgsVectorLayer("Polygon", self.default_name, "memory")
            # self.layer.setCrs(self.qgis_project.crs())
            self.layer.setCrs(self.img_crs_manager.img_crs)
            self.render_layer()

            MessageTool.MessageBar(
                "Note:",
                "Output Shapefile is not specified. "
                f"A temporal layer: '{self.layer_name}' is created, "
                "remember to save it before quit.",
                duration=30,
            )
            # TODO: if field exists, whether need to add it again?
            # Set the provider to accept the data source
            # change by Joey, if layer exist keep the layer untouched.
            prov = self.layer.dataProvider()
            prov.addAttributes(SAM_Feature_Fields)
            self.layer.updateFields()

        self.ensure_edit_mode()

    def render_layer(self):
        """Render the SAM output layer on canvas"""
        self.qgis_project.addMapLayer(self.layer)
        self.layer.startEditing()
        symbol = QgsFillSymbol.createSimple(
            {"color": "0,255,0,40", "color_border": "green", "width_border": "0.6"}
        )
        render = self.layer.renderer()
        if render is not None:
            render.setSymbol(symbol)
        self.layer.triggerRepaint()

    def add_geojson_feature_to_canvas(
        self,
        geojson: Dict,
        selector: "Selector",
        target: str = "preview",
        overwrite_geojson: bool = False,
    ) -> None:
        """Add a geojson feature to the layer and update the status labels

        Parameters:
        ----------
        geojson: Dict
            features in geojson format
        selector: Selector
            The Geo SAM Selector object
        target: str, one of ['preview', 'prompt']
            the target of the geojson, default 'preview'
        overwrite_geojson: bool
            whether overwrite the geojson of this class.
            False for showing geometry greater than t_area.
            True for showing new SAM result.
        """
        areas = []
        geometries = []
        for geom in geojson:
            points_project_crs = []
            points_layer_crs = []
            coordinates = geom["geometry"]["coordinates"][0]
            for coord in coordinates:
                # transform pointXY from img_crs to polygon layer crs, if not match
                point = QgsPointXY(*coord)

                # show the point on canvas in project crs
                pt_project_crs = self.img_crs_manager.img_point_to_crs(
                    point, self.qgis_project.crs()
                )
                points_project_crs.append(pt_project_crs)

                # calculate the area in layer crs
                pt_layer_crs = self.img_crs_manager.img_point_to_crs(
                    point, self.layer.crs()
                )
                points_layer_crs.append(pt_layer_crs)

            geometry = QgsGeometry.fromPolygonXY([points_project_crs])
            geometries.append(geometry)

            geometry_layer_crs = QgsGeometry.fromPolygonXY([points_layer_crs])
            areas.append(geometry_layer_crs.area())
            # geometry_area = feature.geometry().area()

        def process_geometry():
            if target == "preview":
                self.canvas_preview_polygon.addPolygon(geometry)
                if overwrite_geojson:
                    self.geojson_canvas_preview = geojson
            else:
                self.canvas_prompt_polygon.addPolygon(geometry)
                if overwrite_geojson:
                    self.geojson_canvas_prompt = geojson

        if selector.max_polygon_mode:
            if len(areas) == 0:
                return None
            idx = np.argmax(areas)
            geometry = geometries[idx]
            process_geometry()
            num_polygon = 1
        else:
            num_polygon = 0
            for geometry, geometry_area in zip(geometries, areas):
                # if the area of the feature is less than t_area,
                # it will not be added to the layer
                if geometry_area < selector.t_area:
                    continue
                process_geometry()
                num_polygon += 1

        selector.update_status_labels(num_polygon)

    def add_geojson_feature_to_layer(
        self,
        geojson: Dict,
        t_area: float = 0,
        prompt_history: List = [],
        max_polygon_mode: bool = False,
        overwrite_geojson: bool = False,
    ):
        """Add a geojson feature to the layer

        Parameters:
        ----------
        geojson: Dict
            features in geojson format
        t_area: float
            the threshold of area
        prompt_history: List
            the prompt history of the feature
        max_polygon_mode: bool
            whether only keep the max object in the geojson
        overwrite_geojson: bool
            whether overwrite the geojson of this class.
            False for showing geometry greater than t_area.
        """
        if overwrite_geojson:
            self.geojson_layer = geojson

        features = []
        num_polygons = self.layer.featureCount()

        group_ulid = GroupId().ulid
        areas = []
        geometries = []
        for idx, geom in enumerate(geojson):
            points = []
            coordinates = geom["geometry"]["coordinates"][0]
            for coord in coordinates:
                # transform pointXY from img_crs to polygon layer crs, if not match
                point = QgsPointXY(*coord)
                point = self.img_crs_manager.img_point_to_crs(point, self.layer.crs())
                points.append(point)

            geometry = QgsGeometry.fromPolygonXY([points])

            geometries.append(geometry)
            areas.append(geometry.area())

        def process_geometry(idx, geometry, geometry_area):
            # add geometry to canvas_preview_polygon
            self.canvas_preview_polygon.addPolygon(geometry)

            # add geometry to shapefile
            # feature = QgsFeature()
            # feature.setGeometry(geometry)
            sam_attribute_list = [
                group_ulid,
                0,
                num_polygons + idx + 1,
                geometry_area,
                prompt_history.count("fgpt"),
                prompt_history.count("bgpt"),
                "bbox" in prompt_history,
            ]
            # attributes_list_write = [SAM_Feature_QgsFields.names().index(field) for field in self.layer.fields().names()]
            attributes_to_write = {}
            for field, attribute in zip(
                SAM_Feature_QgsFields.names(), sam_attribute_list
            ):
                # 0-based column indexing. The field index if found or -1 in case it cannot be found.
                idx = self.layer.fields().indexOf(field)
                if idx > -1:
                    attributes_to_write[idx] = attribute
            # Creates a new feature ready for insertion into a layer. Default values and constraints (e.g., unique constraints) will automatically be handled.
            feature = QgsVectorLayerUtils.createFeature(
                layer=self.layer, geometry=geometry, attributes=attributes_to_write
            )

            # feature.setAttributes(sam_attribute_list)
            features.append(feature)

        if max_polygon_mode:
            if len(areas) == 0:
                return None
            idx_max = np.argmax(areas)
            geometry = geometries[idx_max]
            geometry_area = areas[idx_max]
            process_geometry(0, geometry, geometry_area)
        else:
            for idx, (geometry, geometry_area) in enumerate(zip(geometries, areas)):
                if geometry_area < t_area:
                    continue
                process_geometry(idx, geometry, geometry_area)

        idx_n_gm = self.layer.fields().indexOf("N_GM")
        if idx_n_gm > -1:
            for feature in features:
                feature[idx_n_gm] = len(features)
        self.ensure_edit_mode()
        self.layer.addFeatures(features)
        self.layer.updateExtents()
        self.layer.triggerRepaint()

    def ensure_edit_mode(self):
        """Ensure the layer is in edit mode"""
        if not self.layer.isEditable():
            self.layer.startEditing()

    def clear_canvas_polygons(self):
        """Clear the polygon on canvas"""
        self.canvas_preview_polygon.clear()
        self.canvas_prompt_polygon.clear()
        self.canvas.refresh()

    # def rollback_changes(self):
    #     '''Roll back the changes'''
    #     try:
    #         # self.layer.rollBack()
    #         self.clear_canvas_polygons()
    #         self.canvas.refresh()
    #     except:
    #         return None

    def commit_changes(self, stop_edit: bool = False):
        """Commit the changes"""
        self.layer.commitChanges(stopEditing=stop_edit)
