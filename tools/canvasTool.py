from typing import List, Any, Dict
import os
import numpy as np
from pathlib import Path
from rasterio.transform import rowcol, Affine
from qgis._gui import QgsMapMouseEvent
from qgis.core import QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsVectorFileWriter, QgsRectangle
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand, QgsMapTool, QgsVertexMarker, QgsMapCanvas
from qgis.core import (
    QgsPointXY, QgsWkbTypes, QgsField, QgsFields, QgsFillSymbol,
    QgsGeometry, QgsFeature, QgsVectorLayer)
from qgis.PyQt.QtCore import QVariant
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from qgis.utils import iface
from .geoTool import ImageCRSManager, LayerExtent
from .ulid import GroupId
from ..ui.cursors import (
    CursorPointFG,
    CursorPointBG,
    CursorRect,
    UI_SCALE,
    customize_fg_point_cursor,
    customize_bg_point_cursor,
    customize_bbox_cursor
)
from .messageTool import MessageTool

SAM_Feature_Fields = [
    QgsField("group_ulid", QVariant.String),
    QgsField("N_GM", QVariant.Int),
    QgsField("id", QVariant.Int),
    QgsField("Area", QVariant.Double),
    QgsField("N_FG", QVariant.Int),
    QgsField("N_BG", QVariant.Int),
    QgsField("BBox", QVariant.Bool)
]
SAM_Feature_QgsFields = QgsFields()
new_fields = QgsFields()
for field in SAM_Feature_Fields:
    SAM_Feature_QgsFields.append(field)


class Canvas_Rectangle:
    '''A class to manage Rectangle on canvas.'''

    def __init__(
        self,
        canvas: QgsMapCanvas,
        img_crs_manager: ImageCRSManager,
        use_type: str = 'bbox',
        alpha: int = 255,
        line_width=None

    ):
        self.canvas = canvas
        self.qgis_project = QgsProject.instance()
        self.rect_list = []
        # self.box_geo = None
        self.img_crs_manager = img_crs_manager
        self.alpha = alpha
        self.line_width = line_width
        self.rubberBand = QgsRubberBand(
            self.canvas, QgsWkbTypes.PolygonGeometry)

        self.colors_bbox = {
            'fill_color': QColor(0, 0, 255, 10),
            'line_color': Qt.blue
        }
        self.colors_extent = {
            'fill_color': QColor(0, 0, 0, 0),
            'line_color': QColor(255, 0, 0)
        }

        if use_type == 'bbox':
            self._init_bbox_layer()
        elif use_type == 'extent':
            self._init_extent_layer()
        elif use_type == 'patch_extent':
            self._init_batch_extent_layer()

    def flush_rect_color(self):
        '''Flush the color of rectangle'''
        if len(self.rect_list) == 0:
            return None
        else:
            startPoint, endPoint = self.rect_list[-1]
            self.showRect(startPoint, endPoint)

    def _init_bbox_layer(self):
        '''Initialize the rectangle layer for bbox prompt'''
        line_width = 1
        self.set_layer_style(
            self.colors_bbox["fill_color"],
            self.colors_bbox["line_color"],
            line_width
        )

    def _init_extent_layer(self):
        '''Initialize the rectangle layer for extent of features'''
        # line_color2 = QColor(255, 255, 255)
        line_color2 = None  # not set secondary color currently
        line_width = 3
        self.set_layer_style(
            self.colors_extent["fill_color"],
            self.colors_extent["line_color"],
            line_width,
            line_color2
        )

    def _init_batch_extent_layer(self):
        '''Initialize the rectangle layer for batch extent'''
        color_random = np.random.randint(0, 255, size=3).tolist()
        fill_color = QColor(0, 0, 0, 0)
        line_color = QColor(
            color_random[0],
            color_random[1],
            color_random[2],
            self.alpha
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
        '''Set the style of the rectangle layer'''
        self.set_fill_color(fill_color)
        self.set_line_color(line_color)
        self.set_line_color_2(line_color_2)
        self.set_line_width(line_width)

    def clear(self):
        '''Clear the rectangle on canvas'''
        self.rubberBand.reset(QgsWkbTypes.PolygonGeometry)
        self.canvas.refresh()
        self.rect_list.clear()

    def addRect(self, startPoint: QgsPointXY, endPoint: QgsPointXY):
        self.rect_list.append((startPoint, endPoint))

    def popRect(self):
        if len(self.rect_list) > 0:
            self.rect_list.pop()
            if len(self.rect_list) > 0:
                startPoint, endPoint = self.rect_list[-1]
                self.showRect(startPoint, endPoint)
            else:
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
        self.rubberBand.addPoint(point4, True)    # true to update canvas
        self.rubberBand.show()

    @property
    def box_geo(self):
        '''Returns a rectangle from two points with img crs'''
        if len(self.rect_list) == 0:
            return None
        else:
            # startPoint endPoint transform
            startPoint, endPoint = self.rect_list[-1]
            startPoint = self.img_crs_manager.point_to_img_crs(
                startPoint, self.qgis_project.crs())
            endPoint = self.img_crs_manager.point_to_img_crs(
                endPoint, self.qgis_project.crs())
            return [startPoint.x(), startPoint.y(), endPoint.x(), endPoint.y()]

    @property
    def extent(self):
        '''Return the extent of the rectangle (minX, maxX. minY, maxY)'''
        if self.box_geo is not None:
            extent = [
                min(self.box_geo[0], self.box_geo[2]),
                max(self.box_geo[0], self.box_geo[2]),
                min(self.box_geo[1], self.box_geo[3]),
                max(self.box_geo[1], self.box_geo[3])
            ]
            return extent
        else:
            return None

    def get_img_box(self, transform):
        '''Return the box for SAM image'''
        if self.box_geo is not None:
            rowcol1 = rowcol(transform, self.box_geo[0], self.box_geo[1])
            rowcol2 = rowcol(transform, self.box_geo[2], self.box_geo[3])
            box = [
                min(rowcol1[1], rowcol2[1]),
                min(rowcol1[0], rowcol2[0]),
                max(rowcol1[1], rowcol2[1]),
                max(rowcol1[0], rowcol2[0])
            ]
            return np.array(box)
        else:
            return None


class RectangleMapTool(QgsMapToolEmitPoint):
    '''A map tool to draw a rectangle on canvas'''

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
        self.hover_mode: bool = False
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
        if self.startPoint is None or self.endPoint is None:
            return None
        elif (self.startPoint.x() == self.endPoint.x() or
              self.startPoint.y() == self.endPoint.y()):
            return None
        else:
            self.canvas_rect.addRect(self.startPoint, self.endPoint)
            self.prompt_history.append('bbox')
            self.execute_SAM.emit()
            self.have_added_for_moving = False  # reset to False

    def clear_hover_prompt(self):
        # remove the last rectangle if have added a rectangle when mouse move
        if self.have_added_for_moving:
            self.canvas_rect.popRect()
            self.have_added_for_moving = False  # reset to False

    def canvasMoveEvent(self, e):
        if not self.isEmittingPoint:
            return

        # update the rectangle as the mouse moves
        self.endPoint = self.toMapCoordinates(e.pos())
        self.canvas_rect.showRect(self.startPoint, self.endPoint)

        # execute SAM when mouse move
        if not self.hover_mode:
            return
        if self.startPoint is None or self.endPoint is None:
            return None
        elif (self.startPoint.x() == self.endPoint.x() or
              self.startPoint.y() == self.endPoint.y()):
            return None
        else:
            self.canvas_rect.popRect()
            self.canvas_rect.addRect(self.startPoint, self.endPoint)
            self.execute_SAM.emit()
            self.have_added_for_moving = True

    def deactivate(self):
        QgsMapTool.deactivate(self)
        self.deactivated.emit()


class Canvas_Extent:
    '''A class to manage feature Extent on canvas.'''

    def __init__(self, canvas: QgsMapCanvas, img_crs_manager: ImageCRSManager) -> None:
        self.canvas = canvas
        self.img_crs_manager = img_crs_manager
        self.canvas_rect_list: List[Canvas_Rectangle] = []
        self.color = None

    def clear(self):
        '''Clear all extents on canvas'''
        for canvas_rect in self.canvas_rect_list:
            canvas_rect.clear()
        self.canvas_rect_list = []

    def set_color(self, color: QColor):
        '''Set the color of the extent'''
        self.color = color

    def add_extent(self, extent: QgsRectangle,
                   use_type: str = 'extent',
                   alpha: int = 255,
                   line_width=None
                   ):
        '''Add a extent on canvas'''
        xMin, yMin, xMax, yMax = extent.xMinimum(
        ), extent.yMinimum(), extent.xMaximum(), extent.yMaximum()
        canvas_rect = Canvas_Rectangle(
            self.canvas,
            self.img_crs_manager,
            use_type=use_type,
            alpha=alpha,
            line_width=line_width
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
        '''
        Parameters:
        ----------
        canvas: QgsMapCanvas
            canvas to add points
        img_crs_manager: ImageCRSManager
            The manager to transform points between image crs and other crs
        '''
        self.canvas = canvas
        self.extent = None
        self.img_crs_manager = img_crs_manager
        self.markers: List[QgsVertexMarker] = []
        self.points_img_crs: List[QgsPointXY] = []
        self.labels: List[bool] = []
        self.foreground_color = QColor(0, 0, 255)
        self.background_color = QColor(255, 0, 0)

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
            m.setIconSize(round(UI_SCALE/3))
            m.setIconType(QgsVertexMarker.ICON_CIRCLE)

        # add to markers and labels
        self.markers.append(m)
        point_img_crs = self.img_crs_manager.point_to_img_crs(
            point, self.project_crs)
        self.points_img_crs.append(point_img_crs)
        self.labels.append(foreground)

        self._update_extent()

    def flush_points_color(self):
        '''Flush the color of points'''
        for i, m in enumerate(self.markers):
            if self.labels[i]:
                m.setColor(self.foreground_color)
                m.setFillColor(self.foreground_color)
            else:
                m.setColor(self.background_color)
                m.setFillColor(self.background_color)

    def popPoint(self):
        """remove the last marker"""
        if len(self.markers) > 0:
            m = self.markers.pop()
            self.canvas.scene().removeItem(m)
            self.points_img_crs.pop()
            self.labels.pop()

            self._update_extent()

    def clear(self):
        """remove all markers"""
        for m in self.markers:
            self.canvas.scene().removeItem(m)
        self.canvas.refresh()

        self.markers = []
        self.points_img_crs = []
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
        '''Returns points and labels for SAM image'''
        if len(self.markers) == 0:
            return None, None
        else:
            points = []
            for point in self.points_img_crs:
                row_point, col_point = rowcol(tf, point.x(), point.y())
                points.append((col_point, row_point))

            points = np.array(points)
            labels = np.array(self.labels).astype(np.uint8)
            return points, labels


class ClickTool(QgsMapToolEmitPoint):
    '''A tool to add points to canvas_points'''

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
        self.hover_mode: bool = False
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
        if not self.hover_mode:
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
    '''A class to manage Rectangle on canvas.'''

    def __init__(
        self,
        canvas: QgsMapCanvas,
    ):
        self.canvas = canvas
        self.geometry_list: List[QgsGeometry] = []
        self.rubber_band_list: List[QgsRubberBand] = []

    def new_rubber_band(self):
        rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
        self.set_layer_style(rubber_band,
                             QColor(0, 255, 0, 10),
                             Qt.green,
                             1)
        return rubber_band

    def set_layer_style(self, rubber_band, fill_color, line_color, line_width, line_color_2=None):
        '''Set the style of the rectangle layer'''
        if fill_color is not None:
            rubber_band.setFillColor(fill_color)
        if line_color is not None:
            rubber_band.setStrokeColor(line_color)
        if line_color_2 is not None:
            rubber_band.setSecondaryStrokeColor(line_color_2)
        if line_width is not None:
            rubber_band.setWidth(line_width)

    def clear(self):
        '''Clear the rectangle on canvas'''
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


class SAM_PolygonFeature:
    '''A polygon feature for SAM output'''

    def __init__(
        self,
        img_crs_manager: ImageCRSManager,
        shapefile: str = None,
        layer: QgsVectorLayer = None,
        default_name: str = 'polygon_sam'
    ):
        self.qgis_project = QgsProject.instance()
        self.img_crs_manager = img_crs_manager
        self.shapefile = shapefile
        self.default_name = default_name
        self.canvas_polygon = Canvas_SAM_Polygon(iface.mapCanvas())
        self.geojson_canvas: Dict = {}
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

    def layer_fields_correct(self, layer: QgsVectorLayer):
        fields = layer.fields().names()
        MessageTool.MessageLog(
            f"New layer fields: {fields}")
        MessageTool.MessageLog(
            f"old feature fields: {SAM_Feature_QgsFields.names()}")

        if len(set(SAM_Feature_QgsFields.names()) - set(fields)) > 0:
            MessageTool.MessageBoxOK(
                'The fields of this vector do not match the SAM feature fields.'
                " Please select a correct existed file or a new file to create it."
            )
            return False
        else:
            return True

    def reset_layer(self, layer: QgsVectorLayer) -> bool:
        '''Reset the layer to a new layer

        Parameters:
        ----------
        layer: QgsVectorLayer
            the new layer to reset

        Returns:
        -------
        bool: whether reset successfully
        '''
        if layer:
            if not self.layer_fields_correct(layer):
                return False
            self.layer = layer
        else:
            self._init_layer()

        self.geojson_canvas = {}
        self.show_layer()
        self.ensure_edit_mode()
        return True

    def _load_shapefile(self, shapefile):
        '''Load the shapefile to the layer.'''
        if isinstance(shapefile, Path):
            shapefile = str(shapefile)
        if Path(shapefile).suffix.lower() != ".shp":
            shapefile = shapefile + ".shp"

        # if file not exists, create a new one into disk
        if not os.path.exists(shapefile):
            save_options = QgsVectorFileWriter.SaveVectorOptions()
            save_options.driverName = "ESRI Shapefile"
            save_options.fileEncoding = "UTF-8"
            transform_context = QgsProject.instance().transformContext()

            writer = QgsVectorFileWriter.create(
                shapefile,
                SAM_Feature_QgsFields,
                QgsWkbTypes.Polygon,
                self.img_crs_manager.img_crs,
                transform_context,
                save_options
            )

            # delete the writer to flush features to disk
            del writer

        layer = QgsVectorLayer(shapefile, Path(shapefile).stem, "ogr")
        if not self.layer_fields_correct(layer):
            return False

        self.layer = layer
        self.show_layer()
        self.ensure_edit_mode()

    def _init_layer(self,):
        '''Initialize the layer. If the layer exists, load it. If not, create a new one on memory'''
        layer_list = QgsProject.instance().mapLayersByName(self.layer_name)
        if layer_list:
            self.layer = layer_list[0]
            self.layer.commitChanges()
        else:
            MessageTool.MessageBar(
                "Note:",
                "Output Shapefile is not specified. "
                "A temporal layer 'polygon_sam' is created, "
                "remember to save it before quit.",
                duration=30
            )
            self.layer = QgsVectorLayer('Polygon', self.default_name, 'memory')
            # self.layer.setCrs(self.qgis_project.crs())
            self.layer.setCrs(self.img_crs_manager.img_crs)
            self.show_layer()

        # TODO: if field exists, whether need to add it again?
        # Set the provider to accept the data source
        prov = self.layer.dataProvider()
        prov.addAttributes(SAM_Feature_Fields)
        self.layer.updateFields()

        self.ensure_edit_mode()

    def show_layer(self):
        '''Show the layer on canvas'''
        self.qgis_project.addMapLayer(self.layer)
        self.layer.startEditing()
        symbol = QgsFillSymbol.createSimple({'color': '0,255,0,40',
                                            'color_border': 'green',
                                             'width_border': '0.6'})
        self.layer.renderer().setSymbol(symbol)
        # show the change
        self.layer.triggerRepaint()

    def add_geojson_feature_to_canvas(
            self,
            geojson: Dict,
            t_area: float,
            overwrite_geojson: bool = False
    ):
        '''Add a geojson feature to the layer

        Parameters:
        ----------
        geojson: Dict
            features in geojson format
        t_area: float
            the threshold of area
        overwrite_geojson: bool
            whether overwrite the geojson of this class. 
            False for showing geometry greater than t_area.
            True for showing new SAM result.
        '''
        if overwrite_geojson:
            self.geojson_canvas = geojson
        for geom in geojson:
            points = []
            coordinates = geom['geometry']['coordinates'][0]
            for coord in coordinates:
                # transform pointXY from img_crs to polygon layer crs, if not match
                point = QgsPointXY(*coord)
                point = self.img_crs_manager.img_point_to_crs(
                    point, self.layer.crs())
                points.append(point)

            geometry = QgsGeometry.fromPolygonXY([points])
            geometry_area = geometry.area()
            # geometry_area = feature.geometry().area()

            # if the area of the feature is less than t_area,
            # it will not be added to the layer
            if geometry_area < t_area:
                continue
            # add geometry to canvas_polygon
            self.canvas_polygon.addPolygon(geometry)

    def add_geojson_feature_to_layer(
            self,
            geojson: Dict,
            t_area: float = 0,
            prompt_history: List = [],
            overwrite_geojson: bool = False
    ):
        '''Add a geojson feature to the layer

        Parameters:
        ----------
        geojson: Dict
            features in geojson format
        t_area: float
            the threshold of area
        overwrite_geojson: bool
            whether overwrite the geojson of this class. 
            False for showing geometry greater than t_area.
        '''
        if overwrite_geojson:
            self.geojson_layer = geojson

        features = []
        num_polygons = self.layer.featureCount()

        group_ulid = GroupId().ulid
        for idx, geom in enumerate(geojson):
            points = []
            coordinates = geom['geometry']['coordinates'][0]
            for coord in coordinates:
                # transform pointXY from img_crs to polygon layer crs, if not match
                point = QgsPointXY(*coord)
                point = self.img_crs_manager.img_point_to_crs(
                    point, self.layer.crs())
                points.append(point)

            geometry = QgsGeometry.fromPolygonXY([points])
            geometry_area = geometry.area()
            # geometry_area = feature.geometry().area()

            # if the area of the feature is less than t_area,
            # it will not be added to the layer
            if geometry_area < t_area:
                continue
            # add geometry to canvas_polygon
            self.canvas_polygon.addPolygon(geometry)

            # add geometry to shapefile
            feature = QgsFeature()
            feature.setGeometry(geometry)
            feature.setAttributes(
                [group_ulid,
                 0,
                 num_polygons+idx+1,
                 geometry_area,
                 prompt_history.count('fgpt'),
                 prompt_history.count('bgpt'),
                 'bbox' in prompt_history]
            )
            features.append(feature)

        for feature in features:
            feature[1] = len(features)
        self.ensure_edit_mode()
        self.layer.addFeatures(features)
        self.layer.updateExtents()
        self.layer.triggerRepaint()

    def ensure_edit_mode(self):
        '''Ensure the layer is in edit mode'''
        if not self.layer.isEditable():
            self.layer.startEditing()

    def rollback_changes(self):
        '''Roll back the changes'''
        # self.layer.rollBack()
        # self.canvas_polygon.clear()
        try:
            self.layer.rollBack()
            self.canvas_polygon.clear()
        except:
            return None
        # except RuntimeError as inst:
        #     print(inst.args[0])
        #     if inst.args[0] == 'wrapped C/C++ object of type QgsVectorLayer has been deleted':
        #         self._init_layer()
        #         print('Polygon layer has been deleted, new polygon layer created')
        # except Exception as err:
        #     print(f"Unexpected {err=}, {type(err)=}")
        #     # self._init_layer()
        #     # print('Polygon layer has been deleted, new polygon layer created')
        #     raise

    def commit_changes(self):
        '''Commit the changes'''
        self.layer.commitChanges()
