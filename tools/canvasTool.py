from typing import List, Any, Dict
import os
import uuid
from PyQt5 import QtGui
import numpy as np
from pathlib import Path
from rasterio.transform import rowcol, Affine
from qgis.core import QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsVectorFileWriter, QgsRectangle, Qgis, QgsMessageLog
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand, QgsMapTool, QgsMapToolPan, QgsVertexMarker, QgsMapCanvas
from qgis.core import (
    QgsPointXY, QgsWkbTypes, QgsMarkerSymbol,  QgsField, QgsFields, QgsFillSymbol, QgsApplication,
    QgsGeometry, QgsFeature, QgsVectorLayer)
from qgis.PyQt.QtCore import QVariant
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QKeySequence, QIcon, QColor, QCursor, QBitmap, QPixmap
from qgis.utils import iface
from .geoTool import ImageCRSManager, LayerExtent
from ..ui.cursors import CursorPointBlue, CursorPointRed, CursorRect, UI_SCALE


class Canvas_Rectangle:
    '''A class to manage Rectangle on canvas.'''

    def __init__(self, canvas: QgsMapCanvas, img_crs_manager: ImageCRSManager, use_type='bbox'):
        self.canvas = canvas
        self.qgis_project = QgsProject.instance()
        self.rect_list = []
        # self.box_geo = None
        self.img_crs_manager = img_crs_manager
        self.rubberBand = QgsRubberBand(
            self.canvas, QgsWkbTypes.PolygonGeometry)

        if use_type == 'bbox':
            self._init_bbox_layer()
        elif use_type == 'extent':
            self._init_extent_layer()

    def _init_bbox_layer(self):
        '''Initialize the rectangle layer for bbox prompt'''
        fill_color = QColor(0, 0, 255, 10)
        line_color = Qt.blue
        line_width = 1
        self.set_layer_style(fill_color, line_color, line_width)

    def _init_extent_layer(self):
        '''Initialize the rectangle layer for extent of features'''
        fill_color = QColor(0, 0, 0, 0)
        line_color = QColor(255, 0, 0)
        # line_color2 = QColor(255, 255, 255)
        line_color2 = None  # not set secondary color currently
        line_width = 2
        self.set_layer_style(fill_color, line_color, line_width, line_color2)

    def set_layer_style(self, fill_color, line_color, line_width, line_color_2=None):
        '''Set the style of the rectangle layer'''
        if fill_color is not None:
            self.rubberBand.setFillColor(fill_color)
        if line_color is not None:
            self.rubberBand.setStrokeColor(line_color)
        if line_color_2 is not None:
            self.rubberBand.setSecondaryStrokeColor(line_color_2)
        if line_width is not None:
            self.rubberBand.setWidth(line_width)

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

    def __init__(self, canvas_rect: Canvas_Rectangle, prompt_history: List[Any], execute_SAM, img_crs_manager: ImageCRSManager):
        self.qgis_project = QgsProject.instance()
        self.canvas_rect = canvas_rect
        self.prompt_history = prompt_history
        self.execute_SAM = execute_SAM
        self.img_crs_manager = img_crs_manager
        QgsMapToolEmitPoint.__init__(self, self.canvas_rect.canvas)
        self.setCursor(CursorRect)

        self.reset()

    def reset(self):
        self.startPoint = self.endPoint = None
        self.isEmittingPoint = False
        self.canvas_rect.rubberBand.reset(QgsWkbTypes.PolygonGeometry)

    def canvasPressEvent(self, e):
        self.startPoint = self.toMapCoordinates(e.pos())
        self.endPoint = self.startPoint
        self.isEmittingPoint = True
        self.canvas_rect.showRect(self.startPoint, self.endPoint)

    def canvasReleaseEvent(self, e):
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

    def canvasMoveEvent(self, e):
        if not self.isEmittingPoint:
            return

        self.endPoint = self.toMapCoordinates(e.pos())
        self.canvas_rect.showRect(self.startPoint, self.endPoint)

    def deactivate(self):
        QgsMapTool.deactivate(self)
        self.deactivated.emit()


class Canvas_Extent:
    '''A class to manage feature Extent on canvas.'''

    def __init__(self, canvas: QgsMapCanvas, img_crs_manager: ImageCRSManager) -> None:
        self.canvas = canvas
        self.img_crs_manager = img_crs_manager

        self.canvas_rect_list: List[Canvas_Rectangle] = []

    def clear(self):
        '''Clear all extents on canvas'''
        for canvas_rect in self.canvas_rect_list:
            canvas_rect.clear()
        self.canvas_rect_list = []

    def add_extent(self, extent: QgsRectangle):
        '''Add a extent on canvas'''
        xMin, yMin, xMax, yMax = extent.xMinimum(
        ), extent.yMinimum(), extent.xMaximum(), extent.yMaximum()
        canvas_rect = Canvas_Rectangle(
            self.canvas, self.img_crs_manager, use_type='extent')

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

    @property
    def project_crs(self):
        return QgsProject.instance().crs()

    def addPoint(self, point: QgsPointXY, foreground: bool):
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
        if foreground:
            m.setColor(QColor(0, 0, 255))
            m.setFillColor(QColor(0, 0, 255))
        else:
            m.setColor(QColor(255, 0, 0))
            m.setFillColor(QColor(255, 0, 0))
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
        QgsMapToolEmitPoint.__init__(self, self.canvas)

        if prompt_type == "fgpt":
            self.setCursor(CursorPointBlue)
        elif prompt_type == "bgpt":
            self.setCursor(CursorPointRed)

    def canvasPressEvent(self, event):
        point = self.toMapCoordinates(event.pos())
        if self.prompt_type == "fgpt":
            self.canvas_points.addPoint(point, foreground=True)
        elif self.prompt_type == "bgpt":
            self.canvas_points.addPoint(point, foreground=False)
        self.prompt_history.append(self.prompt_type)
        self.execute_SAM.emit()

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


class SAM_PolygonFeature:
    '''A polygon feature for SAM output'''

    def __init__(self, img_crs_manager: ImageCRSManager, shapefile=None):
        self.qgis_project = QgsProject.instance()
        self.img_crs_manager = img_crs_manager
        self.shapefile = shapefile
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
            return "polygon_sam"

    @property
    def layer_id(self):
        try:
            return self.layer.id()
        except:
            return None

    def _load_shapefile(self, shapefile):
        '''Load the shapefile to the layer.'''
        if isinstance(shapefile, Path):
            shapefile = str(shapefile)
        if Path(shapefile).suffix.lower() != ".shp":
            shapefile = shapefile + ".shp"

        # if file not exists, create a new one into disk
        if not os.path.exists(shapefile):
            fields = QgsFields()
            fields.extend(
                [QgsField("Group_uuid", QVariant.String),
                 QgsField("id", QVariant.Int),
                 QgsField("Area", QVariant.Double),
                 QgsField("N_FG", QVariant.Int),
                 QgsField("N_BG", QVariant.Int),
                 QgsField("BBox", QVariant.Bool)]
            )

            save_options = QgsVectorFileWriter.SaveVectorOptions()
            save_options.driverName = "ESRI Shapefile"
            save_options.fileEncoding = "UTF-8"
            transform_context = QgsProject.instance().transformContext()

            writer = QgsVectorFileWriter.create(
                shapefile,
                fields,
                QgsWkbTypes.Polygon,
                self.img_crs_manager.img_crs,
                transform_context,
                save_options
            )

            # delete the writer to flush features to disk
            del writer

        self.layer = QgsVectorLayer(shapefile, Path(shapefile).stem, "ogr")
        self.show_layer()
        self.ensure_edit_mode()

    def _init_layer(self,):
        '''Initialize the layer. If the layer exists, load it. If not, create a new one on memory'''
        layer_list = QgsProject.instance().mapLayersByName(self.layer_name)
        if layer_list:
            self.layer = layer_list[0]
            self.layer.commitChanges()
        else:
            iface.messageBar().pushMessage(
                "Note:",
                "Output Shapefile is not specified. "
                "A temporal layer 'Polygon_sam' is created, "
                "remember to save it before quit.",
                level=Qgis.Info,
                duration=30)
            self.layer = QgsVectorLayer('Polygon', 'polygon_sam', 'memory')
            # self.layer.setCrs(self.qgis_project.crs())
            self.layer.setCrs(self.img_crs_manager.img_crs)
            self.show_layer()

        # TODO: if field exists, whether need to add it again?
        # Set the provider to accept the data source
        prov = self.layer.dataProvider()
        prov.addAttributes(
            [QgsField("Group_uuid", QVariant.String),
             QgsField("id", QVariant.Int),
             QgsField("Area", QVariant.Double),
             QgsField("N_FG", QVariant.Int),
             QgsField("N_BG", QVariant.Int),
             QgsField("BBox", QVariant.Bool)
             ]
        )
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

    def add_geojson_feature(self, geojson: Dict,
                            prompt_history: List,
                            t_area: float = 0):
        '''Add a geojson feature to the layer'''
        features = []
        num_polygons = self.layer.featureCount()
        group_uuid = str(uuid.uuid4())
        for idx, geom in enumerate(geojson):
            points = []
            coordinates = geom['geometry']['coordinates'][0]
            for coord in coordinates:
                # transform pointXY from img_crs to polygon layer crs, if not match
                point = QgsPointXY(*coord)
                point = self.img_crs_manager.img_point_to_crs(
                    point, self.layer.crs())
                points.append(point)

            # Add a new feature and assign the geometry
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPolygonXY([points]))
            ft_area = feature.geometry().area()
            if ft_area < t_area:
                return None

            feature.setAttributes(
                [group_uuid,
                 num_polygons+idx+1,
                 ft_area,
                 prompt_history.count('fgpt'),
                 prompt_history.count('bgpt'),
                 'bbox' in prompt_history]
            )
            features.append(feature)

        self.ensure_edit_mode()
        self.layer.addFeatures(features)
        self.layer.updateExtents()

    def ensure_edit_mode(self):
        '''Ensure the layer is in edit mode'''
        if not self.layer.isEditable():
            self.layer.startEditing()

    def rollback_changes(self):
        '''Roll back the changes'''
        try:
            self.layer.rollBack()
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

    def ensure_exist(self):
        layer_list = QgsProject.instance().mapLayersByName(self.layer_name)
        if not layer_list:
            self.init_layer()
