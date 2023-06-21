from typing import List
import os
from PyQt5 import QtGui
import numpy as np
from pathlib import Path
from rasterio.transform import rowcol, Affine
from qgis.core import QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, Qgis, QgsMessageLog
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand, QgsMapTool, QgsMapToolPan, QgsVertexMarker
from qgis.core import (
    QgsPointXY, QgsWkbTypes, QgsMarkerSymbol,  QgsField, QgsFillSymbol, QgsApplication,
    QgsGeometry, QgsFeature, QgsVectorLayer)
from qgis.PyQt.QtCore import QVariant
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QKeySequence, QIcon, QColor, QCursor, QBitmap, QPixmap

from .geoTool import ImageCRSManager, LayerExtent
from ..ui.cursors import CursorPointBlue, CursorPointRed, CursorRect


class RectangleMapTool(QgsMapToolEmitPoint):
    '''A map tool to draw a rectangle on canvas'''

    def __init__(self, canvas_rect, prompts, execute_SAM, img_crs_manager: ImageCRSManager):
        self.qgis_project = QgsProject.instance()
        self.canvas_rect = canvas_rect
        self.prompts = prompts
        self.rubberBand = canvas_rect.rubberBand
        self.execute_SAM = execute_SAM
        self.img_crs_manager = img_crs_manager
        QgsMapToolEmitPoint.__init__(self, self.canvas_rect.canvas)
        self.setCursor(CursorRect)

        self.reset()

    def reset(self):
        self.startPoint = self.endPoint = None
        self.isEmittingPoint = False
        self.rubberBand.reset(QgsWkbTypes.PolygonGeometry)

    def canvasPressEvent(self, e):
        self.startPoint = self.toMapCoordinates(e.pos())
        self.endPoint = self.startPoint
        self.isEmittingPoint = True
        self.showRect(self.startPoint, self.endPoint)

    def canvasReleaseEvent(self, e):
        self.isEmittingPoint = False
        box_geo = self.rectangle()
        if box_geo is not None:
            self.canvas_rect.box_geo = box_geo
            self.prompts.append('bbox')
            self.execute_SAM.emit()

    def canvasMoveEvent(self, e):
        if not self.isEmittingPoint:
            return

        self.endPoint = self.toMapCoordinates(e.pos())
        self.showRect(self.startPoint, self.endPoint)

    def showRect(self, startPoint, endPoint):
        self.rubberBand.reset(QgsWkbTypes.PolygonGeometry)
        if startPoint.x() == endPoint.x() or startPoint.y() == endPoint.y():
            return

        point1 = QgsPointXY(startPoint.x(), startPoint.y())
        point2 = QgsPointXY(startPoint.x(), endPoint.y())
        point3 = QgsPointXY(endPoint.x(), endPoint.y())
        point4 = QgsPointXY(endPoint.x(), startPoint.y())

        self.rubberBand.addPoint(point1, False)
        self.rubberBand.addPoint(point2, False)
        self.rubberBand.addPoint(point3, False)
        self.rubberBand.addPoint(point4, True)    # true to update canvas
        self.rubberBand.show()

    def rectangle(self):
        '''Returns a rectangle from two points with img crs'''
        if self.startPoint is None or self.endPoint is None:
            return None
        elif (self.startPoint.x() == self.endPoint.x() or
              self.startPoint.y() == self.endPoint.y()):
            return None
        else:
            # startPoint endPoint transform
            if self.qgis_project.crs() != self.img_crs_manager.img_crs:
                self.startPoint = self.img_crs_manager.point_to_img_crs(
                    self.startPoint, self.qgis_project.crs())
                self.endPoint = self.img_crs_manager.point_to_img_crs(
                    self.endPoint, self.qgis_project.crs())
            return [self.startPoint.x(), self.startPoint.y(), self.endPoint.x(), self.endPoint.y()]

    def deactivate(self):
        QgsMapTool.deactivate(self)
        self.deactivated.emit()


class Canvas_Rectangle:
    '''A class to manage Rectangle on canvas.'''

    def __init__(self, canvas, img_crs_manager: ImageCRSManager):
        self.canvas = canvas
        self.qgis_project = QgsProject.instance()
        self.box_geo = None
        self.img_crs_manager = img_crs_manager
        self.rubberBand = QgsRubberBand(
            self.canvas, QgsWkbTypes.PolygonGeometry)
        self._init_rect_layer()

    def _init_rect_layer(self):
        '''Initialize the rectangle layer'''
        self.rubberBand.setColor(Qt.blue)
        self.rubberBand.setFillColor(QColor(0, 0, 255, 10))
        self.rubberBand.setWidth(1)

    def clear(self):
        '''Clear the rectangle on canvas'''
        self.rubberBand.reset(QgsWkbTypes.PolygonGeometry)
        self.canvas.refresh()
        self.box_geo = None

    @property
    def extent(self):
        '''Return the extent of the rectangle'''
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

    def get_img_box(self, tf):
        '''Return the box for SAM image'''
        if self.box_geo is not None:
            rowcol1 = rowcol(tf, self.box_geo[0], self.box_geo[1])
            rowcol2 = rowcol(tf, self.box_geo[2], self.box_geo[3])
            box = [
                min(rowcol1[1], rowcol2[1]),
                min(rowcol1[0], rowcol2[0]),
                max(rowcol1[1], rowcol2[1]),
                max(rowcol1[0], rowcol2[0])
            ]
            return np.array(box)
        else:
            return None


class Canvas_Points:
    """
    A class to manage points on canvas.
    """

    def __init__(self, canvas, img_crs_manager: ImageCRSManager):
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
        m.setIconSize(12)
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
        prompts: List,
        execute_SAM: pyqtSignal,
    ):

        self.canvas = canvas
        self.canvas_points = canvas_points
        self.prompts = prompts
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
        self.prompts.append(self.prompt_type)
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
        if shapefile:
            self._load_shapefile(shapefile)
        else:
            self._init_layer()

    def _load_shapefile(self, shapefile):
        '''Load the shapefile to the layer.'''
        if isinstance(shapefile, Path):
            shapefile = str(shapefile)
        self.layer = QgsVectorLayer(shapefile, "polygon_sam", "ogr")
        self.show_layer()
        self.ensure_edit_mode()

    def _init_layer(self,):
        '''Initialize the layer. If the layer exists, load it. If not, create a new one on memory'''
        layer_list = QgsProject.instance().mapLayersByName("polygon_sam")
        if layer_list:
            self.layer = layer_list[0]
            self.layer.commitChanges()
        else:
            self.layer = QgsVectorLayer('Polygon', 'polygon_sam', 'memory')
            # self.layer.setCrs(self.qgis_project.crs())
            self.layer.setCrs(self.img_crs_manager.img_crs)
        # Set the provider to accept the data source
        prov = self.layer.dataProvider()
        prov.addAttributes([QgsField("id", QVariant.Int),
                           QgsField("Area", QVariant.Double)])
        self.layer.updateFields()
        self.show_layer()
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

    def add_geojson_feature(self, geojson):
        '''Add a geojson feature to the layer'''
        features = []
        num_polygons = self.layer.featureCount()
        for idx, geom in enumerate(geojson):
            points = []
            coordinates = geom['geometry']['coordinates'][0]
            for coord in coordinates:
                # transform pointXY from img_crs to polygon layer crs, if not match
                point = QgsPointXY(*coord)
                if self.layer.crs() != self.img_crs_manager.img_crs:
                    point = self.img_crs_manager.img_point_to_crs(
                        point, self.layer.crs())
                points.append(point)

            # Add a new feature and assign the geometry
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPolygonXY([points]))
            feature.setAttributes(
                [num_polygons+idx+1, feature.geometry().area()])

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
        except RuntimeError as inst:
            print(inst.args[0])
            if inst.args[0] == 'wrapped C/C++ object of type QgsVectorLayer has been deleted':
                self._init_layer()
                print('Polygon layer has been deleted, new polygon layer created')
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            # self._init_layer()
            # print('Polygon layer has been deleted, new polygon layer created')
            raise

    def commit_changes(self):
        '''Commit the changes'''
        self.layer.commitChanges()
