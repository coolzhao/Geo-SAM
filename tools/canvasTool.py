import typing
import os
import numpy as np
from pathlib import Path
from rasterio.transform import rowcol
from qgis.core import QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand, QgsMapTool, QgsMapToolPan
from qgis.core import (
    QgsPointXY, QgsWkbTypes, QgsMarkerSymbol,  QgsField, QgsFillSymbol,
    QgsGeometry, QgsFeature, QgsVectorLayer)
from qgis.PyQt.QtCore import QVariant
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QIcon, QColor

from .geoTool import TransformCRS, LayerExtent


class RectangleMapTool(QgsMapToolEmitPoint):
    def __init__(self, canvas_rect, execute_SAM, transform_crs: TransformCRS):
        self.qgis_project = QgsProject.instance()
        self.canvas_rect = canvas_rect
        self.rubberBand = canvas_rect.rubberBand
        self.execute_SAM = execute_SAM
        self.transform_crs = transform_crs
        QgsMapToolEmitPoint.__init__(self, self.canvas_rect.canvas)

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
        '''Returns a rectangle from two points for SAM'''
        if self.startPoint is None or self.endPoint is None:
            return None
        elif (self.startPoint.x() == self.endPoint.x() or
              self.startPoint.y() == self.endPoint.y()):
            return None
        else:
            # TODO startPoint endPoint transform
            if self.qgis_project.crs() != self.transform_crs.feature_crs:
                self.startPoint = self.transform_crs.transform_point_to_feature_crs(
                    self.startPoint, self.qgis_project.crs())
                self.endPoint = self.transform_crs.transform_point_to_feature_crs(
                    self.endPoint, self.qgis_project.crs())
            return [self.startPoint.x(), self.startPoint.y(), self.endPoint.x(), self.endPoint.y()]

    def deactivate(self):
        QgsMapTool.deactivate(self)
        self.deactivated.emit()


class ClickTool(QgsMapToolEmitPoint):
    def __init__(self, canvas, feature, layer, execute_SAM):
        self.canvas = canvas
        self.feature = feature
        self.layer = layer
        self.execute_SAM = execute_SAM
        QgsMapToolEmitPoint.__init__(self, self.canvas)

    def canvasPressEvent(self, event):
        point = self.toMapCoordinates(event.pos())
        self.feature.setGeometry(QgsGeometry.fromPointXY(point))
        # self.layer.dataProvider().addFeatures([self.feature])
        self.layer.addFeatures([self.feature])  # add by zyzhao
        # self.layer.updateExtents() # add by zyzhao
        self.layer.triggerRepaint()
        self.execute_SAM.emit()

        # Convert all points to string and print to console
        points_str = ""
        for feature in self.layer.getFeatures():
            point = feature.geometry().asPoint()
            points_str += f"{point.x()}, {point.y()}\n"
        print(points_str)

    def activate(self):
        QgsProject.instance().addMapLayer(self.layer)
        if not self.layer.isEditable():  # add by zyzhao
            self.layer.startEditing()
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


class Canvas_Points:
    def __init__(self, canvas, transform_crs: TransformCRS):
        self.canvas = canvas
        self.qgis_project = QgsProject.instance()
        self.transform_crs = transform_crs

    def init_points_layer(self):
        """initialize the points layer"""
        self.layer_fg = self._init_points_layer("Foreground Points", "blue")
        self.layer_bg = self._init_points_layer("Background Points", "red")

        self.feature_fg = QgsFeature()
        self.feature_bg = QgsFeature()

    def _init_points_layer(self, layer_name, color):
        """find the layer with the given name, or create it if it doesn't exist"""
        layer_list = QgsProject.instance().mapLayersByName(layer_name)
        if layer_list:
            layer = layer_list[0]
            layer.commitChanges()
        else:
            layer = QgsVectorLayer("Point", layer_name, "memory")
            layer.setCrs(self.qgis_project.crs())
            # layer.setCrs(self.transform_crs.feature_crs)

            # set default color
            symbol = QgsMarkerSymbol.createSimple(
                {'name': 'circle', 'color': color})
            layer.renderer().setSymbol(symbol)
            layer.triggerRepaint()

        return layer

    def _del_layer(self, layer):
        """Delete the layer from the map canvas"""
        QgsProject.instance().removeMapLayer(layer.id())
        self.canvas.refresh()

    def _clear_features(self, layer):
        '''Clear all features from the layer'''
        # layer.commitChanges()
        # layer.startEditing()
        layer.deleteFeatures([f.id() for f in layer.getFeatures()])

    def _reset_points_layer(self):
        """Delete all points from the layer"""
        self._clear_features(self.layer_fg)
        self._clear_features(self.layer_bg)
        self.canvas.refresh()

    def get_points_and_labels(self, tf):
        '''Get the points and labels from the foreground and background layers'''
        if self.layer_fg.featureCount() == 0 and self.layer_bg.featureCount() == 0:
            return None, None
        else:
            points = []
            labels = []
            for feature in self.layer_fg.getFeatures():
                point = feature.geometry().asPoint()
                if self.layer_fg.crs() != self.transform_crs.feature_crs:
                    point = self.transform_crs.transform_point_to_feature_crs(
                        point, self.layer_fg.crs())
                row_point, col_point = rowcol(tf, point.x(), point.y())
                points.append((col_point, row_point))
                labels.append(1)

            for feature in self.layer_bg.getFeatures():
                point = feature.geometry().asPoint()
                if self.layer_bg.crs() != self.transform_crs.feature_crs:
                    point = self.transform_crs.transform_point_to_feature_crs(
                        point, self.layer_bg.crs())
                row_point, col_point = rowcol(tf, point.x(), point.y())
                points.append((col_point, row_point))
                labels.append(0)
            points, labels = np.array(points), np.array(labels)

            return points, labels

    @property
    def extent(self):
        e = LayerExtent.union_layer_extent(
            self.layer_fg, self.layer_bg, self.transform_crs)
        print(e)
        return e


class Canvas_Rectangle:
    def __init__(self, canvas, transform_crs: TransformCRS):
        self.canvas = canvas
        self.qgis_project = QgsProject.instance()
        self.box_geo = None
        self.transform_crs = transform_crs

    def _init_rect_layer(self):
        self.rubberBand = QgsRubberBand(
            self.canvas, QgsWkbTypes.PolygonGeometry)
        self.rubberBand.setColor(Qt.blue)
        self.rubberBand.setFillColor(QColor(0, 0, 255, 10))
        self.rubberBand.setWidth(1)

    def _reset_rect_layer(self):
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
        if self.box_geo is not None:
            rowcol1 = rowcol(tf, self.box_geo[0], self.box_geo[1])
            rowcol2 = rowcol(tf, self.box_geo[2], self.box_geo[3])
            return np.array([rowcol1[1], rowcol1[0], rowcol2[1], rowcol2[0]])
        else:
            return None


class SAM_PolygonFeature:
    def __init__(self, transform_crs: TransformCRS, shapefile=None):
        '''SAM_PolygonFeature class'''
        self.qgis_project = QgsProject.instance()
        self.transform_crs = transform_crs
        if shapefile:
            self._load_shapefile(shapefile)
        else:
            self._init_layer()

    def _load_shapefile(self, shapefile):
        if isinstance(shapefile, Path):
            shapefile = str(shapefile)
        self.layer = QgsVectorLayer(shapefile, "polygon_sam", "ogr")
        self.show_layer()
        self.ensure_edit_mode()

    def _init_layer(self,):
        layer_list = QgsProject.instance().mapLayersByName("polygon_sam")
        if layer_list:
            self.layer = layer_list[0]
            self.layer.commitChanges()
        else:
            self.layer = QgsVectorLayer('Polygon', 'polygon_sam', 'memory')
            # self.layer.setCrs(self.qgis_project.crs())
            self.layer.setCrs(self.transform_crs.feature_crs)
        # Set the provider to accept the data source
        prov = self.layer.dataProvider()
        prov.addAttributes([QgsField("id", QVariant.Int),
                           QgsField("Area", QVariant.Double)])
        self.layer.updateFields()
        self.show_layer()
        self.ensure_edit_mode()

    def show_layer(self):
        self.qgis_project.addMapLayer(self.layer)
        self.layer.startEditing()
        symbol = QgsFillSymbol.createSimple({'color': '0,255,0,40',
                                            'color_border': 'green',
                                             'width_border': '0.6'})
        self.layer.renderer().setSymbol(symbol)
        # show the change
        self.layer.triggerRepaint()

    def add_geojson_feature(self, geojson):
        features = []
        num_polygons = self.layer.featureCount()
        for idx, geom in enumerate(geojson):
            points = []
            coordinates = geom['geometry']['coordinates'][0]
            for coord in coordinates:
                # TODO transform pointXY from feature_crs to polygon layer crs, if not match
                point = QgsPointXY(*coord)
                if self.layer.crs() != self.transform_crs.feature_crs:
                    point = self.transform_crs.transform_point_from_feature_crs(
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
        if not self.layer.isEditable():
            self.layer.startEditing()

    def rollback_changes(self):
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
        self.layer.commitChanges()
