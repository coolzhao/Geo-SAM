import os
import typing
import numpy as np
from qgis.core import (QgsProject, QgsCoordinateReferenceSystem,
                       QgsCoordinateTransform, QgsPointXY,  QgsRectangle, QgsVectorLayer)


class TransformCRS:
    def __init__(self, feature_crs) -> None:
        # self.rect_crs = self.point_crs
        # self.polygon_crs = QgsCoordinateReferenceSystem(polygon_crs)
        self.feature_crs = QgsCoordinateReferenceSystem(
            feature_crs)  # from str to QgsCRS
        print(self.feature_crs.authid())

    def transform_point_from_feature_crs(self, point: QgsPointXY, point_crs: QgsCoordinateReferenceSystem):
        '''transform point from feature crs to point crs'''
        # point_crs = QgsCoordinateReferenceSystem(point_crs)
        transform = QgsCoordinateTransform(
            self.feature_crs, point_crs, QgsProject.instance())
        point_transformed = transform.transform(point)
        return point_transformed

    def transform_point_to_feature_crs(self, point: QgsPointXY, point_crs: QgsCoordinateReferenceSystem):
        '''transform point from point crs to feature crs'''
        # point_crs = QgsCoordinateReferenceSystem(point_crs)
        transform = QgsCoordinateTransform(
            point_crs, self.feature_crs, QgsProject.instance())
        point_transformed = transform.transform(point)  # direction can be used
        return point_transformed

    def transform_extent_to_feature_crs(self, extent: QgsRectangle, point_crs: QgsCoordinateReferenceSystem):
        '''transform extent from point crs to feature crs'''
        # point_crs = QgsCoordinateReferenceSystem(point_crs)
        transform = QgsCoordinateTransform(
            point_crs, self.feature_crs, QgsProject.instance())
        extent_transformed = transform.transformBoundingBox(extent)
        return extent_transformed


class LayerExtent:
    def __init__(self):
        pass

    @staticmethod
    def get_layer_extent(layer: QgsVectorLayer, transform_crs: TransformCRS = None):
        '''Get the extent of the layer'''
        if layer.featureCount() == 0:
            return None
        else:
            layer_ext = layer.extent()
            layer.updateExtents()
            layer_ext = layer.extent()
            # TODO transform extent
            if layer.crs() != transform_crs.feature_crs:
                layer_ext = transform_crs.transform_extent_to_feature_crs(
                    layer_ext, layer.crs())
            max_x = layer_ext.xMaximum()
            max_y = layer_ext.yMaximum()
            min_x = layer_ext.xMinimum()
            min_y = layer_ext.yMinimum()
            return min_x, max_x, min_y, max_y

    @staticmethod
    def _union_extent(extent1, extent2):
        '''Get the union of two extents'''
        min_x1, max_x1, min_y1, max_y1 = extent1
        min_x2, max_x2, min_y2, max_y2 = extent2

        min_x = min(min_x1, min_x2)
        max_x = max(max_x1, max_x2)
        min_y = min(min_y1, min_y2)
        max_y = max(max_y1, max_y2)

        return min_x, max_x, min_y, max_y

    @classmethod
    def union_extent(cls, extent1, extent2):
        '''Get the union of two extents (None is allowed)'''
        if extent1 is not None and extent2 is not None:
            min_x, max_x, min_y, max_y = cls._union_extent(extent1, extent2)
        elif extent1 is None and extent2 is not None:
            min_x, max_x, min_y, max_y = extent2
        elif extent1 is not None and extent2 is None:
            min_x, max_x, min_y, max_y = extent1
        else:
            return None

        return min_x, max_x, min_y, max_y

    @classmethod
    def union_layer_extent(cls, layer1, layer2, transform_crs: TransformCRS = None):
        '''Get the union of two layer extents'''
        extent_fg = cls.get_layer_extent(layer1, transform_crs)
        extent_bg = cls.get_layer_extent(layer2, transform_crs)

        return cls.union_extent(extent_fg, extent_bg)
