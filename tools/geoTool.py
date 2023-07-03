import os
import typing
import numpy as np
from qgis.core import (QgsProject, QgsCoordinateReferenceSystem, Qgis, QgsMessageLog,
                       QgsCoordinateTransform, QgsPointXY,  QgsRectangle, QgsVectorLayer)


class ImageCRSManager:
    '''Manage image crs and transform point and extent between image crs and other crs'''

    def __init__(self, img_crs) -> None:
        self.img_crs = QgsCoordinateReferenceSystem(
            img_crs)  # from str to QgsCRS
        # print(self.img_crs.authid())

    def img_point_to_crs(
        self, point: QgsPointXY, dst_crs: QgsCoordinateReferenceSystem
    ):
        """transform point from this image crs to destination crs

        Parameters:
        ----------
        point: QgsPointXY
            point in this image crs
        dst_crs: QgsCoordinateReferenceSystem
            destination crs for point
        """
        if dst_crs == self.img_crs:
            return point
        transform = QgsCoordinateTransform(
            self.img_crs, dst_crs, QgsProject.instance())
        point_transformed = transform.transform(point)
        return point_transformed

    def point_to_img_crs(
        self, point: QgsPointXY, dst_crs: QgsCoordinateReferenceSystem
    ):
        """transform point from point crs to this image crs

        Parameters:
        ----------
        point: QgsPointXY
            point in itself crs
        point_crs: QgsCoordinateReferenceSystem
            crs of point
        """
        if dst_crs == self.img_crs:
            return point
        transform = QgsCoordinateTransform(
            dst_crs, self.img_crs, QgsProject.instance()
        )
        point_transformed = transform.transform(point)  # direction can be used
        return point_transformed

    def extent_to_img_crs(
        self, extent: QgsRectangle, dst_crs: QgsCoordinateReferenceSystem
    ):
        """transform extent from point crs to this image crs

        Parameters:
        ----------
        extent: QgsRectangle
            extent in itself crs
        dst_crs: QgsCoordinateReferenceSystem
            destination crs for extent
        """
        if dst_crs == self.img_crs:
            return extent
        transform = QgsCoordinateTransform(
            dst_crs, self.img_crs, QgsProject.instance())
        extent_transformed = transform.transformBoundingBox(extent)
        return extent_transformed

    def img_extent_to_crs(self, extent: QgsRectangle, dst_crs: QgsCoordinateReferenceSystem):
        '''transform extent from this image crs to destination crs

        Parameters:
        ----------
        extent: QgsRectangle
            extent in this image crs
        dst_crs: QgsCoordinateReferenceSystem
            destination crs for extent
        '''
        if dst_crs == self.img_crs:
            return extent
        transform = QgsCoordinateTransform(
            self.img_crs, dst_crs, QgsProject.instance())
        extent_transformed = transform.transformBoundingBox(extent)
        return extent_transformed


class LayerExtent:
    def __init__(self):
        pass

    @staticmethod
    def from_qgis_extent(extent: QgsRectangle):
        max_x = extent.xMaximum()
        max_y = extent.yMaximum()
        min_x = extent.xMinimum()
        min_y = extent.yMinimum()
        return min_x, max_x, min_y, max_y

    @classmethod
    def get_layer_extent(
        cls, layer: QgsVectorLayer, img_crs_manager: ImageCRSManager = None
    ):
        """Get the extent of the layer"""
        if layer.featureCount() == 0:
            return None
        else:
            layer.updateExtents()
            layer_ext = layer.extent()
            if layer.crs() != img_crs_manager.img_crs:
                try:
                    layer_ext = img_crs_manager.extent_to_img_crs(
                        layer_ext, layer.crs())
                except Exception as e:
                    QgsMessageLog.logMessage(
                        f">>> Error in extent: {layer_ext} \n type:{type(layer_ext)} \n: {e}", level=Qgis.Critical)
                    return None

            return cls.from_qgis_extent(layer_ext)

    @staticmethod
    def _union_extent(extent1, extent2):
        """Get the union of two extents"""
        min_x1, max_x1, min_y1, max_y1 = extent1
        min_x2, max_x2, min_y2, max_y2 = extent2

        min_x = min(min_x1, min_x2)
        max_x = max(max_x1, max_x2)
        min_y = min(min_y1, min_y2)
        max_y = max(max_y1, max_y2)

        return min_x, max_x, min_y, max_y

    @classmethod
    def union_extent(cls, extent1, extent2):
        """Get the union of two extents (None is allowed)"""
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
    def union_layer_extent(
        cls, layer1, layer2, img_crs_manager: ImageCRSManager = None
    ):
        """Get the union of two layer extents"""
        extent1 = cls.get_layer_extent(layer1, img_crs_manager)
        extent2 = cls.get_layer_extent(layer2, img_crs_manager)

        return cls.union_extent(extent1, extent2)
