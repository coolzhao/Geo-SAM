from qgis.core import QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand, QgsMapTool, QgsMapToolPan
from qgis.core import (
    QgsPointXY, QgsWkbTypes,QgsMarkerSymbol, QgsRectangle, QgsField, QgsFillSymbol,
    QgsGeometry, QgsFeature, QgsVectorLayer, QgsRasterLayer, QgsSimpleMarkerSymbolLayer, QgsSingleSymbolRenderer)
from qgis.PyQt.QtCore import QCoreApplication,QVariant
from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QObject, QTimer
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QShortcut,
    QMainWindow,
    QFileDialog,
    QAction,
    QToolBar,
    QDockWidget,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QApplication
)
from PyQt5.QtGui import QKeySequence, QIcon, QColor
from PyQt5 import uic
from qgis.utils import iface
import os
import numpy as np
from pathlib import Path
# from .ui.UI import UI_Selector

## Enable High DPI display for Qt5
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
os.environ["QT_ENABLE_HIGHDPI_SCALING"]   = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
# os.environ["QT_SCALE_FACTOR"]             = "1"
# qapp = QApplication(sys.argv)
# qapp.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

import sys
import os
import importlib
import typing
from PyQt5.QtCore import QObject
from console.console import _console

# script_path = _console.console.tabEditorWidget.currentWidget().path
# print(script_path)
# cwd = os.path.dirname(script_path)
# cwd_path = Path(cwd)
# print(type(cwd))
# print(cwd_path / "ui//Selector.ui")
# feature_dir = cwd + "/features/beiluhe_google_img_201211_utm_new_export_pyramid_clip"

# if cwd not in sys.path:
#     sys.path.append(cwd)

import numpy as np
import rasterio as rio
from rasterio.transform import rowcol
from torch.utils.data import DataLoader
from torchgeo.datasets import BoundingBox, stack_samples
from torchgeo.samplers import Units
from .torchgeo_sam import SamTestFeatureDataset, SamTestFeatureGeoSampler
from .sam_ext import sam_model_registry_no_encoder, SamPredictorNoImgEncoder
# from qgis_ext import SamEventFilter, SamTimer
from qgis.core import QgsProject, QgsCoordinateReferenceSystem, QgsCoordinateTransform

# importlib.reload(sys.modules['torchgeo_sam'])
# importlib.reload(sys.modules['sam_ext'])
# importlib.reload(sys.modules['qgis_ext'])

# UI_Selector = uic.loadUi(cwd_path / "ui/Selector.ui")

class TransformCRS:
    def __init__(self, feature_crs) -> None:
        # self.rect_crs = self.point_crs
        # self.polygon_crs = QgsCoordinateReferenceSystem(polygon_crs)
        self.feature_crs = QgsCoordinateReferenceSystem(feature_crs) # from str to QgsCRS
        print(self.feature_crs.authid())

    def transform_point_from_feature_crs(self, point: QgsPointXY, point_crs: QgsCoordinateReferenceSystem):
        '''transform point from feature crs to point crs'''
        # point_crs = QgsCoordinateReferenceSystem(point_crs)
        transform = QgsCoordinateTransform(self.feature_crs, point_crs, QgsProject.instance())
        point_transformed = transform.transform(point)
        return point_transformed

    def transform_point_to_feature_crs(self, point: QgsPointXY, point_crs: QgsCoordinateReferenceSystem):
        '''transform point from point crs to feature crs'''
        # point_crs = QgsCoordinateReferenceSystem(point_crs)
        transform = QgsCoordinateTransform(point_crs, self.feature_crs, QgsProject.instance())
        point_transformed = transform.transform(point) # direction can be used
        return point_transformed

    def transform_extent_to_feature_crs(self, extent: QgsRectangle, point_crs: QgsCoordinateReferenceSystem):
        '''transform extent from point crs to feature crs'''
        # point_crs = QgsCoordinateReferenceSystem(point_crs)
        transform = QgsCoordinateTransform(point_crs, self.feature_crs, QgsProject.instance())
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
                layer_ext = transform_crs.transform_extent_to_feature_crs(layer_ext, layer.crs())
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
        elif (self.startPoint.x() == self.endPoint.x() or \
            self.startPoint.y() == self.endPoint.y()):
            return None
        else:
            # TODO startPoint endPoint transform
            if self.qgis_project.crs() != self.transform_crs.feature_crs:
                self.startPoint = self.transform_crs.transform_point_to_feature_crs(self.startPoint, self.qgis_project.crs())
                self.endPoint = self.transform_crs.transform_point_to_feature_crs(self.endPoint, self.qgis_project.crs())
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
        self.layer.addFeatures([self.feature]) # add by zyzhao
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
        if not self.layer.isEditable(): # add by zyzhao
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
        self.layer_fg = self._init_points_layer( "Foreground Points", "blue")
        self.layer_bg = self._init_points_layer( "Background Points", "red")
        
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
            symbol = QgsMarkerSymbol.createSimple({'name': 'circle', 'color': color})
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
                    point = self.transform_crs.transform_point_to_feature_crs(point, self.layer_fg.crs())
                row_point, col_point = rowcol(tf, point.x(), point.y())
                points.append((col_point, row_point))
                labels.append(1)

            for feature in self.layer_bg.getFeatures():
                point = feature.geometry().asPoint()
                if self.layer_bg.crs() != self.transform_crs.feature_crs:
                    point = self.transform_crs.transform_point_to_feature_crs(point, self.layer_bg.crs())
                row_point, col_point = rowcol(tf, point.x(), point.y())
                points.append((col_point, row_point))
                labels.append(0)
            points, labels= np.array(points), np.array(labels)

            return points, labels
    @property
    def extent(self):
        e = LayerExtent.union_layer_extent(self.layer_fg, self.layer_bg, self.transform_crs)
        print(e)
        return e

class Canvas_Rectangle:
    def __init__(self, canvas, transform_crs: TransformCRS):
        self.canvas = canvas
        self.qgis_project = QgsProject.instance()
        self.box_geo = None
        self.transform_crs = transform_crs
        
    def _init_rect_layer(self):
        self.rubberBand = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
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
            return np.array([rowcol1[1],rowcol1[0],rowcol2[1],rowcol2[0]])
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
            self.layer = QgsVectorLayer('Polygon', 'polygon_sam' , 'memory')
            # self.layer.setCrs(self.qgis_project.crs())
            self.layer.setCrs(self.transform_crs.feature_crs)
        # Set the provider to accept the data source
        prov = self.layer.dataProvider()
        prov.addAttributes([QgsField("id", QVariant.Int), QgsField("Area", QVariant.Double)])
        self.layer.updateFields()
        self.show_layer()
        self.ensure_edit_mode()
        
    def show_layer(self):
        self.qgis_project.addMapLayer(self.layer)
        self.layer.startEditing()
        symbol = QgsFillSymbol.createSimple({'color':'0,255,0,40',
                                            'color_border':'green', 
                                            'width_border':'0.6'})
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
                    point = self.transform_crs.transform_point_from_feature_crs(point, self.layer.crs())
                points.append(point)
                
            # Add a new feature and assign the geometry
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPolygonXY([points]))
            feature.setAttributes([num_polygons+idx+1, feature.geometry().area()])

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

class SAM_Model:
    def __init__(self, feature_dir, cwd, model_type="vit_h"):
        self.feature_dir = feature_dir
        self.sam_checkpoint = cwd + "/checkpoint/sam_vit_h_4b8939_no_img_encoder.pth"
        self.model_type = model_type
        self._prepare_data_and_layer()
        
    def _prepare_data_and_layer(self):
        """Prepares data and layer."""
        self.test_features = SamTestFeatureDataset(root=self.feature_dir, bands=None, cache=False) #display(test_imgs.index) # 
        self.feature_crs = str(self.test_features.crs)
        # Load sam decoder
        sam = sam_model_registry_no_encoder[self.model_type](checkpoint=self.sam_checkpoint)
        self.predictor = SamPredictorNoImgEncoder(sam)

    def sam_predict(self, canvas_points, canvas_rect, sam_polygon):
        min_x, max_x, min_y, max_y = LayerExtent.union_extent(canvas_points.extent, canvas_rect.extent)    
        
        points_roi = BoundingBox(min_x, max_x, min_y, max_y, self.test_features.index.bounds[4], self.test_features.index.bounds[5])

        test_sampler = SamTestFeatureGeoSampler(self.test_features, feature_size=64, roi=points_roi, units=Units.PIXELS) # Units.CRS or Units.PIXELS

        if len(test_sampler) == 0:
            mb = QMessageBox()
            mb.setText('Point is located outside of the image boundary') #,  please press CMD/Ctrl+Z to undo the edit
            mb.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            return_value = mb.exec()
            if return_value == QMessageBox.Ok:
                print('You pressed OK')
            elif return_value == QMessageBox.Cancel:
                print('You pressed Cancel')
            return False
        test_dataloader = DataLoader(self.test_features, batch_size=1, sampler=test_sampler, collate_fn=stack_samples) # 

        for batch in test_dataloader:
            # print(batch.keys())
            # print(batch['image'].shape)
            # print(batch['path'])
            # print(batch['bbox'])
            # print(len(batch['image']))
            # break
            pass

        bbox = batch['bbox'][0]
        # TODO: Change to sam.img_encoder.img_size
        width = height = 1024
        img_clip_transform = rio.transform.from_bounds(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height)

        input_point, input_label = canvas_points.get_points_and_labels(img_clip_transform)
        box = canvas_rect.get_img_box(img_clip_transform)
        print("box", box)
        
        img_features = batch['image']
        self.predictor.set_image_feature(img_features, img_shape=(1024, 1024))

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=box,
            multimask_output=False,
        )
        print(masks.shape)  # (number_of_masks) x H x W

        mask = masks[0, ...]
        # mask = mask_morph

        # convert mask to geojson
        results = ({'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(rio.features.shapes(mask.astype(np.uint8), mask=mask, transform=img_clip_transform)))
        geoms = list(results)

        # add to layer
        sam_polygon.rollback_changes()
        sam_polygon.add_geojson_feature(geoms)
        return True
  
class Geo_SAM(QObject):
    execute_SAM = pyqtSignal()
    activate_fg = pyqtSignal()
    
    def __init__(self, iface, cwd: str):
        super().__init__()
        self.iface = iface
        self.cwd = cwd
        self.canvas = iface.mapCanvas()
        # self.create_toolbar()
        self.demo_img_name = "beiluhe_google_img_201211_clip"
        feature_dir = cwd + "/features/" + self.demo_img_name
        self.feature_dir = feature_dir
        self.toolPan = QgsMapToolPan(self.canvas)

    def initGui(self):
        # self.toolbar = self.iface.addToolBar("GeoSAM")
        icon_path = os.path.join(self.cwd, "icons/geo_sam_tool.png")
        # icon_path = cwd_path / "asset/icons/sel_points.png"
        # print(icon_path)
        self.action = QAction(
            QIcon(icon_path),
            "GeoSAM Tool",
            self.iface.mainWindow()
            # self.toolbar,
        )
        self.action.triggered.connect(self.create_widget_selector)
        # self.toolbar.addAction(self.action)
        self.iface.addPluginToMenu('&Geo SAM', self.action)
        self.iface.addToolBarIcon(self.action)

    def load_demo_img(self):
        layer_list = QgsProject.instance().mapLayersByName(self.demo_img_name)
        print(layer_list)
        if layer_list:
            rlayer = layer_list[0]
        else:
            img_path = os.path.join(self.cwd, "rasters", self.demo_img_name+'.tif')
            if os.path.exists(img_path):
                rlayer = QgsRasterLayer(img_path, self.demo_img_name)
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    canvas = self.iface.mapCanvas()
                    canvas.setExtent(rlayer.extent())
                    canvas.refresh()
                else:
                    print("Demo image layer failed to load!")
                # self.iface.addRasterLayer(img_path, self.demo_img_name)
            else:
                print(img_path, 'does not exist')

    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        del self.action
    
    def _init_feature_related(self):
        self.sam_model = SAM_Model(self.feature_dir, self.cwd)
        self.transform_crs = TransformCRS(self.sam_model.feature_crs)
        self.canvas_points = Canvas_Points(self.canvas, self.transform_crs)
        self.canvas_rect = Canvas_Rectangle(self.canvas, self.transform_crs)

        self.canvas_points.init_points_layer()
        self.canvas_rect._init_rect_layer()

        # self.shortcut_undo = QShortcut(QKeySequence(Qt.ControlModifier + Qt.Key_Z), self.iface.mainWindow())
        # self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self.iface.mainWindow())
        # self.shortcut_undo.setContext(Qt.ApplicationShortcut)
        # self.shortcut_undo.activated.connect(self.execute_segmentation)
        
        self.execute_SAM.connect(self.execute_segmentation)
        self.activate_fg.connect(self.draw_foreground_point)
        
        # init tools
        self.tool_click_fg = ClickTool(self.canvas, self.canvas_points.feature_fg, self.canvas_points.layer_fg, self.execute_SAM)
        self.tool_click_bg = ClickTool(self.canvas, self.canvas_points.feature_bg, self.canvas_points.layer_bg, self.execute_SAM)
        self.tool_click_rect = RectangleMapTool(self.canvas_rect, self.execute_SAM, self.transform_crs)

    def create_widget_selector(self):
        self._init_feature_related()
        self.load_demo_img()
        
        UI_Selector = uic.loadUi(os.path.join(self.cwd, "ui/Selector.ui"))
        # connect signals for buttons
        self.wdg_sel = UI_Selector
        self.wdg_sel.pushButton_fg.clicked.connect(self.draw_foreground_point)
        self.wdg_sel.pushButton_bg.clicked.connect(self.draw_background_point)
        self.wdg_sel.pushButton_rect.clicked.connect(self.draw_rect)
        self.wdg_sel.pushButton_clear.clicked.connect(self.clear_layers)
        self.wdg_sel.pushButton_find_file.clicked.connect(self.find_file)
        self.wdg_sel.pushButton_load_file.clicked.connect(self.load_shp_file)
        self.wdg_sel.pushButton_save.clicked.connect(self.save_shp_file)
        self.wdg_sel.pushButton_find_feature.clicked.connect(self.find_feature)
        self.wdg_sel.pushButton_load_feature.clicked.connect(self.load_feature)

        self.wdg_sel.radioButton_enable.setChecked(True)
        self.wdg_sel.radioButton_enable.toggled.connect(self.enable_disable)
        self.wdg_sel.pushButton_fg.setCheckable(True)
        self.wdg_sel.pushButton_bg.setCheckable(True)
        self.wdg_sel.pushButton_rect.setCheckable(True)
        
        # add widget to QGIS
        self.wdg_sel.setFloating(True)
        self.iface.addDockWidget(Qt.TopDockWidgetArea, self.wdg_sel)

        # start with fg 
        self.draw_foreground_point()
        
    def enable_disable(self):
        radioButton = self.sender()
        if not radioButton.isChecked():
            # UI_Selector.setEnabled(False)
            self.canvas.setMapTool(self.toolPan)
            # findChildren(self, type, name: str = '', options: Union[Qt.FindChildOptions, Qt.FindChildOption] = Qt.FindChildrenRecursively): argument 2 has unexpected type 'FindChildOption'
            # for wdg in UI_Selector.findChildren(QPushButton, 'pushButton_fg'): # PyQt5.QtWidgets.QDockWidget
            # QApplication.restoreOverrideCursor()
            # print(wdg.objectName())
            self.wdg_sel.pushButton_fg.setEnabled(False)
            self.wdg_sel.pushButton_bg.setEnabled(False)
            self.wdg_sel.pushButton_rect.setEnabled(False)
            # self.tool_click_fg.deactivate()
            # self.tool_click_bg.deactivate()
            # self.tool_click_rect.deactivate()
        else:
            # for wdg in UI_Selector.findChildren(QWidget, 'pushButton', Qt.FindChildrenRecursively):
            #     print(wdg.objectName())
            #     wdg.setEnabled(True)
            self.wdg_sel.pushButton_fg.setEnabled(True)
            self.wdg_sel.pushButton_bg.setEnabled(True)
            self.wdg_sel.pushButton_rect.setEnabled(True)
            self.reset_label_tool()

            # UI_Selector.setEnabled(True)
        
        # self.wdg_sel.radioButton_enable.setEnabled(True)

    
    def execute_segmentation(self):
        if not hasattr(self, "polygon"):
            self.load_shp_file()
        self.sam_model.sam_predict(self.canvas_points, self.canvas_rect, self.polygon)
        
    # def _set_button_selected(self, button):
    #     buttons_map = {'foreground': self.wdg_sel.pushButton_fg, 
    #                    'background': self.wdg_sel.pushButton_bg, 
    #                    'rectangle': self.wdg_sel.pushButton_rect}
    #     button_selected = buttons_map.pop(button)
    #     button_selected.setStyleSheet(f"background-color: #c8c8c8")
    #     for button_other in buttons_map:
    #         buttons_map[button_other].setStyleSheet(f"background-color: #f0f0f0")

    def draw_foreground_point(self):
        self.canvas.setMapTool(self.tool_click_fg)
        button = self.wdg_sel.pushButton_fg # self.sender()
        if button.isChecked():
            print(button.objectName(), "is checked")
        else:
            button.toggle()
        # self._set_button_selected('foreground')
        if self.wdg_sel.pushButton_bg.isChecked():
            self.wdg_sel.pushButton_bg.toggle()
        if self.wdg_sel.pushButton_rect.isChecked():
            self.wdg_sel.pushButton_rect.toggle()
        self.tool_click_bg.deactivate()
        self.prompt_type = 'fgpt'

    def draw_background_point(self):
        self.canvas.setMapTool(self.tool_click_bg)
        button = self.wdg_sel.pushButton_bg # self.sender()
        if button.isChecked():
            print(button.objectName(), "is checked")
        else:
            button.toggle()
        # self._set_button_selected('background')
        if self.wdg_sel.pushButton_fg.isChecked():
            self.wdg_sel.pushButton_fg.toggle()
        if self.wdg_sel.pushButton_rect.isChecked():
            self.wdg_sel.pushButton_rect.toggle()
        self.tool_click_fg.deactivate()
        self.prompt_type = 'bgpt'

    def draw_rect(self):
        self.canvas.setMapTool(self.tool_click_rect)
        button = self.wdg_sel.pushButton_rect #self.sender()
        if button.isChecked():
            print(button.objectName(), "is checked")
        else:
            button.toggle()
        # self._set_button_selected('rectangle')
        if self.wdg_sel.pushButton_fg.isChecked():
            self.wdg_sel.pushButton_fg.toggle()
        if self.wdg_sel.pushButton_bg.isChecked():
            self.wdg_sel.pushButton_bg.toggle()
        self.tool_click_fg.deactivate()
        self.tool_click_bg.deactivate()
        self.prompt_type = 'bbox'

    def find_file(self):
        path, _ = QFileDialog.getSaveFileName(None, "Save shapefile", "")
        self.wdg_sel.path_out.setText(path)

    def load_shp_file(self):
        text = self.wdg_sel.path_out.text()
        self.polygon = SAM_PolygonFeature(self.transform_crs, text)

    def find_feature(self):
        feature_dir_str = QFileDialog.getExistingDirectory(None, "get feature directory", "")
        self.wdg_sel.path_feature.setText(feature_dir_str)

    def load_feature(self):
        self.feature_dir = self.wdg_sel.path_feature.text()
        self._init_feature_related()
        self.load_shp_file()
        self.draw_foreground_point()

    def clear_layers(self):
        self.canvas_points._reset_points_layer()
        self.canvas_rect._reset_rect_layer()
        if hasattr(self, "polygon"):
            self.polygon.rollback_changes()
        # self.activate_fg.emit()
        self.reset_label_tool()
        
    def save_shp_file(self):
        self.polygon.commit_changes()
        self.clear_layers()
        # self.activate_fg.emit()
    
    def reset_label_tool(self):
        if self.prompt_type == 'bbox':
            self.draw_rect()
        else:
            self.draw_foreground_point()
            self.prompt_type = 'fgpt'


# gs = Geo_SAM(iface, cwd)

