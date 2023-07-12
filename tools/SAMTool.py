from typing import List
import numpy as np
from pathlib import Path
import time

import numpy as np
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.features import shapes as get_shapes
from PyQt5.QtWidgets import QMessageBox
from qgis.core import QgsRectangle, QgsMessageLog, Qgis
from torch.utils.data import DataLoader
from .torchgeo_sam import SamTestFeatureDataset, SamTestFeatureGeoSampler
from .sam_ext import build_sam_no_encoder, SamPredictorNoImgEncoder
from .geoTool import LayerExtent, ImageCRSManager
from .canvasTool import SAM_PolygonFeature, Canvas_Rectangle, Canvas_Points
from torchgeo.datasets import BoundingBox, stack_samples
from torchgeo.samplers import Units


class SAM_Model:
    def __init__(self, feature_dir, cwd):
        self.feature_dir = feature_dir
        self.sam_checkpoint = {
            "vit_h": cwd + "/checkpoint/sam_vit_h_4b8939_no_img_encoder.pth",  # vit huge model
            "vit_l": cwd + "/checkpoint/sam_vit_l_0b3195_no_img_encoder.pth",  # vit large model
            "vit_b": cwd + "/checkpoint/sam_vit_b_01ec64_no_img_encoder.pth",  # vit base model
        }
        self.model_type = None
        self.img_crs = None
        self.extent = None
        self.sample_path = None  # necessary
        self._prepare_data_and_layer()

    def _prepare_data_and_layer(self):
        """Prepares data and layer."""
        self.test_features = SamTestFeatureDataset(
            root=self.feature_dir, bands=None, cache=False)
        self.img_crs = str(self.test_features.crs)
        # Load sam decoder
        self.model_type = self.test_features.model_type
        if self.model_type is None:
            raise Exception("No sam model type info. found in feature files")

        sam = build_sam_no_encoder(
            checkpoint=self.sam_checkpoint[self.model_type])
        self.predictor = SamPredictorNoImgEncoder(sam)

        feature_bounds = self.test_features.index.bounds
        self.feature_size = len(self.test_features.index)  # .get_size()
        self.extent = QgsRectangle(
            feature_bounds[0], feature_bounds[2], feature_bounds[1], feature_bounds[3])

    def sam_predict(self,
                    canvas_points: Canvas_Points,
                    canvas_rect: Canvas_Rectangle,
                    sam_polygon: SAM_PolygonFeature,
                    prompt_history: List) -> bool:
        extent_union = LayerExtent.union_extent(
            canvas_points.extent, canvas_rect.extent)

        if extent_union is None:
            sam_polygon.rollback_changes()
            return True

        min_x, max_x, min_y, max_y = extent_union

        prompts_roi = BoundingBox(
            min_x, max_x, min_y, max_y, self.test_features.index.bounds[4], self.test_features.index.bounds[5])

        start_time = time.time()
        test_sampler = SamTestFeatureGeoSampler(
            self.test_features, roi=prompts_roi)

        if len(test_sampler) == 0:
            mb = QMessageBox()
            mb.setText(
                'Point/rectangle is located outside of the feature boundary, click OK to undo last prompt.')
            mb.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            return_value = mb.exec()
            # TODO: Clear last point falls outside the boundary
            if return_value == QMessageBox.Ok:
                return False
            elif return_value == QMessageBox.Cancel:
                return True

        for query in test_sampler:
            # different query than last time, update feature
            if query['path'] == self.sample_path:
                QgsMessageLog.logMessage(
                    f"Same feature as last time", 'Geo SAM', level=Qgis.Info)
                break

            self.sample = self.test_features[query]
            self.sample_path = self.sample['path']

            bbox = self.sample['bbox']  # batch['bbox'][0]
            img_width = img_height = self.predictor.model.image_encoder.img_size  # 1024
            input_width = input_height = self.predictor.model.image_encoder.img_size  # 1024
            if 'img_shape' in self.sample.keys():
                img_height = self.sample['img_shape'][-2]
                img_width = self.sample['img_shape'][-1]
                input_height = self.sample['input_shape'][-2]
                input_width = self.sample['input_shape'][-1]

            self.img_clip_transform = transform_from_bounds(
                bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, img_width, img_height)

            img_features = self.sample['image']
            self.predictor.set_image_feature(
                img_features=img_features,
                img_size=(img_height, img_width),
                input_size=(input_height, input_width)
            )

            QgsMessageLog.logMessage(
                f"Load new feature", 'Geo SAM', level=Qgis.Info)
            break

        input_point, input_label = canvas_points.get_points_and_labels(
            self.img_clip_transform)

        input_box = canvas_rect.get_img_box(self.img_clip_transform)
        # TODO: Points or rectangles can be negative or exceed 1024, should be regulated
        # also may consider remove those points after checking
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )
        end_time = time.time()
        # get the execution time of sam predictor, ms
        elapsed_time = (end_time - start_time) * 1000

        QgsMessageLog.logMessage(
            f"SAM predict executed with {elapsed_time:.3f} ms", 'Geo SAM', level=Qgis.Info)

        # QgsMessageLog.logMessage(
        #     f"SAM feature shape {masks.shape}", 'Geo SAM', level=Qgis.Info)

        # shape (1, 1024, 1024)
        mask = masks[0, ...]
        # mask = mask_morph

        # convert mask to geojson
        # results = ({'properties': {'raster_val': v}, 'geometry': s}
        #            for i, (s, v) in enumerate(rio.features.shapes(mask.astype(np.uint8), mask=mask, transform=img_clip_transform)))
        # geoms = list(results)
        shape_generator = get_shapes(
            mask.astype(np.uint8),
            mask=mask,
            connectivity=4,  # change from default:4 to 8
            transform=self.img_clip_transform
        )
        geojson = [{'properties': {'raster_val': value}, 'geometry': polygon}
                   for polygon, value in shape_generator]

        # add to layer
        sam_polygon.rollback_changes()
        sam_polygon.add_geojson_feature(geojson, prompt_history)
        return True
