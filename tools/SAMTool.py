import typing
import numpy as np
from pathlib import Path
import time

import numpy as np
import rasterio as rio
from PyQt5.QtWidgets import QMessageBox
from qgis.core import QgsRectangle, QgsMessageLog, Qgis
from torch.utils.data import DataLoader
from .torchgeo_sam import SamTestFeatureDataset, SamTestFeatureGeoSampler
from .sam_ext import sam_model_registry_no_encoder, SamPredictorNoImgEncoder
from .geoTool import LayerExtent, ImageCRSManager
from .canvasTool import SAM_PolygonFeature, Canvas_Rectangle, Canvas_Points
from torchgeo.datasets import BoundingBox, stack_samples
from torchgeo.samplers import Units


class SAM_Model:
    def __init__(self, feature_dir, cwd, model_type="vit_h"):
        self.feature_dir = feature_dir
        self.sam_checkpoint = cwd + "/checkpoint/sam_vit_h_4b8939_no_img_encoder.pth"
        self.model_type = model_type
        self._prepare_data_and_layer()
        self.sample_path = None

    def _prepare_data_and_layer(self):
        """Prepares data and layer."""
        self.test_features = SamTestFeatureDataset(
            root=self.feature_dir, bands=None, cache=False)  # display(test_imgs.index) #
        self.img_crs = str(self.test_features.crs)
        # Load sam decoder
        sam = sam_model_registry_no_encoder[self.model_type](
            checkpoint=self.sam_checkpoint)
        self.predictor = SamPredictorNoImgEncoder(sam)

        feature_bounds = self.test_features.index.bounds
        self.extent = QgsRectangle(
            feature_bounds[0], feature_bounds[2], feature_bounds[1], feature_bounds[3])

    def sam_predict(self, canvas_points: Canvas_Points, canvas_rect: Canvas_Rectangle, sam_polygon: SAM_PolygonFeature):
        min_x, max_x, min_y, max_y = LayerExtent.union_extent(
            canvas_points.extent, canvas_rect.extent)

        points_roi = BoundingBox(
            min_x, max_x, min_y, max_y, self.test_features.index.bounds[4], self.test_features.index.bounds[5])

        start_time = time.time()
        test_sampler = SamTestFeatureGeoSampler(
            self.test_features, feature_size=64, roi=points_roi, units=Units.PIXELS)  # Units.CRS or Units.PIXELS

        if len(test_sampler) == 0:
            mb = QMessageBox()
            # ,  please press CMD/Ctrl+Z to undo the edit
            mb.setText('Point is located outside of the image boundary')
            mb.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            return_value = mb.exec()
            # TODO: Clear last point falls outside the boundary
            if return_value == QMessageBox.Ok:
                print('You pressed OK')
            elif return_value == QMessageBox.Cancel:
                print('You pressed Cancel')

            return False

        for query in test_sampler:
            # different query than last time, update feature
            if query['path'] == self.sample_path:
                break
            sample = self.test_features[query]
            self.sample_path = sample['path']
            self.sample_bbox = sample['bbox']
            self.img_features = sample['image']
            break

        # test_dataloader = DataLoader(
        #     self.test_features, batch_size=1, sampler=test_sampler, collate_fn=stack_samples)

        # for batch in test_dataloader:
        #     # print(batch.keys())
        #     # print(batch['image'].shape)
        #     # print(batch['path'])
        #     # print(batch['bbox'])
        #     # print(len(batch['image']))
        #     # break
        #     pass

        bbox = self.sample_bbox  # batch['bbox'][0]
        # Change to sam.img_encoder.img_size
        img_width = img_height = self.predictor.model.image_encoder.img_size  # 1024
        img_clip_transform = rio.transform.from_bounds(
            bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, img_width, img_height)

        input_point, input_label = canvas_points.get_points_and_labels(
            img_clip_transform)

        input_box = canvas_rect.get_img_box(img_clip_transform)
        # print("box", input_box)

        # img_features = batch['image']
        self.predictor.set_image_feature(
            self.img_features, img_shape=(img_height, img_width))

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

        mask = masks[0, ...]
        # mask = mask_morph

        # convert mask to geojson
        # results = ({'properties': {'raster_val': v}, 'geometry': s}
        #            for i, (s, v) in enumerate(rio.features.shapes(mask.astype(np.uint8), mask=mask, transform=img_clip_transform)))
        # geoms = list(results)
        shape_generator = rio.features.shapes(
            mask.astype(np.uint8),
            mask=mask,
            transform=img_clip_transform
        )
        geojson = [{'properties': {'raster_val': value}, 'geometry': polygon}
                 for polygon, value in shape_generator]

        # add to layer
        sam_polygon.rollback_changes()
        sam_polygon.add_geojson_feature(geojson)
        return True
