import typing
import numpy as np
from pathlib import Path

import numpy as np
import rasterio as rio
from PyQt5.QtWidgets import QMessageBox
from qgis.core import QgsRectangle, QgsMessageLog, Qgis
from torch.utils.data import DataLoader
from .torchgeo_sam import SamTestFeatureDataset, SamTestFeatureGeoSampler
from .sam_ext import sam_model_registry_no_encoder, SamPredictorNoImgEncoder
from .geoTool import LayerExtent, ImageCRSManager
from torchgeo.datasets import BoundingBox, stack_samples
from torchgeo.samplers import Units


class SAM_Model:
    def __init__(self, feature_dir, cwd, model_type="vit_h"):
        self.feature_dir = feature_dir
        self.sam_checkpoint = cwd + "/checkpoint/sam_vit_h_4b8939_no_img_encoder.pth"
        self.model_type = model_type
        self._prepare_data_and_layer()

    def _prepare_data_and_layer(self):
        """Prepares data and layer."""
        self.test_features = SamTestFeatureDataset(
            root=self.feature_dir, bands=None, cache=False)  # display(test_imgs.index) #
        self.img_crs = str(self.test_features.crs)
        # Load sam decoder
        sam = sam_model_registry_no_encoder[self.model_type](
            checkpoint=self.sam_checkpoint)
        self.predictor = SamPredictorNoImgEncoder(sam)
        feature_bounds = self.test_features.index.bounds # list [minx, maxx, miny, maxy, mint, maxt]
        self.extent = QgsRectangle(feature_bounds[0], feature_bounds[2], feature_bounds[1], feature_bounds[3])

    def sam_predict(self, canvas_points, canvas_rect, sam_polygon):
        min_x, max_x, min_y, max_y = LayerExtent.union_extent(
            canvas_points.extent, canvas_rect.extent)

        points_roi = BoundingBox(
            min_x, max_x, min_y, max_y, self.test_features.index.bounds[4], self.test_features.index.bounds[5])

        test_sampler = SamTestFeatureGeoSampler(
            self.test_features, feature_size=64, roi=points_roi, units=Units.PIXELS)  # Units.CRS or Units.PIXELS

        if len(test_sampler) == 0:
            mb = QMessageBox()
            # ,  please press CMD/Ctrl+Z to undo the edit
            mb.setText('Point is located outside of the image boundary')
            mb.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            return_value = mb.exec()
            if return_value == QMessageBox.Ok:
                print('You pressed OK')
            elif return_value == QMessageBox.Cancel:
                print('You pressed Cancel')
            return False
        test_dataloader = DataLoader(
            self.test_features, batch_size=1, sampler=test_sampler, collate_fn=stack_samples)

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
        img_clip_transform = rio.transform.from_bounds(
            bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height)

        input_point, input_label = canvas_points.get_points_and_labels(
            img_clip_transform)
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
        
        QgsMessageLog.logMessage("SAM predict executed", 'Geo SAM', level=Qgis.Info)

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
