import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle
from rasterio.features import shapes as get_shapes
from rasterio.transform import from_bounds as transform_from_bounds
from torchgeo.datasets import BoundingBox

from .canvasTool import Canvas_Points, Canvas_Rectangle, SAM_PolygonFeature
from .geoTool import LayerExtent
from .messageTool import MessageTool
from .sam_ext import SamPredictorNoImgEncoder, build_sam_no_encoder
from .torchgeo_sam import SamTestFeatureDataset, SamTestFeatureGeoSampler

if TYPE_CHECKING:
    from .widgetTool import Selector


class SAM_Model:
    def __init__(self, feature_dir, cwd):
        self.feature_dir = feature_dir
        self.sam_checkpoint = {
            "vit_h": cwd
            + "/checkpoint/sam_vit_h_4b8939_no_img_encoder.pth",  # vit huge model
            "vit_l": cwd
            + "/checkpoint/sam_vit_l_0b3195_no_img_encoder.pth",  # vit large model
            "vit_b": cwd
            + "/checkpoint/sam_vit_b_01ec64_no_img_encoder.pth",  # vit base model
        }
        self.model_type = None
        self.img_crs = None
        self.extent = None
        self.sample_path = None  # necessary
        self._prepare_data_and_layer()

    def _prepare_data_and_layer(self):
        """Prepares data and layer."""
        self.test_features = SamTestFeatureDataset(
            root=self.feature_dir, bands=None, cache=False
        )
        self.img_qgs_crs = QgsCoordinateReferenceSystem(self.test_features.crs)
        self.img_crs = str(self.test_features.crs)
        # Load sam decoder
        self.model_type = self.test_features.model_type
        if self.model_type is None:
            raise Exception("No sam model type info. found in feature files")

        sam = build_sam_no_encoder(checkpoint=self.sam_checkpoint[self.model_type])
        self.predictor = SamPredictorNoImgEncoder(sam)

        feature_bounds = self.test_features.index.bounds
        self.feature_size = len(self.test_features.index)  # .get_size()
        self.extent = QgsRectangle(
            feature_bounds[0], feature_bounds[2], feature_bounds[1], feature_bounds[3]
        )

    def sam_predict(self, selector: "Selector") -> bool:
        extent_union = LayerExtent.union_extent(
            selector.canvas_points.extent, selector.canvas_rect.extent
        )

        if extent_union is None:
            return True  # no extent to predict

        min_x, max_x, min_y, max_y = extent_union

        prompts_roi = BoundingBox(
            min_x,
            max_x,
            min_y,
            max_y,
            self.test_features.index.bounds[4],
            self.test_features.index.bounds[5],
        )

        start_time = time.time()
        test_sampler = SamTestFeatureGeoSampler(self.test_features, roi=prompts_roi)

        # if preview mode, check if the hover location is outside the image
        if len(test_sampler) == 0:
            if selector.preview_mode:
                MessageTool.MessageLog(
                    "Hover location outside the boundary of the image",
                    "warning",
                    notify_user=False,
                )
                return True

        for query in test_sampler:
            # different query than last time, update feature
            if query["path"] == self.sample_path:
                MessageTool.MessageLog("Same feature as last time")
                break

            self.sample = self.test_features[query]
            self.sample_path = self.sample["path"]

            bbox = self.sample["bbox"]  # batch['bbox'][0]
            img_width = img_height = self.predictor.model.image_encoder.img_size  # 1024
            input_width = (
                input_height
            ) = self.predictor.model.image_encoder.img_size  # 1024
            if "img_shape" in self.sample.keys():
                img_height = self.sample["img_shape"][-2]
                img_width = self.sample["img_shape"][-1]
                input_height = self.sample["input_shape"][-2]
                input_width = self.sample["input_shape"][-1]

            self.img_clip_transform = transform_from_bounds(
                bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, img_width, img_height
            )

            img_features = self.sample["image"]
            self.predictor.set_image_feature(
                img_features=img_features,
                img_size=(img_height, img_width),
                input_size=(input_height, input_width),
            )
            MessageTool.MessageLog("Load new feature")
            break

        input_point, input_label = selector.canvas_points.get_points_and_labels(
            self.img_clip_transform
        )

        input_box = selector.canvas_rect.get_img_box(self.img_clip_transform)
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

        MessageTool.MessageLog(f"SAM predict executed with {elapsed_time:.3f} ms")

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
            transform=self.img_clip_transform,
        )
        geojson = [
            {"properties": {"raster_val": value}, "geometry": polygon}
            for polygon, value in shape_generator
        ]

        selector.polygon.canvas_preview_polygon.clear()
        selector.polygon.canvas_preview_polygon.clear()

        target = "prompt"
        if selector.preview_mode:
            target = "preview"

        # overwrite geojson using the new one
        selector.polygon.add_geojson_feature_to_canvas(
            geojson,
            selector,
            target=target,
            overwrite_geojson=True,
        )
        return True
