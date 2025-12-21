import glob
import os
import re
import sys
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import torch
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds as window_from_bounds
from torch import Tensor
from torch.nn import functional as F
from shapely.geometry import box

from .torchgeo_local.utils import BoundingBox, _to_tuple, tile_to_chips
from .torchgeo_local.datasets import LocalGeoDataset, LocalRasterDataset
from .torchgeo_local.samplers import LocalGeoSampler, Units

from .messageTool import MessageTool

# remove rtree dependency and len method hack

def get_pixel_size(res: Union[float, Tuple[float, float]]) -> Tuple[float, float]:
    """Get pixel size from resolution.

    Args:
        res: resolution in pixels or CRS units

    Returns:
        pixel size in pixels
    """
    if isinstance(res, float):
        pixel_size = (res, res)
    elif isinstance(res, tuple):
        pixel_size = (res[0], res[1])
    else:
        raise TypeError("Resolution must be a float or a tuple of floats")
    return pixel_size

class SamTestGridGeoSampler(LocalGeoSampler):
    """Samples elements in a grid-like fashion.
    accept image smaller than desired patch_size
    """

    def __init__(
        self,
        dataset: LocalGeoDataset,
        size: Union[Tuple[float, float], float],
        stride: Union[Tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            stride: distance to skip between each patch
            roi: region of interest to sample from (minx, maxx, miny, maxy)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.patch_size = self.size
        self.stride = _to_tuple(stride)

        if units == Units.PIXELS:
            pixel_size = get_pixel_size(self.res)

            self.size = (self.size[0] * pixel_size[0], self.size[1] * pixel_size[1])
            self.stride = (self.stride[0] * pixel_size[0], self.stride[1] * pixel_size[1])

        self.hits = []
        self.hits_small = []
        
        # Geopandas index query
        if not self.index.empty:
            intersecting_gdf = self.index[self.index.intersects(self.roi.geometry)]
            for _, row in intersecting_gdf.iterrows():
                b = row.geometry.bounds # (minx, miny, maxx, maxy)
                row_bbox = BoundingBox(b[0], b[2], b[1], b[3], self.dataset.crs)
                if (
                    row_bbox.maxx - row_bbox.minx >= self.size[1]
                    or row_bbox.maxy - row_bbox.miny >= self.size[0]
                ):
                    self.hits.append(row)
                else:
                    self.hits_small.append(row)

        self.length = 0
        for row in self.hits:
            b = row.geometry.bounds
            bounds = BoundingBox(b[0], b[2], b[1], b[3], self.dataset.crs)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            self.length += rows * cols

        for _ in self.hits_small:
            self.length += 1

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return the index of a dataset.

        Returns:
            bbox(minx, maxx, miny, maxy) coordinates to index a dataset
            & raster file path
        """
        # For each tile...
        for row in self.hits + self.hits_small:
            b = row.geometry.bounds
            bounds = BoundingBox(b[0], b[2], b[1], b[3], self.dataset.crs)
            if any(row is h for h in self.hits):
                rows, cols = tile_to_chips(bounds, self.size, self.stride)

                # For each row...
                for i in range(rows):
                    miny = bounds.miny + i * self.stride[0]
                    maxy = miny + self.size[0]
                    if maxy > bounds.maxy:
                        maxy = bounds.maxy
                        miny = bounds.maxy - self.size[0]
                        if miny < bounds.miny:
                            miny = bounds.miny

                    # For each column...
                    for j in range(cols):
                        minx = bounds.minx + j * self.stride[1]
                        maxx = minx + self.size[1]
                        if maxx > bounds.maxx:
                            maxx = bounds.maxx
                            minx = bounds.maxx - self.size[1]
                            if minx < bounds.minx:
                                minx = bounds.minx

                        query = {
                            "bbox": BoundingBox(minx, maxx, miny, maxy, self.dataset.crs),
                            "path": cast(str, row["path"]),
                            "size": int(self.patch_size[0]),
                        }

                        yield query
            else:
                query = {
                    "bbox": bounds,
                    "path": cast(str, row["path"]),
                    "size": int(self.patch_size[0]),
                }

                yield query

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length



class SamTestFeatureDataset(LocalRasterDataset):
    filename_glob = "*.tif"
    filename_regex = ".*"
    date_format = ""
    is_image = True
    separate_files = False
    all_bands = []
    rgb_bands = []

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        if self.separate_files:
            raise NotImplementedError(
                "Testing for separated files are not supported yet"
            )
        super().__init__(root, crs, res, transforms, cache)
        
        self.bands = bands or self.all_bands
        self.band_indexes = None
        if self.bands:
            self.band_indexes = [self.all_bands.index(b) + 1 for b in self.bands]
        self.model_type = None
        model_type_list = ["vit_h", "vit_l", "vit_b"]

        # Populate the dataset index
        pathname = os.path.join(root, self.filename_glob)
        raster_list = glob.glob(pathname, recursive=True)
        if not raster_list:
             raise FileNotFoundError(f"No data found in {root}")

        raster_name = os.path.basename(raster_list[0])
        for m_type in model_type_list:
            if m_type in raster_name:
                self.model_type = m_type
                break
        
        dir_name = os.path.basename(root)
        csv_filepath = os.path.join(root, dir_name + ".csv")
        
        data = []
        if os.path.exists(csv_filepath):
            index_df = pd.read_csv(csv_filepath)
            if len(index_df) == len(raster_list):
                for _, row_df in index_df.iterrows():
                    if crs is None:
                        try:
                            # Prefer pyproj.CRS as it is often more resilient in this environment
                            crs = CRS.from_user_input(row_df["crs"])
                        except Exception as e:
                            MessageTool.MessageLog(f"Warning: Failed to parse CRS with pyproj: {e}. Falling back to string.")
                            crs = row_df["crs"]
                    
                    if res is None:
                        res = row_df["res"]
                    
                    filepath = os.path.join(root, os.path.basename(row_df["filepath"]))
                    geom = box(row_df["minx"], row_df["miny"], row_df["maxx"], row_df["maxy"])
                    data.append({"geometry": geom, "path": filepath})
                
                try:
                    self.index = gpd.GeoDataFrame(data, geometry="geometry", crs=crs)
                except Exception as e:
                    MessageTool.MessageLog(f"Warning: Failed to create GeoDataFrame with CRS object: {e}. Retrying without CRS.")
                    self.index = gpd.GeoDataFrame(data, geometry="geometry")
                    self.index.crs = crs
                MessageTool.MessageLog(
                    f"Index loaded from: {os.path.basename(csv_filepath)}"
                )
            else:
                MessageTool.MessageLog(
                    f"Index file: {os.path.basename(csv_filepath)} mismatch, recreating."
                )

        if self.index.empty:
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for filepath in raster_list:
                match = re.match(filename_regex, os.path.basename(filepath))
                if match is not None:
                    try:
                        with rasterio.open(filepath) as src:
                            if crs is None:
                                try:
                                    crs = CRS.from_user_input(src.crs.to_string())
                                except:
                                    crs = src.crs
                            if res is None:
                                res = src.res[0]

                            with WarpedVRT(src, crs=crs) as vrt:
                                minx, miny, maxx, maxy = vrt.bounds
                                geom = box(minx, miny, maxx, maxy)
                                data.append({"geometry": geom, "path": filepath})
                    except Exception as e:
                        if "PROJ: internal_proj_create_from_database" in str(e):
                            MessageTool.MessageBoxOK(
                                "Critical PROJ Environment Error Detected:\n\n"
                                f"{e}\n\n"
                                "This is usually caused by a conflict between QGIS and other conda environments "
                                "(like ISCE2) in your PYTHONPATH. Please check your environment variables or "
                                "run 'conda update -n qgis --all'."
                            )
                        MessageTool.MessageLog(f"Error opening {filepath}: {e}")
                        continue
            
            try:
                self.index = gpd.GeoDataFrame(data, geometry="geometry", crs=crs)
            except Exception as e:
                MessageTool.MessageLog(f"Warning: Failed to create GeoDataFrame with CRS: {e}")
                self.index = gpd.GeoDataFrame(data, geometry="geometry")
                self.index.crs = crs
            # Save CSV for future use
            save_df = pd.DataFrame(data)
            save_df["filepath"] = save_df["path"].apply(os.path.basename)
            save_df["minx"] = save_df["geometry"].apply(lambda g: g.bounds[0])
            save_df["miny"] = save_df["geometry"].apply(lambda g: g.bounds[1])
            save_df["maxx"] = save_df["geometry"].apply(lambda g: g.bounds[2])
            save_df["maxy"] = save_df["geometry"].apply(lambda g: g.bounds[3])
            save_df["crs"] = str(crs)
            save_df["res"] = res
            save_df.to_csv(csv_filepath, index=False)
            MessageTool.MessageLog(f"Index file: {os.path.basename(csv_filepath)} saved")

        self._crs = crs
        self.res = res
        if not self.index.empty:
            self.index["res"] = res
            self.index_df = self.index

    def __getitem__(self, query: Dict[str, Any]) -> Dict[str, Any]:
        bbox = query["bbox"]
        filepath = query["path"]

        # Geopandas query
        hits = self.index[self.index.intersects(bbox.geometry)]
        if filepath not in hits["path"].values:
             raise IndexError(f"query: {bbox} not found for {filepath}")

        vrt_fh = self._load_warp_file(filepath)
        src = vrt_fh
        dest = src.read()
        tags = src.tags()
        
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)

        sample = {"crs": self.crs, "bbox": bbox, "path": filepath}
        if "img_shape" in tags.keys():
            sample["img_shape"] = eval(tags["img_shape"])
            sample["input_shape"] = eval(tags["input_shape"])

        if self.is_image:
            sample["image"] = tensor.float()
        else:
            sample["mask"] = tensor

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class SamTestRasterDataset(SamTestFeatureDataset):
    all_bands = ["Red", "Green", "Blue", "Alpha"]
    rgb_bands = ["Red", "Green", "Blue"]

    def __getitem__(self, query: Dict[str, Any]) -> Dict[str, Any]:
        bbox = query["bbox"]
        filepath = query["path"]
        patch_size = query["size"]

        hits = self.index[self.index.intersects(bbox.geometry)]
        if filepath not in hits["path"].values:
            raise IndexError(f"query: {bbox} not found for {filepath}")

        vrt_fh = self._load_warp_file(filepath)
        src = vrt_fh
        pixel_size = get_pixel_size(self.res)
        out_width = round((bbox.maxx - bbox.minx) / pixel_size[0])
        out_height = round((bbox.maxy - bbox.miny) / pixel_size[1])
        
        band_indexes = getattr(self, "band_indexes", None)
        count = len(band_indexes) if band_indexes else src.count
        out_shape = (count, out_height, out_width)

        if out_height == 0 or out_width == 0:
            raise Exception("Patch size should be greater than zero.")

        target_height, target_width = self.get_preprocess_shape(
            out_height, out_width, patch_size
        )
        target_shape = (count, target_height, target_width)
        dest = src.read(
            indexes=band_indexes,
            out_shape=target_shape,
            window=window_from_bounds(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, src.transform),
            resampling=Resampling.bilinear,
        )

        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)
        if torch.isnan(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0)
        tensor = self.pad_patch(tensor, patch_size)

        sample = {
            "crs": self.crs,
            "bbox": bbox,
            "path": filepath,
            "img_shape": out_shape,
            "input_shape": target_shape,
        }
        if self.is_image:
            sample["image"] = tensor.float()
        else:
            sample["mask"] = tensor

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @staticmethod
    def get_preprocess_shape(
        old_h: int, old_w: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(old_h, old_w)
        new_h, new_w = old_h * scale, old_w * scale
        new_w = int(new_w + 0.5)  # floor
        new_h = int(new_h + 0.5)
        return (new_h, new_w)

    @staticmethod
    def pad_patch(x: Tensor, patch_size: int):
        """
        Pad the patch to desired patch_size
        """
        h, w = x.shape[-2:]
        pad_h = patch_size - h
        pad_w = patch_size - w
        # pads are described starting from the last dimension and moving forward.
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def plot(self, sample, bright=1):
        check_rgb = all(item in self.bands for item in self.rgb_bands)
        if not check_rgb:
            raise Exception("Need R G B bands to visualize")

        # Find the correct bands
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.bands.index(band))

        # Reorder and rescale the image
        if sample["image"].ndim == 4:
            image = sample["image"][0, rgb_indices, :, :].permute(1, 2, 0)
        else:
            image = sample["image"][rgb_indices, :, :].permute(1, 2, 0)

        # if image.max() > 10:
        #     image = self.apply_scale(image)
        image = torch.clamp(image * bright / 255, min=0, max=1)

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")

        return fig


class SamTestFeatureGeoSampler(LocalGeoSampler):
    """Samples entire files at a time.
    """

    def __init__(
        self,
        dataset: LocalGeoDataset,
        roi: Optional[BoundingBox] = None,
        shuffle: bool = False,
    ) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy)
                (defaults to the bounds of ``dataset.index``)
            shuffle: if True, reshuffle data at every epoch
        """
        super().__init__(dataset, roi)
        self.dist_roi = None
        self.q_bbox = None
        self.q_path = None
        if roi is None:
            raise Exception("roi should be defined based on prompts!!!")
        
        self.shuffle = shuffle
        if not self.index.empty:
            idx = len(self.index)
            
            if idx > 0:
                center_x_roi = (roi.maxx + roi.minx) / 2
                center_y_roi = (roi.maxy + roi.miny) / 2
                
                # Find nearest patch center to ROI center
                min_dist = float('inf')
                for _, row in self.index.iterrows():
                    b = row.geometry.bounds
                    bbox = BoundingBox(b[0], b[2], b[1], b[3], dataset.crs)
                    
                    center_x_bbox = (bbox.maxx + bbox.minx) / 2
                    center_y_bbox = (bbox.maxy + bbox.miny) / 2
                    
                    dist = (center_x_bbox - center_x_roi) ** 2 + (
                        center_y_bbox - center_y_roi
                    ) ** 2
                    
                    if dist < min_dist:
                        min_dist = dist
                        self.q_bbox = bbox
                        self.q_path = row["path"]
                
                self.dist_roi = min_dist

        MessageTool.MessageLog(f"Prompt intersected with {idx} feature patches")

        if self.q_bbox is None:
            self.length = 0
        else:
            self.length = 1

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy) coordinates to index a dataset
            and the exact single file filepath
        """
        generator: Callable[[int], Iterable[int]] = range
        if self.shuffle:
            generator = torch.randperm

        for idx in generator(len(self)):
            query = {"bbox": self.q_bbox, "path": cast(str, self.q_path)}
            yield query

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length
