# Extension of torchgeo library by zyzhao 
import sys
import glob
import os
import re
from datetime import datetime
import math

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast, Union, Iterator, Iterable

import rasterio as rio
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from rasterio.crs import CRS
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torchgeo.datasets import unbind_samples, stack_samples, BoundingBox, RasterDataset, GeoDataset, IntersectionDataset
from torchgeo.datasets.utils import disambiguate_timestamp
from torchgeo.samplers import Units, GeoSampler, PreChippedGeoSampler, GridGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box, tile_to_chips
import matplotlib.pyplot as plt

from rtree.index import Index, Property

class TestGridGeoSampler(GridGeoSampler):
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return the index of a dataset.

        Returns:
            bbox(minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
            & raster file path
        """
        # For each tile...
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            mint = bounds.mint
            maxt = bounds.maxt

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]
                if maxy > bounds.maxy:
                    maxy = bounds.maxy
                    miny = bounds.maxy - self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    if maxx > bounds.maxx:
                        maxx = bounds.maxx
                        minx = bounds.maxx - self.size[1]
                    query = {"bbox": BoundingBox(minx, maxx, miny, maxy, mint, maxt),
                            "path": cast(str, hit.object)}

                    yield query # BoundingBox(minx, maxx, miny, maxy, mint, maxt)

class SamTestFeatureDataset(RasterDataset):
    filename_glob = "*.tif" 
    # filename_regex = r"^S2.{5}_(?P<date>\d{8})_N\d{4}_R\d{3}_6Bands_S\d{1}"
    filename_regex = ".*"
    date_format = ""
    is_image = True
    separate_files = False
    all_bands = []
    rgb_bands = []

    def __init__(self, root: str = "data",
                 crs: Optional[CRS] = None,
                 res: Optional[float] = None,
                 bands: Optional[Sequence[str]] = None,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 cache: bool = True
        ) -> None:
        if self.separate_files:
            raise NotImplementedError(
                'Testing for separated files are not supported yet'
            )
        # super().__init__(root, crs, res, bands, transforms, cache)
        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        self.root = root
        self.bands = bands or self.all_bands
        self.cache = cache

        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, "**", self.filename_glob)
        raster_list = glob.glob(pathname, recursive=True)
        dir_name = os.path.basename(root)
        csv_filepath = os.path.join(root, dir_name + '.csv')
        index_set = False
        if os.path.exists(csv_filepath):
            self.index_df = pd.read_csv(csv_filepath)
            filepath_csv = self.index_df.loc[0, 'filepath']
            if len(self.index_df) == len(raster_list) and os.path.dirname(filepath_csv) == os.path.dirname(raster_list[0]):
                for _, row_df in self.index_df.iterrows():
                    if crs is None:
                        crs = row_df['crs']
                    if res is None:
                        res = row_df['res']
                    id = row_df['id']
                    coords = (row_df['minx'], row_df['maxx'],
                              row_df['miny'], row_df['maxy'],
                              row_df['mint'], row_df['maxt'])
                    filepath = row_df['filepath']
                    self.index.insert(id, coords, filepath)
                    i += 1
                # print(coords[0].dtype)
                index_set = True
                print('index loaded from: ', os.path.basename(csv_filepath))
            else:
                print('index file does not match the raster list, it will be recreated.')
        if not index_set:
            self.index_df = pd.DataFrame(columns = ['id',
                                                    'minx', 'maxx', 'miny', 'maxy', 'mint', 'maxt',
                                                    'filepath',
                                                    'crs', 'res'])
            id_list = []
            coords_list = []
            filepath_list = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for filepath in raster_list: # glob.iglob(pathname, recursive=True):
                match = re.match(filename_regex, os.path.basename(filepath))
                if match is not None:
                    try:
                        with rio.open(filepath) as src:
                            # See if file has a color map
                            if len(self.cmap) == 0:
                                try:
                                    self.cmap = src.colormap(1)
                                except ValueError:
                                    pass

                            if crs is None:
                                crs = src.crs
                            if res is None:
                                res = src.res[0]

                            with WarpedVRT(src, crs=crs) as vrt:
                                minx, miny, maxx, maxy = vrt.bounds
                    except rio.errors.RasterioIOError:
                        # Skip files that rasterio is unable to read
                        continue
                    else:
                        mint: float = 0
                        maxt: float = sys.maxsize
                        if "date" in match.groupdict():
                            date = match.group("date")
                            mint, maxt = disambiguate_timestamp(date, self.date_format)

                        coords = (minx, maxx, miny, maxy, mint, maxt)
                        self.index.insert(i, coords, filepath)
                        id_list.append(i)
                        coords_list.append(coords)
                        filepath_list.append(filepath)
                        i += 1
            self.index_df['id'] = id_list
            self.index_df['filepath'] = filepath_list
            self.index_df['minx'] = pd.to_numeric([coord[0] for coord in coords_list], downcast='float')
            self.index_df['maxx'] = pd.to_numeric([coord[1] for coord in coords_list], downcast='float')
            self.index_df['miny'] = pd.to_numeric([coord[2] for coord in coords_list], downcast='float')
            self.index_df['maxy'] = pd.to_numeric([coord[3] for coord in coords_list], downcast='float')
            self.index_df['mint'] = pd.to_numeric([coord[4] for coord in coords_list], downcast='float')
            self.index_df['maxt'] = pd.to_numeric([coord[5] for coord in coords_list], downcast='float')
            # print(type(crs), res)
            self.index_df.loc[:, 'crs'] = str(crs)
            self.index_df.loc[:, 'res'] = res
            # print(self.index_df.dtypes)
            index_set = True
            self.index_df.to_csv(csv_filepath)
            print('index file: ', os.path.basename(csv_filepath), ' saved')

        if i == 0:
            msg = f"No {self.__class__.__name__} data was found in `root='{self.root}'`"
            if self.bands:
                msg += f" with `bands={self.bands}`"
            raise FileNotFoundError(msg)

        if not self.separate_files:
            self.band_indexes = None
            if self.bands:
                if self.all_bands:
                    self.band_indexes = [
                        self.all_bands.index(i) + 1 for i in self.bands
                    ]
                else:
                    msg = (
                        f"{self.__class__.__name__} is missing an `all_bands` "
                        "attribute, so `bands` cannot be specified."
                    )
                    raise AssertionError(msg)

        self._crs = cast(CRS, crs)
        self.res = cast(float, res)


    def __getitem__(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        # query may include the prompt points, bbox, or mask
        bbox = query['bbox']
        filepath = query['path']

        hits = self.index.intersection(tuple(bbox), objects=True)
        filepaths = cast(List[str], [hit.object for hit in hits])

        if not filepath in filepaths:
            raise IndexError(
                f"query: {bbox} not found in index with bounds: {self.bounds}"
            )

        if self.cache:
            vrt_fh = self._cached_load_warp_file(filepath)
        else:
            vrt_fh = self._load_warp_file(filepath)

        # bounds = (bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)
        # band_indexes = self.band_indexes

        src = vrt_fh
        dest = src.read() # read all bands
        # print(src.profile)
        # print(src.compression)

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest) # .float()

        # bbox may be useful to form the final mask results (geo-info)
        sample = {"crs": self.crs, "bbox": bbox, "path": filepath}
        if self.is_image:
            sample["image"] = tensor.float()
        else:
            sample["mask"] = tensor # .float() #long() # modified zyzhao

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

class SamTestRasterDataset(RasterDataset):
    filename_glob = "*.tif" 
    # filename_regex = r"^S2.{5}_(?P<date>\d{8})_N\d{4}_R\d{3}_6Bands_S\d{1}"
    filename_regex = ".*"
    date_format = ""
    is_image = True
    separate_files = False
    all_bands = ['Red', 'Green', 'Blue', 'Alpha']
    rgb_bands = ['Red', 'Green', 'Blue']

    def __init__(self, root: str = "data",
                 crs: Optional[CRS] = None,
                 res: Optional[float] = None,
                 bands: Optional[Sequence[str]] = None,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 cache: bool = True
        ) -> None:
        if self.separate_files:
            raise NotImplementedError(
                'Testing for separated files are not supported yet'
            )
        super().__init__(root, crs, res, bands, transforms, cache)

    def __getitem__(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        bbox = query['bbox']
        filepath = query['path']

        hits = self.index.intersection(tuple(bbox), objects=True)
        filepaths = cast(List[str], [hit.object for hit in hits])

        if not filepath in filepaths:
            raise IndexError(
                f"query: {bbox} not found in index with bounds: {self.bounds}"
            )

        if self.cache:
            vrt_fh = self._cached_load_warp_file(filepath)
        else:
            vrt_fh = self._load_warp_file(filepath)

        bounds = (bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)
        band_indexes = self.band_indexes

        src = vrt_fh
        # out_width = round((bbox.maxx - bbox.minx) / self.res)
        # out_height = round((bbox.maxy - bbox.miny) / self.res)
        out_width = math.ceil((bbox.maxx - bbox.minx) / self.res)
        out_height = math.ceil((bbox.maxy - bbox.miny) / self.res)
        count = len(band_indexes) if band_indexes else src.count
        out_shape = (count, out_height, out_width)
        dest = src.read(
            indexes=band_indexes,
            out_shape=out_shape,
            window=from_bounds(*bounds, src.transform),
        )

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest) # .float()

        sample = {"crs": self.crs, "bbox": bbox, "path": filepath}
        if self.is_image:
            sample["image"] = tensor.float()
        else:
            sample["mask"] = tensor # .float() #long() # modified zyzhao

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

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
        image = torch.clamp(image*bright/255, min=0, max=1)

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')

        return fig

class SamTestFeatureGeoSampler(GeoSampler):
    """Samples entire files at a time.

    This is particularly useful for datasets that contain geospatial metadata
    and subclass :class:`~torchgeo.datasets.GeoDataset` but have already been
    pre-processed into :term:`chips <chip>`.

    This sampler should not be used with :class:`~torchgeo.datasets.NonGeoDataset`.
    You may encounter problems when using an :term:`ROI <region of interest (ROI)>`
    that partially intersects with one of the file bounding boxes, when using an
    :class:`~torchgeo.datasets.IntersectionDataset`, or when each file is in a
    different CRS. These issues can be solved by adding padding.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        feature_size: float,
        roi: Optional[BoundingBox] = None,
        shuffle: bool = False,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        .. versionadded:: 0.3

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            shuffle: if True, reshuffle data at every epoch
        """

        self.res = dataset.res
        self.dist_roi = ((feature_size*dataset.res/2)**2)*2
        self.q_bbox = None
        self.q_path = None
        if roi is None:
            # self.index = dataset.index
            # roi = BoundingBox(*self.index.bounds)
            raise Exception('roi should be defined based on prompts!!!')
        else:
            self.index = Index(interleaved=False, properties=Property(dimension=3))
            hits = dataset.index.intersection(tuple(roi), objects=True)
            # hit_nearest = list(dataset.index.nearest(tuple(roi), num_results=1, objects=True))[0]
            # print('nearest hit: ', hit_nearest.object)
            idx = 0
            for hit in hits:
                idx += 1
                bbox = BoundingBox(*hit.bounds)  # & roi
                # print(bbox)
                center_x_bbox = (bbox.maxx + bbox.minx)/2
                center_y_bbox = (bbox.maxy + bbox.miny)/2

                center_x_roi = (roi.maxx + roi.minx)/2
                center_y_roi = (roi.maxy + roi.miny)/2
                dist_roi_tmp = (center_x_bbox - center_x_roi)**2 + (center_y_bbox - center_y_roi)**2
                # print(dist_roi_tmp)
                if dist_roi_tmp < self.dist_roi:
                    self.dist_roi = dist_roi_tmp
                    self.q_bbox = bbox
                    self.q_path = hit.object
                # self.index.insert(hit.id, tuple(bbox), hit.object)

        print('intersected features: ', idx)
        # print('selected hit: ', self.q_path)
        if self.q_bbox is None:
            self.length = 0
            # raise Exception('no feature found intersected with prompts')
        else:
            self.length = 1
        self.roi = roi

        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
            and the exact single file filepath
        """
        generator: Callable[[int], Iterable[int]] = range
        if self.shuffle:
            generator = torch.randperm

        for idx in generator(len(self)):
            query = {"bbox": self.q_bbox,
                     "path": cast(str, self.q_path)}
            yield query

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length #len(self.q_path)

