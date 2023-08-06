# Extension of torchgeo library by zyzhao
import sys
import glob
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast, Union, Iterator, Iterable
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.enums import Resampling
from rasterio.crs import CRS
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torchgeo.datasets import BoundingBox, RasterDataset, GeoDataset
from torchgeo.datasets.utils import disambiguate_timestamp
from torchgeo.samplers import Units, GeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple, tile_to_chips
import matplotlib.pyplot as plt

from rtree.index import Index, Property


class SamTestGridGeoSampler(GeoSampler):
    """Samples elements in a grid-like fashion.
    accept image smaller than desired patch_size
    """

    def __init__(
        self,
        dataset: GeoDataset,
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

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            stride: distance to skip between each patch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.patch_size = self.size
        self.stride = _to_tuple(stride)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res,
                           self.stride[1] * self.res)

        self.hits = []
        self.hits_small = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                # change 'and' to 'or' for handling strip images
                or bounds.maxy - bounds.miny >= self.size[0]
            ):
                self.hits.append(hit)
            else:
                self.hits_small.append(hit)

        self.length = 0
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            self.length += rows * cols

        for hit in self.hits_small:
            bounds = BoundingBox(*hit.bounds)
            self.length += 1

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return the index of a dataset.

        Returns:
            bbox(minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
            & raster file path
        """
        # For each tile...
        for hit in self.hits + self.hits_small:
            if hit in self.hits:
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
                            "bbox": BoundingBox(minx, maxx, miny, maxy, mint, maxt),
                            "path": cast(str, hit.object),
                            "size": int(self.patch_size[0])
                        }

                        # BoundingBox(minx, maxx, miny, maxy, mint, maxt)
                        yield query
            else:
                bounds = BoundingBox(*hit.bounds)
                minx = bounds.minx
                miny = bounds.miny
                maxx = bounds.maxx
                maxy = bounds.maxy
                mint = bounds.mint
                maxt = bounds.maxt
                query = {
                    "bbox": BoundingBox(minx, maxx, miny, maxy, mint, maxt),
                    "path": cast(str, hit.object),
                    "size": int(self.patch_size[0])
                }

                yield query

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length


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
                 transforms: Optional[Callable[[
                     Dict[str, Any]], Dict[str, Any]]] = None,
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
        self.model_type = None
        model_type_list = ["vit_h", "vit_l", "vit_b"]

        # Populate the dataset index
        i = 0
        # pathname = os.path.join(root, "**", self.filename_glob)
        pathname = os.path.join(root, self.filename_glob)
        raster_list = glob.glob(pathname, recursive=True)
        raster_name = os.path.basename(raster_list[0])
        for m_type in model_type_list:
            if m_type in raster_name:
                self.model_type = m_type
                break
        dir_name = os.path.basename(root)
        csv_filepath = os.path.join(root, dir_name + '.csv')
        index_set = False
        if os.path.exists(csv_filepath):
            self.index_df = pd.read_csv(csv_filepath)
            # filepath_csv = self.index_df.loc[0, 'filepath']
            # and os.path.dirname(filepath_csv) == os.path.dirname(raster_list[0]):
            if len(self.index_df) == len(raster_list):
                for _, row_df in self.index_df.iterrows():
                    if crs is None:
                        crs = row_df['crs']
                    if res is None:
                        res = row_df['res']
                    # id = row_df['id']
                    coords = (row_df['minx'], row_df['maxx'],
                              row_df['miny'], row_df['maxy'],
                              row_df['mint'], row_df['maxt'])
                    # change to relative path
                    filepath = os.path.join(
                        root, os.path.basename(row_df['filepath']))
                    self.index.insert(i, coords, filepath)
                    i += 1
                # print(coords[0].dtype)
                index_set = True
                # print('index loaded from: ', os.path.basename(csv_filepath))
                print(f"Index loaded from: {os.path.basename(csv_filepath)}")
            else:
                # print('index file does not match the raster list, it will be recreated.')
                print(f"Index file does not match the raster list, will be recreated.")

        if not index_set:
            self.index_df = pd.DataFrame(columns=['id',
                                                  'minx', 'maxx', 'miny', 'maxy', 'mint', 'maxt',
                                                  'filepath',
                                                  'crs', 'res'])
            id_list = []
            coords_list = []
            filepath_list = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            # glob.iglob(pathname, recursive=True):
            for filepath in raster_list:
                match = re.match(filename_regex, os.path.basename(filepath))
                if match is not None:
                    try:
                        with rasterio.open(filepath) as src:
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
                    except rasterio.errors.RasterioIOError:
                        # Skip files that rasterio is unable to read
                        continue
                    else:
                        mint: float = 0
                        maxt: float = sys.maxsize
                        if "date" in match.groupdict():
                            date = match.group("date")
                            mint, maxt = disambiguate_timestamp(
                                date, self.date_format)

                        coords = (minx, maxx, miny, maxy, mint, maxt)
                        self.index.insert(i, coords, filepath)
                        id_list.append(i)
                        coords_list.append(coords)
                        # change to relative path
                        filepath_list.append(os.path.basename(filepath))
                        i += 1
            if i > 0:
                self.index_df['id'] = id_list
                self.index_df['filepath'] = filepath_list
                self.index_df['minx'] = [coord[0] for coord in coords_list]
                self.index_df['maxx'] = [coord[1] for coord in coords_list]
                self.index_df['miny'] = [coord[2] for coord in coords_list]
                self.index_df['maxy'] = [coord[3] for coord in coords_list]
                self.index_df['mint'] = [coord[4] for coord in coords_list]
                self.index_df['maxt'] = [coord[5] for coord in coords_list]
                self.index_df.loc[:, 'crs'] = str(crs)
                self.index_df.loc[:, 'res'] = res
                # print(self.index_df.dtypes)
                index_set = True
                self.index_df.to_csv(csv_filepath)
                # print('index file: ', os.path.basename(csv_filepath), ' saved')
                print(f"Index file: {os.path.basename(csv_filepath)} saved")

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
        dest = src.read()  # read all bands
        # print(src.profile)
        # print(src.compression)
        tags = src.tags()
        if 'img_shape' in tags.keys():
            img_shape = tags['img_shape']
            input_shape = tags['input_shape']

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)  # .float()

        # bbox may be useful to form the final mask results (geo-info)
        sample = {"crs": self.crs, "bbox": bbox, "path": filepath}
        if 'img_shape' in tags.keys():
            # convert string to python data structure
            sample['img_shape'] = eval(img_shape)
            sample['input_shape'] = eval(input_shape)

        if self.is_image:
            sample["image"] = tensor.float()
        else:
            sample["mask"] = tensor  # .float() #long() # modified zyzhao

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class SamTestRasterDataset(RasterDataset):
    filename_glob = "*.tif"
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
                 transforms: Optional[Callable[[
                     Dict[str, Any]], Dict[str, Any]]] = None,
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
        patch_size = query['size']

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
        out_width = round((bbox.maxx - bbox.minx) / self.res)
        out_height = round((bbox.maxy - bbox.miny) / self.res)
        # out_width = math.ceil((bbox.maxx - bbox.minx) / self.res)
        # out_height = math.ceil((bbox.maxy - bbox.miny) / self.res)
        count = len(band_indexes) if band_indexes else src.count
        out_shape = (count, out_height, out_width)
        # if out_height == patch_size or out_width == patch_size:
        #     dest = src.read(
        #         indexes=band_indexes,
        #         out_shape=out_shape,
        #         window=window_from_bounds(*bounds, src.transform),
        #     )
        # else:
        # resize
        if out_height == 0 or out_width == 0:
            raise Exception("Patch size should be greater than zero.")

        target_height, target_width = self.get_preprocess_shape(
            out_height, out_width, patch_size)
        target_shape = (count, target_height, target_width)
        dest = src.read(
            indexes=band_indexes,
            out_shape=target_shape,
            window=window_from_bounds(*bounds, src.transform),
            resampling=Resampling.bilinear,
        )

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)  # .float()
        if torch.isnan(tensor).any():
            # , posinf=0.0, neginf=0.0
            tensor = torch.nan_to_num(tensor, nan=0.0)
        tensor = self.pad_patch(tensor, patch_size)

        sample = {"crs": self.crs, "bbox": bbox,
                  "path": filepath, "img_shape": out_shape, "input_shape": target_shape}
        if self.is_image:
            sample["image"] = tensor.float()
        else:
            sample["mask"] = tensor  # .float() #long() # modified zyzhao

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @staticmethod
    def get_preprocess_shape(old_h: int, old_w: int, long_side_length: int) -> Tuple[int, int]:
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
        roi: Optional[BoundingBox] = None,
        shuffle: bool = False,
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
        # self.dist_roi = ((feature_size*dataset.res/2)**2)*2
        self.dist_roi = None
        self.q_bbox = None
        self.q_path = None
        if roi is None:
            # self.index = dataset.index
            # roi = BoundingBox(*self.index.bounds)
            raise Exception('roi should be defined based on prompts!!!')
        else:
            self.index = Index(interleaved=False,
                               properties=Property(dimension=3))
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
                dist_roi_tmp = (center_x_bbox - center_x_roi)**2 + \
                    (center_y_bbox - center_y_roi)**2
                # print(dist_roi_tmp)
                if idx == 1:
                    self.dist_roi = dist_roi_tmp
                    self.q_bbox = bbox
                    self.q_path = hit.object
                elif dist_roi_tmp < self.dist_roi:
                    self.dist_roi = dist_roi_tmp
                    self.q_bbox = bbox
                    self.q_path = hit.object
                # self.index.insert(hit.id, tuple(bbox), hit.object)

        # print('intersected features: ', idx)
        print(f"Prompt intersected with {idx} feature patches")

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
        return self.length  # len(self.q_path)
