from __future__ import annotations

import os
import glob
import re
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset
from rasterio.crs import CRS
from pyproj import CRS as pyprojCRS
from .utils import BoundingBox

class LocalGeoDataset(Dataset[Dict[str, Any]]):
    """Base class for geospatial datasets."""

    def __init__(self) -> None:
        self.index = gpd.GeoDataFrame(columns=["geometry", "path"], geometry="geometry")
        self._crs: CRS | None = None
        self.res: float = 1.0

    @property
    def crs(self) -> CRS | None:
        return self._crs

    @property
    def bounds(self) -> BoundingBox:
        if self.index.empty:
            return BoundingBox(0, 0, 0, 0, self.crs)
        minx, miny, maxx, maxy = self.index.total_bounds
        return BoundingBox(minx, maxx, miny, maxy, self.crs)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        raise NotImplementedError


class LocalRasterDataset(LocalGeoDataset):
    """Base class for raster-based geospatial datasets."""

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self._crs = crs
        self.res = res or 1.0
        self.transforms = transforms
        self.cache = cache
        self._vrt_cache: Dict[str, Any] = {}

    def _load_warp_file(self, path: str) -> Any:
        if path in self._vrt_cache:
            return self._vrt_cache[path]
        
        src = rasterio.open(path)
        if self.cache:
            self._vrt_cache[path] = src
        return src

    def __getitem__(self, query: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation will be handled by subclasses for now or simplified here
        pass

def stack_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack a list of samples into a single batch."""
    collated: Dict[str, Any] = {}
    for key in samples[0]:
        val = [sample[key] for sample in samples]
        if isinstance(val[0], Tensor):
            collated[key] = torch.stack(val)
        elif isinstance(val[0], (int, float)):
            collated[key] = torch.tensor(val)
        else:
            collated[key] = val
    return collated
