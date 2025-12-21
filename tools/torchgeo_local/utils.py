from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterator, Sequence, Union, overload, Tuple

import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import box, Polygon

CrsLike = Union[str, int, CRS, None]

@dataclass(frozen=True)
class BoundingBox:
    """Data class for indexing spatial data.

    This is a simplified version of TorchGeo's BoundingBox, removing time
    and adding CRS support.
    """

    minx: float
    maxx: float
    miny: float
    maxy: float
    crs: CRS | None = None

    def __post_init__(self) -> None:
        """Validate the arguments passed to :meth:`__init__`.

        Raises:
            ValueError: if bounding box is invalid (minx > maxx or miny > maxy)
        """
        if self.minx > self.maxx:
            raise ValueError(
                f"Bounding box is invalid: 'minx={self.minx}' > 'maxx={self.maxx}'"
            )
        if self.miny > self.maxy:
            raise ValueError(
                f"Bounding box is invalid: 'miny={self.miny}' > 'maxy={self.maxy}'"
            )
        
        if self.crs is not None and not isinstance(self.crs, CRS):
            # Use object.__setattr__ because the dataclass is frozen
            object.__setattr__(self, 'crs', CRS.from_user_input(self.crs))

    @overload
    def __getitem__(self, key: int) -> float:
        pass

    @overload
    def __getitem__(self, key: slice) -> list[float]:
        pass

    def __getitem__(self, key: int | slice) -> float | list[float]:
        """Index the (minx, maxx, miny, maxy) tuple.

        Args:
            key: integer or slice object

        Returns:
            the value(s) at that index

        Raises:
            IndexError: if key is out of bounds
        """
        return [self.minx, self.maxx, self.miny, self.maxy][key]

    def __iter__(self) -> Iterator[float]:
        """Container iterator.

        Returns:
            iterator object that iterates over all coordinates
        """
        yield from [self.minx, self.maxx, self.miny, self.maxy]

    def __contains__(self, other: BoundingBox) -> bool:
        """Whether or not other is within the bounds of this bounding box.

        Note: CRS must match if both have CRS.
        """
        if self.crs and other.crs and self.crs != other.crs:
            other = other.to_crs(self.crs)

        return (
            (self.minx <= other.minx <= self.maxx)
            and (self.minx <= other.maxx <= self.maxx)
            and (self.miny <= other.miny <= self.maxy)
            and (self.miny <= other.maxy <= self.maxy)
        )

    def __or__(self, other: BoundingBox) -> BoundingBox:
        """The union operator."""
        if self.crs and other.crs and self.crs != other.crs:
            other = other.to_crs(self.crs)
            
        return BoundingBox(
            min(self.minx, other.minx),
            max(self.maxx, other.maxx),
            min(self.miny, other.miny),
            max(self.maxy, other.maxy),
            self.crs
        )

    def __and__(self, other: BoundingBox) -> BoundingBox:
        """The intersection operator."""
        if self.crs and other.crs and self.crs != other.crs:
            other = other.to_crs(self.crs)

        try:
            return BoundingBox(
                max(self.minx, other.minx),
                min(self.maxx, other.maxx),
                max(self.miny, other.miny),
                min(self.maxy, other.maxy),
                self.crs
            )
        except ValueError:
            raise ValueError(f"Bounding boxes {self} and {other} do not overlap")

    @property
    def area(self) -> float:
        """Area of bounding box."""
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    def intersects(self, other: BoundingBox) -> bool:
        """Whether or not two bounding boxes intersect."""
        if self.crs and other.crs and self.crs != other.crs:
            other = other.to_crs(self.crs)

        return (
            self.minx <= other.maxx
            and self.maxx >= other.minx
            and self.miny <= other.maxy
            and self.maxy >= other.miny
        )

    @property
    def geometry(self) -> Polygon:
        """Return the shapely geometry of the bounding box."""
        return box(self.minx, self.miny, self.maxx, self.maxy)

    def to_crs(self, crs: CrsLike) -> BoundingBox:
        """Return a new BoundingBox with new CRS.

        If self.crs is None, sets the new CRS without changing coordinates.
        If self.crs is present, reprojects the coordinates.
        """
        if not isinstance(crs, CRS):
            crs = CRS.from_user_input(crs)
        
        if self.crs is None:
            return BoundingBox(self.minx, self.maxx, self.miny, self.maxy, crs)
            
        if self.crs == crs:
            return self

        import geopandas as gpd
        gdf = gpd.GeoDataFrame(geometry=[self.geometry], crs=self.crs)
        gdf = gdf.to_crs(crs)
        new_minx, new_miny, new_maxx, new_maxy = gdf.total_bounds
        return BoundingBox(new_minx, new_maxx, new_miny, new_maxy, crs)

def _to_tuple(value: Union[float, Tuple[float, float]]) -> Tuple[float, float]:
    """Convert a value to a tuple of two values."""
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    return (float(value[0]), float(value[1]))

def tile_to_chips(
    bounds: BoundingBox, size: Tuple[float, float], stride: Tuple[float, float]
) -> Tuple[int, int]:
    """Compute the number of chips in a tile."""
    rows = int((bounds.maxy - bounds.miny - size[0]) // stride[0]) + 1
    cols = int((bounds.maxx - bounds.minx - size[1]) // stride[1]) + 1
    return max(0, rows), max(0, cols)
