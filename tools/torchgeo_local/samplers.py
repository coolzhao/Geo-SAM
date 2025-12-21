from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterator, Optional, Union

from .datasets import LocalGeoDataset
from .utils import BoundingBox

class Units(Enum):
    """Units for grid-based sampling."""
    PIXELS = 0
    CRS = 1

class LocalGeoSampler:
    """Base class for geospatial samplers."""

    def __init__(
        self,
        dataset: LocalGeoDataset,
        roi: Optional[BoundingBox] = None,
    ) -> None:
        self.dataset = dataset
        self.res = dataset.res
        if roi is None:
            self.roi = dataset.bounds
        else:
            self.roi = roi
        
        self.index = dataset.index
        # Perform initial intersection with ROI
        if self.roi is not None:
            self.index = self.index[self.index.intersects(self.roi.geometry)]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.index)
