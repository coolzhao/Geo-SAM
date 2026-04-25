"""QGIS processing provider for Geo-SAM algorithms."""

from __future__ import annotations

import logging

from qgis.core import QgsProcessingProvider
# from qgis.PyQt.QtGui import QIcon

from .ui.icons import QIcon_EncoderTool
# from processing_provider.example_processing_algorithm import ExampleProcessingAlgorithm

logger = logging.getLogger(__name__)


class GeoSamProvider(QgsProcessingProvider):

    def loadAlgorithms(self, *args, **kwargs):
        """Load processing algorithms when QGIS initializes the provider.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded by QGIS.
        **kwargs : Any
            Keyword arguments forwarded by QGIS.
        """
        del args, kwargs
        try:
            from .tools.sam_processing_algorithm import SamProcessingAlgorithm
        except ModuleNotFoundError as exc:
            logger.warning(
                "Geo-SAM processing algorithm was not loaded because dependency "
                "is missing: %s",
                getattr(exc, "name", exc),
            )
            return

        self.addAlgorithm(SamProcessingAlgorithm())
        # add additional algorithms here
        # self.addAlgorithm(MyOtherAlgorithm())

    def id(self, *args, **kwargs):
        """The ID of your plugin, used for identifying the provider.

        This string should be a unique, short, character only string,
        eg "qgis" or "gdal". This string should not be localised.
        """
        return 'geo_sam'

    def name(self, *args, **kwargs):
        """The human friendly name of your plugin in Processing.

        This string should be as short as possible (e.g. "Lastools", not
        "Lastools version 1.0.1 64-bit") and localised.
        """
        return self.tr('Geo-SAM')

    def icon(self):
        """Should return a QIcon which is used for your provider inside
        the Processing toolbox.
        """
        return QIcon_EncoderTool

    def longName(self) -> str:
        return self.name()
