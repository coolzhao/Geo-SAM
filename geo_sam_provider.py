import os
from qgis.core import QgsProcessingProvider
# from qgis.PyQt.QtGui import QIcon

from .ui.icons import QIcon_GeoSAMEncoder
# from processing_provider.example_processing_algorithm import ExampleProcessingAlgorithm
from .tools.sam_processing_algorithm import SamProcessingAlgorithm


class GeoSamProvider(QgsProcessingProvider):

    def loadAlgorithms(self, *args, **kwargs):
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
        return QIcon_GeoSAMEncoder

    def longName(self) -> str:
        return self.name()
