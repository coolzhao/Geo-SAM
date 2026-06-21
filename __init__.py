"""QGIS plugin entry point for Geo-SAM."""

from __future__ import annotations

import inspect
import os

from .tools.dependency_path import register_plugin_managed_dependency_path

register_plugin_managed_dependency_path()

cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]


def classFactory(iface):
    """Create the Geo-SAM plugin instance for QGIS.

    Parameters
    ----------
    iface
        QGIS application interface provided by the plugin loader.

    Returns
    -------
    Geo_SAM
        Initialized plugin object.

    """
    from .geo_sam_tool import Geo_SAM

    return Geo_SAM(iface, cmd_folder)
