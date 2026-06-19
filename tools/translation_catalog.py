"""Extraction-only catalog for dynamically translated user messages.

This module is scanned by ``pylupdate5``. It is not imported at runtime;
:mod:`tools.messageTool` translates these strings immediately before display.
"""

from __future__ import annotations

from qgis.PyQt.QtCore import QCoreApplication

DYNAMIC_USER_TEXT = (
    QCoreApplication.translate("GeoSAM", "Warning"),
    QCoreApplication.translate("GeoSAM", "Geo-SAM"),
    QCoreApplication.translate("GeoSAM", "Note:"),
    QCoreApplication.translate("GeoSAM", "Attention"),
    QCoreApplication.translate("GeoSAM", "Documentation"),
    QCoreApplication.translate("GeoSAM", "GitHub"),
    QCoreApplication.translate("GeoSAM", "Report Bug"),
    QCoreApplication.translate("GeoSAM", "Discussions"),
    QCoreApplication.translate("GeoSAM", "Installed"),
    QCoreApplication.translate("GeoSAM", "Missing"),
    QCoreApplication.translate("GeoSAM", "Geo-SAM managed"),
    QCoreApplication.translate("GeoSAM", "QGIS runtime or Geo-SAM"),
    QCoreApplication.translate("GeoSAM", "QGIS runtime"),
    QCoreApplication.translate("GeoSAM", "Model Not Downloaded"),
    QCoreApplication.translate("GeoSAM", "Input Required"),
    QCoreApplication.translate("GeoSAM", "Model Required"),
    QCoreApplication.translate("GeoSAM", "Geo-SAM Query Failed"),
    QCoreApplication.translate("GeoSAM", "Geo-SAM Dependencies Missing"),
    QCoreApplication.translate("GeoSAM", "Invalid Feature Folder"),
    QCoreApplication.translate("GeoSAM", "Feature Folder Failed"),
    QCoreApplication.translate("GeoSAM", "Patch Preview Failed"),
    QCoreApplication.translate(
        "GeoSAM",
        "The selected model is not downloaded yet.\n"
        "1. Open Geo-SAM Settings and download the checkpoint.\n"
        "2. Documentation: https://geo-sam.readthedocs.io/en/latest/",
    ),
    QCoreApplication.translate(
        "GeoSAM",
        "The selected model is not downloaded yet.\n"
        "Open Geo-SAM Settings to download it.",
    ),
    QCoreApplication.translate(
        "GeoSAM",
        "The model recorded in the selected feature folder is not downloaded yet.\n"
        "Open Geo-SAM Settings to download it.",
    ),
    QCoreApplication.translate(
        "GeoSAM", "Please choose a RealTime Layer or a Feature folder first."
    ),
    QCoreApplication.translate("GeoSAM", "Please choose a SAM model first."),
    QCoreApplication.translate("GeoSAM", "No input source loaded."),
    QCoreApplication.translate(
        "GeoSAM", "Choose an input source before configuring the output layer."
    ),
    QCoreApplication.translate("GeoSAM", "Feature folder does not exist."),
    QCoreApplication.translate("GeoSAM", "Project CRS has been reset to original CRS."),
    QCoreApplication.translate(
        "GeoSAM",
        "Project CRS has been changed to the active source CRS temporarily, "
        "and will be reset to original CRS when this widget is closed.",
    ),
    QCoreApplication.translate(
        "GeoSAM",
        "Project crs has been changed to the layer crs temporarily. "
        "It will be reset to the original crs when this widget is closed.",
    ),
    QCoreApplication.translate(
        "GeoSAM",
        "RealTime Layer takes precedence. Clear it to use the feature folder.",
    ),
    QCoreApplication.translate(
        "GeoSAM",
        "Max Polygon Only is enabled. All prompts are still queried, but only "
        "the largest polygon from the current mask is shown.",
    ),
    QCoreApplication.translate(
        "GeoSAM",
        "Preview mode only shows prompt previews. Click first to apply the prompt.",
    ),
    QCoreApplication.translate(
        "GeoSAM",
        "Point or rectangle is outside the source boundary. "
        "Click OK to undo the last prompt.",
    ),
    QCoreApplication.translate(
        "GeoSAM", "Oops: Invalid Raster Layer. Please select a valid raster layer!"
    ),
    QCoreApplication.translate(
        "GeoSAM", "Oops: Extent has not been set. Please set extent first."
    ),
    QCoreApplication.translate(
        "GeoSAM",
        "Oops: Raster Layer has not been selected/detected. "
        "Please set/reset Raster Layer first!",
    ),
    QCoreApplication.translate(
        "GeoSAM",
        "Setting has been copied to clipboard. You can paste it to Geo-SAM Image "
        "Encoder or a json file now.",
    ),
    QCoreApplication.translate(
        "GeoSAM",
        "Oops: Failed to save setting to file. Please choose a valid directory first.",
    ),
    QCoreApplication.translate(
        "GeoSAM",
        "No available patch sample inside the chosen extent!!! "
        "Please choose another extent.",
    ),
    QCoreApplication.translate("GeoSAM", "Oops!!!"),
    QCoreApplication.translate(
        "GeoSAM", "Oops: Failed to open file, please choose a existing folder"
    ),
    QCoreApplication.translate(
        "GeoSAM",
        "The fields of this vector do not match the SAM feature fields. "
        "Please select a correct existed file or a new file to create it.",
    ),
    QCoreApplication.translate(
        "GeoSAM",
        "Output Shapefile is not specified. A temporal layer 'polygon_sam' is "
        "created, remember to save it before quit.",
    ),
)
