"""QGIS processing algorithm for building GeoSAM feature caches."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBand,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterCrs,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRange,
    QgsProcessingParameterRasterLayer,
    QgsRasterBandStats,
    QgsRectangle,
)
from qgis.gui import QgsFileWidget
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import QAction, QDockWidget
from qgis.utils import iface

from ..docs import encoder_help
from ..ui.icons import QIcon_EncoderTool
from .geosam_backend import configure_geosam_qgis_runtime
from .geosam_runtime import (
    chip_extent_rectangles_for_source,
    sanitize_path_component,
)
from .model_manager import create_model_spec_from_checkpoint, get_model_display_items

# 0 for meters, 6 for degrees, 9 for unknown
UNIT_METERS = 0
UNIT_DEGREES = 6


class SamProcessingAlgorithm(QgsProcessingAlgorithm):
    """Build a GeoSAM-compatible feature cache from a raster layer."""

    INPUT = "INPUT"
    CKPT = "CKPT"
    MODEL_TYPE = "MODEL_TYPE"
    BANDS = "BANDS"
    STRIDE = "STRIDE"
    EXTENT = "EXTENT"
    LOAD = "LOAD"
    OUTPUT = "OUTPUT"
    RANGE = "RANGE"
    RESOLUTION = "RESOLUTION"
    CRS = "CRS"
    CUDA = "CUDA"
    BATCH_SIZE = "BATCH_SIZE"
    CUDA_ID = "CUDA_ID"

    def flags(self) -> Any:
        """Return processing flags for the algorithm."""
        return super().flags() | QgsProcessingAlgorithm.FlagNoThreading

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        """Define processing inputs."""
        del config
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.INPUT,
                description=self.tr("Input raster layer or image file path"),
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                name=self.BANDS,
                description=self.tr(
                    "Select no more than 3 bands (preferably in RGB order)."
                ),
                parentLayerParameterName=self.INPUT,
                optional=True,
                allowMultiple=True,
            )
        )

        crs_param = QgsProcessingParameterCrs(
            name=self.CRS,
            description=self.tr("Target CRS (default to original CRS)"),
            optional=True,
        )
        res_param = QgsProcessingParameterNumber(
            name=self.RESOLUTION,
            description=self.tr(
                "Target resolution in meters (default to native resolution)"
            ),
            type=QgsProcessingParameterNumber.Double,
            optional=True,
            minValue=0,
            maxValue=100000,
        )
        range_param = QgsProcessingParameterRange(
            name=self.RANGE,
            description=self.tr(
                "Data value range rescaled to [0, 255] (optional fixed range)."
            ),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=None,
            optional=True,
        )
        cuda_id_param = QgsProcessingParameterNumber(
            name=self.CUDA_ID,
            description=self.tr("CUDA device id (default to 0)"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0,
            minValue=0,
            maxValue=9,
        )
        self.addParameter(
            QgsProcessingParameterExtent(
                name=self.EXTENT,
                description=self.tr("Processing extent (default to entire image)"),
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.STRIDE,
                description=self.tr("Sliding-window stride."),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=512,
                minValue=1,
                maxValue=1024,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                name=self.CKPT,
                description=self.tr("GeoSAM checkpoint path"),
                extension="pt",
            )
        )

        self.model_options = get_model_display_items()
        self.addParameter(
            QgsProcessingParameterEnum(
                name=self.MODEL_TYPE,
                description=self.tr("GeoSAM model"),
                options=[label for _model_id, label in self.model_options],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT,
                self.tr("Output feature-cache directory"),
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CUDA,
                self.tr("Use GPU if CUDA is available."),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.BATCH_SIZE,
                description=self.tr(
                    "Batch size placeholder. GeoSAM currently encodes one chip "
                    "at a time."
                ),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=1,
                maxValue=1024,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOAD,
                self.tr("Load output features in Geo-SAM tool after processing"),
                defaultValue=True,
            )
        )

        for param in (crs_param, res_param, range_param, cuda_id_param):
            param.setFlags(
                param.flags() | QgsProcessingParameterDefinition.FlagAdvanced
            )
            self.addParameter(param)

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        """Run the cache-building workflow."""
        configure_geosam_qgis_runtime(
            feedback=feedback,
            transform_context=context.transformContext(),
        )
        try:
            from geosam import BoundingBox, RasterDataset, build_model_adapter
        except ModuleNotFoundError as exc:
            raise QgsProcessingException(
                self.tr(
                    "GeoSAM dependencies are not installed yet. "
                    "Open Geo-SAM Settings and install dependencies first."
                )
            ) from exc

        self.feature_dir = ""
        self.iPatch = 0
        self.load_feature = False

        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if raster_layer is None:
            raise QgsProcessingException(
                self.invalidRasterError(parameters, self.INPUT)
            )

        selected_bands = self.parameterAsInts(parameters, self.BANDS, context)
        if len(selected_bands) == 0:
            selected_bands = list(range(1, min(3, raster_layer.bandCount()) + 1))
        if len(selected_bands) > 3:
            raise QgsProcessingException(
                self.tr("Please choose no more than three bands.")
            )
        if max(selected_bands) > raster_layer.bandCount():
            raise QgsProcessingException(
                self.tr("The chosen bands exceed the largest band number.")
            )

        checkpoint_path = self.parameterAsFile(parameters, self.CKPT, context)
        model_index = self.parameterAsEnum(parameters, self.MODEL_TYPE, context)
        stride = self.parameterAsInt(parameters, self.STRIDE, context)
        resolution = self.parameterAsDouble(parameters, self.RESOLUTION, context)
        target_crs = self.parameterAsCrs(parameters, self.CRS, context)
        extent = self.parameterAsExtent(parameters, self.EXTENT, context)
        self.load_feature = self.parameterAsBoolean(parameters, self.LOAD, context)
        use_gpu = self.parameterAsBoolean(parameters, self.CUDA, context)
        batch_size = self.parameterAsInt(parameters, self.BATCH_SIZE, context)
        range_value = self.parameterAsRange(parameters, self.RANGE, context)
        output_dir = Path(self.parameterAsString(parameters, self.OUTPUT, context))
        cuda_id = self.parameterAsInt(parameters, self.CUDA_ID, context)

        if batch_size != 1:
            feedback.pushWarning(
                self.tr(
                    "GeoSAM currently encodes one chip at a time. "
                    "Batch size is ignored."
                )
            )

        if not checkpoint_path:
            raise QgsProcessingException(self.tr("Checkpoint path is required."))

        model_id = self.model_options[model_index][0]
        device = self._select_device(use_gpu=use_gpu, cuda_id=cuda_id)
        model_spec = create_model_spec_from_checkpoint(
            checkpoint_path,
            model_id=model_id,
            device=device,
        )
        adapter = build_model_adapter(model_spec)
        if not model_spec.resolved_supports_feature_reuse:
            raise QgsProcessingException(
                self.tr("The selected model does not support reusable features.")
            )

        if target_crs is None or not target_crs.isValid():
            target_crs = raster_layer.crs()

        layer_units = (
            "degrees" if raster_layer.crs().mapUnits() == UNIT_DEGREES else "meters"
        )
        if math.isnan(resolution) or resolution == 0:
            resolution = raster_layer.rasterUnitsPerPixelX()
            target_units = layer_units
        else:
            if target_crs.mapUnits() != UNIT_METERS:
                if raster_layer.crs().mapUnits() == UNIT_DEGREES:
                    target_crs = self.estimate_utm_crs(raster_layer.extent())
                else:
                    raise QgsProcessingException(
                        self.tr(
                            "Resampling to meters is only supported when the "
                            "target CRS is metric or can be estimated from a "
                            "geographic raster."
                        )
                    )
            target_units = "meters"

        source_path = raster_layer.dataProvider().dataSourceUri()
        target_crs_text = target_crs.authid() or target_crs.toWkt()
        dataset = RasterDataset(
            source_path,
            indexes=selected_bands,
            crs=target_crs_text,
            res=resolution,
        )

        extent_value, extent_crs_text = self._resolve_processing_extent(
            parameters=parameters,
            context=context,
            raster_layer=raster_layer,
            target_crs=target_crs,
            extent=extent,
        )
        chip_rectangles = chip_extent_rectangles_for_source(
            source_path,
            bands=selected_bands,
            crs=target_crs_text,
            res=resolution,
            extent=extent_value,
            extent_crs=extent_crs_text,
            chip_size=int(max(model_spec.resolved_imgsz)),
            stride=stride,
        )
        if len(chip_rectangles) == 0:
            raise QgsProcessingException(
                self.tr("No available patch sample inside the chosen extent.")
            )

        value_range = self._normalize_value_range(
            raster_layer=raster_layer,
            selected_bands=selected_bands,
            range_value=range_value,
            extent=extent,
            extent_crs=self.parameterAsExtentCrs(parameters, self.EXTENT, context),
            context=context,
            feedback=feedback,
        )
        output_dir = output_dir / sanitize_path_component(raster_layer.name())
        output_dir.mkdir(parents=True, exist_ok=True)
        features_dir = output_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        feedback.pushInfo(f"Layer path: {source_path}")
        feedback.pushInfo(f"Layer name: {raster_layer.name()}")
        feedback.pushInfo(f"Bands selected: {selected_bands}")
        feedback.pushInfo(f"Target CRS: {target_crs.authid() or target_crs.toWkt()}")
        feedback.pushInfo(f"Target resolution: {resolution} {target_units}")
        feedback.pushInfo(f"Device type: {device or 'cpu'}")
        feedback.pushInfo(f"Patch size: {model_spec.resolved_imgsz}")
        feedback.pushInfo(f"Patch sample num: {len(chip_rectangles)}")

        rows: list[dict[str, Any]] = []
        total = 100 / len(chip_rectangles)
        start_time = time.time()
        for index, rectangle in enumerate(chip_rectangles):
            if feedback.isCanceled():
                self.load_feature = False
                feedback.pushWarning(self.tr("Processing canceled by user."))
                break

            chip_bounds = BoundingBox(
                rectangle[0],
                rectangle[1],
                rectangle[2],
                rectangle[3],
                crs=dataset.crs,
            )
            sample = dataset[chip_bounds]
            model_image = sample.to_model_image(value_range=value_range)
            encoded = adapter.encode_image(model_image)
            chip_id = f"chip_{index:06d}"
            feature_path = features_dir / f"{chip_id}.pt"
            encoded.save(feature_path)
            rows.append(
                self._manifest_row(
                    sample=sample,
                    encoded=encoded,
                    chip_id=chip_id,
                    feature_path=feature_path,
                )
            )
            self.iPatch += 1
            feedback.setProgress(int((index + 1) * total))

        if len(rows) == 0:
            raise QgsProcessingException(self.tr("No feature cache was written."))

        import geopandas as gpd

        manifest = gpd.GeoDataFrame(rows, geometry="geometry", crs=dataset.crs)
        manifest_path = output_dir / "manifest.parquet"
        manifest.to_parquet(manifest_path)
        elapsed_time = time.time() - start_time
        feedback.pushInfo(f"GeoSAM encoding completed in {elapsed_time:.3f}s.")
        self.feature_dir = str(output_dir)
        return {
            "Output feature path": self.feature_dir,
            "Patch samples saved": self.iPatch,
            "Feature folder loaded": False,
        }

    def postProcessAlgorithm(
        self,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        """Optionally load the generated cache into the plugin UI."""
        del context
        if self.load_feature and self.feature_dir:
            self.load_feature = self.load_feature_dir(feedback=feedback)
        return {
            "Output feature path": self.feature_dir,
            "Patch samples saved": self.iPatch,
            "Feature folder loaded": self.load_feature,
        }

    @staticmethod
    def _select_device(*, use_gpu: bool, cuda_id: int) -> str | None:
        """Resolve the preferred inference device string."""
        if not use_gpu:
            return None
        try:
            import torch
        except (ImportError, OSError):
            return None
        if torch.cuda.is_available():
            if cuda_id + 1 > torch.cuda.device_count():
                cuda_id = torch.cuda.device_count() - 1
            return f"cuda:{cuda_id}"
        if torch.backends.mps.is_available():
            return "mps"
        return None

    def _resolve_processing_extent(
        self,
        *,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        raster_layer,
        target_crs: QgsCoordinateReferenceSystem,
        extent: QgsRectangle,
    ) -> tuple[tuple[float, float, float, float] | None, str]:
        """Resolve processing extent and its CRS text."""
        if extent.isNull():
            return None, target_crs.authid() or target_crs.toWkt()
        if extent.isEmpty():
            raise QgsProcessingException(
                self.tr("The processing extent cannot be empty.")
            )
        extent_crs = self.parameterAsExtentCrs(parameters, self.EXTENT, context)
        if extent_crs is None or not extent_crs.isValid():
            extent_crs = raster_layer.crs()

        if extent_crs != target_crs:
            transform = QgsCoordinateTransform(
                extent_crs,
                target_crs,
                context.transformContext(),
            )
            extent_polygon = QgsGeometry.fromRect(extent)
            extent_polygon.transform(transform)
            extent = extent_polygon.boundingBox()
            extent_crs = target_crs
        return (
            (
                extent.xMinimum(),
                extent.yMinimum(),
                extent.xMaximum(),
                extent.yMaximum(),
            ),
            extent_crs.authid() or extent_crs.toWkt(),
        )

    def _normalize_value_range(
        self,
        *,
        raster_layer,
        selected_bands: list[int],
        range_value: list[float],
        extent: QgsRectangle,
        extent_crs,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> tuple[float, float] | None:
        """Resolve a fixed value range when one is requested."""
        if (
            len(range_value) >= 2
            and not math.isnan(range_value[0])
            and not math.isnan(range_value[1])
        ):
            if range_value[0] >= range_value[1]:
                raise QgsProcessingException(
                    self.tr("Data value range is invalid or constant.")
                )
            feedback.pushInfo(
                f"Input data value range to be rescaled: {range_value} (user)"
            )
            return (float(range_value[0]), float(range_value[1]))

        if extent.isNull():
            stat_extent = raster_layer.extent()
        elif extent_crs == raster_layer.crs() or extent_crs is None:
            stat_extent = extent
        else:
            transform = QgsCoordinateTransform(
                extent_crs,
                raster_layer.crs(),
                context.transformContext(),
            )
            stat_extent = transform.transformBoundingBox(extent)

        min_values: list[float] = []
        max_values: list[float] = []
        provider = raster_layer.dataProvider()
        for band in selected_bands:
            band_stats = provider.bandStatistics(
                bandNo=band,
                stats=QgsRasterBandStats.All,
                extent=stat_extent,
                sampleSize=min(int(1e8), raster_layer.width() * raster_layer.height()),
            )
            min_values.append(float(band_stats.minimumValue))
            max_values.append(float(band_stats.maximumValue))
        if len(min_values) == 0:
            return None
        resolved_range = (min(min_values), max(max_values))
        feedback.pushInfo(
            "Input data value range to be rescaled: "
            f"{resolved_range} (computed from raster statistics)"
        )
        if resolved_range[0] >= resolved_range[1]:
            return None
        return resolved_range

    @staticmethod
    def _manifest_row(
        *,
        sample,
        encoded,
        chip_id: str,
        feature_path: Path,
    ) -> dict[str, Any]:
        """Build one manifest row."""
        return {
            "feature_path": str(feature_path),
            "chip_id": chip_id,
            "source_path": sample.source_path,
            "checkpoint_path": encoded.checkpoint_path,
            "model_type": encoded.model_type,
            "transform": json.dumps(list(sample.transform)[:6]),
            "shape": json.dumps(list(sample.shape)),
            "crs": sample.crs.to_string(),
            "dst_shape": json.dumps(list(encoded.dst_shape)),
            "chip_center_x": sample.bbox.center[0],
            "chip_center_y": sample.bbox.center[1],
            "geometry": sample.bbox.to_geometry(),
        }

    def load_feature_dir(self, feedback: QgsProcessingFeedback) -> bool:
        """Open the Geo-SAM widget and load the generated feature folder."""
        sam_tool_action: QAction = iface.mainWindow().findChild(
            QAction,
            "mActionGeoSamTool",
        )
        if sam_tool_action is None:
            feedback.pushInfo("\n Geo-SAM tool action not found. \n")
            return False

        sam_tool_action.trigger()
        start_time = time.time()
        while True:
            if feedback.isCanceled():
                feedback.pushInfo(self.tr("Loading feature is canceled by user."))
                return False
            sam_tool_widget: QDockWidget = iface.mainWindow().findChild(
                QDockWidget,
                "GeoSAM",
            )
            elapsed_time = (time.time() - start_time) * 1000
            if sam_tool_widget:
                load_feature_widget: QgsFileWidget = sam_tool_widget.QgsFile_feature
                load_feature_widget.setFilePath(self.feature_dir)
                sam_tool_widget.pushButton_load_feature.click()
                loaded_message = (
                    "\n GeoSAM widget found and features loaded in "
                    f"{elapsed_time:.3f} ms \n"
                )
                feedback.pushInfo(loaded_message)
                return True
            if elapsed_time > 3000:
                feedback.pushInfo(
                    f"\n GeoSAM widget not found {elapsed_time:.3f} ms \n"
                )
                return False

    @staticmethod
    def estimate_utm_crs(extent: QgsRectangle) -> QgsCoordinateReferenceSystem:
        """Estimate a metric UTM CRS from a geographic extent."""
        center_x = (extent.xMinimum() + extent.xMaximum()) / 2.0
        center_y = (extent.yMinimum() + extent.yMaximum()) / 2.0
        zone = int((center_x + 180.0) / 6.0) + 1
        epsg_code = 32600 + zone if center_y >= 0 else 32700 + zone
        return QgsCoordinateReferenceSystem(f"EPSG:{epsg_code}")

    def tr(self, string: str) -> str:
        """Return a translated string."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        """Create the processing algorithm instance."""
        return SamProcessingAlgorithm()

    def name(self) -> str:
        """Return the provider-internal algorithm name."""
        return "geo_sam_encoder"

    def displayName(self) -> str:
        """Return the user-facing algorithm name."""
        return self.tr("Geo-SAM Image Encoder")

    def group(self) -> str:
        """Return the algorithm group label."""
        return self.tr("")

    def groupId(self) -> str:
        """Return the algorithm group id."""
        return ""

    def shortHelpString(self) -> str:
        """Return the algorithm help text."""
        help_path = encoder_help
        if not os.path.exists(help_path):
            return self.tr("Generate reusable image features using GeoSAM.")
        with open(help_path) as help_file:
            return help_file.read()

    def icon(self):
        """Return the algorithm icon."""
        return QIcon_EncoderTool
