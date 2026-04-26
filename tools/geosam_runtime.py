"""GeoSAM-backed runtime helpers for the QGIS plugin."""

from __future__ import annotations

import gc
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal
from urllib.parse import urlparse

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsGeometry,
    QgsRasterLayer,
    QgsRectangle,
)

from .geosam_backend import configure_geosam_qgis_runtime
from .messageTool import MessageTool
from .model_manager import (
    create_model_spec,
    effective_model_imgsz,
    get_model_definition,
    infer_model_id_from_checkpoint_path,
)
from .online_tile_export import (
    OnlineRasterExportError,
    OnlineTileExportPlan,
    OnlineRasterPreflightError,
)
from .online_tile_export import (
    export_online_raster_source as export_online_tile_raster_source,
)
from .online_tile_export import (
    export_online_raster_plan as export_online_tile_raster_plan,
)
from .online_tile_export import (
    online_export_failure_message as describe_online_export_failure,
)
from .online_tile_export import (
    prepare_online_raster_export_plan,
)
from .plugin_settings import (
    clear_cache,
    cleanup_cache,
    get_cache_directory,
    initialize_rasterio_proj_data,
    load_plugin_settings,
)

if TYPE_CHECKING:
    from geosam.engines import QueryResult
    from geosam.models import ModelSpec
    from geosam.query import BoundingBox, Points, PromptSet

logger = logging.getLogger(__name__)

ONLINE_QUERY_REFRESH_MARGIN_PIXELS = 128
PerformanceMode = Literal["balanced", "fastest", "low_memory"]
PreviewRenderMode = Literal["pixel_level", "simplified"]

_DATACLASS_SLOTS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}
_FROZEN_DATACLASS_KWARGS = {"frozen": True, **_DATACLASS_SLOTS_KWARGS}
_MUTABLE_DATACLASS_KWARGS = dict(_DATACLASS_SLOTS_KWARGS)

@dataclass(**_FROZEN_DATACLASS_KWARGS)
class FeatureSourceSummary:
    """Summary metadata for a cached GeoSAM feature source."""

    manifest_path: Path
    crs_text: str
    extent: tuple[float, float, float, float]
    chip_count: int
    pixel_area: float
    model_id: str | None = None
    checkpoint_path: str | None = None


@dataclass(**_MUTABLE_DATACLASS_KWARGS)
class RealtimeQueryCache:
    """Session cache for realtime raster queries."""

    layer_id: str | None = None
    model_id: str | None = None
    source_candidate: str | None = None
    engine: Any | None = None
    query_cache: Any | None = None

    def clear(self) -> None:
        """Reset the cache state."""
        self.layer_id = None
        self.model_id = None
        self.source_candidate = None
        self.engine = None
        self.query_cache = None


@dataclass(**_FROZEN_DATACLASS_KWARGS)
class PreparedPersistentQueryCacheHit:
    """Persistent realtime-query cache metadata prepared on the main thread.

    Parameters
    ----------
    source_path : str
        Raster source path stored with the reusable encoded chip.
    feature_path : Path
        Path to the persisted encoded image features.
    chip_bounds : Any
        GeoSAM bounding box for the encoded chip.
    chip_grid : Any
        GeoSAM grid for the encoded chip.
    query : Any
        Query projected into the encoded chip CRS on the main thread.
    query_bounds_value : Any
        Query bounds prepared on the main thread for result metadata.
    prompt_kwargs : dict[str, Any]
        Model prompt arguments prepared on the main thread.
    crs_text : str
        Chip CRS text prepared on the main thread for cache metadata.

    """

    source_path: str
    feature_path: Path
    chip_bounds: Any
    chip_grid: Any
    query: Any
    query_bounds_value: Any
    prompt_kwargs: dict[str, Any]
    crs_text: str


@dataclass(**_FROZEN_DATACLASS_KWARGS)
class PreparedRealtimeRasterSample:
    """Raster chip data prepared on the main thread for background encoding.

    Parameters
    ----------
    source_path : str
        Raster source used to read the prepared chip.
    model_image : Any
        Chip image converted to model-ready ``HWC`` uint8 format.
    chip_bounds : Any
        GeoSAM bounding box for the prepared chip.
    chip_grid : Any
        GeoSAM grid for the prepared chip.
    query : Any
        Query projected into the prepared chip CRS on the main thread.
    query_bounds_value : Any
        Query bounds prepared on the main thread for result metadata.
    prompt_kwargs : dict[str, Any]
        Model prompt arguments prepared on the main thread.
    crs_text : str
        Chip CRS text prepared on the main thread for cache metadata.

    """

    source_path: str
    model_image: Any
    chip_bounds: Any
    chip_grid: Any
    query: Any
    query_bounds_value: Any
    prompt_kwargs: dict[str, Any]
    crs_text: str


RealtimeQueryProgressCallback = Callable[[str, float], None]
RealtimeQueryCancelCallback = Callable[[], bool]


class _RealtimeQueryCanceledError(RuntimeError):
    """Internal exception used to stop a background realtime query task."""


@dataclass(**_FROZEN_DATACLASS_KWARGS)
class PreparedRealtimeRasterQuery:
    """Prepared background-query inputs for a realtime raster request.

    Parameters
    ----------
    model_id : str
        Selected GeoSAM model identifier.
    layer_id : str
        Source QGIS layer id captured on the main thread.
    layer_name : str
        Human-readable layer name used in cache metadata.
    source_fingerprint : str
        Stable fingerprint used to validate persistent cache entries.
    layer_crs_text : str
        Active realtime layer CRS captured on the main thread.
    cache_directory : Path
        Directory where realtime query caches are persisted.
    online_raster_cache_directory : Path | None
        Cache directory used for exported online raster snapshots.
    source_candidates : tuple[str, ...]
        Candidate raster sources that can be opened without touching QGIS APIs.
    prepared_source_samples : tuple[PreparedRealtimeRasterSample, ...]
        Raster chips prepared on the main thread for local source candidates.
    online_export_plan : OnlineTileExportPlan | None
        Prepared online export plan for non-local raster sources.
    supports_feature_reuse : bool
        Whether the selected model supports reusable encoded features.
    persistent_cache_hit : PreparedPersistentQueryCacheHit | None, optional
        Metadata for a reusable persistent cache entry. CRS-heavy parsing is
        done on the main thread; encoded features are loaded in the worker.

    """

    model_id: str
    layer_id: str
    layer_name: str
    source_fingerprint: str
    layer_crs_text: str
    cache_directory: Path
    online_raster_cache_directory: Path | None
    source_candidates: tuple[str, ...]
    prepared_source_samples: tuple[PreparedRealtimeRasterSample, ...]
    online_export_plan: OnlineTileExportPlan | None
    supports_feature_reuse: bool
    persistent_cache_hit: PreparedPersistentQueryCacheHit | None = None


@dataclass(**_MUTABLE_DATACLASS_KWARGS)
class PreparedRealtimeRasterQueryResult:
    """Result payload returned by a background realtime raster query.

    Parameters
    ----------
    source_path : str
        Raster source used by the background task.
    result : QueryResult
        GeoSAM query result.
    query_cache : Any | None, optional
        In-memory query cache produced during encoding when feature reuse is
        supported.

    """

    source_path: str
    result: QueryResult
    query_cache: Any | None = None


@dataclass(**_FROZEN_DATACLASS_KWARGS)
class QueryResultRenderPayload:
    """Rendered query payload used by Geo-SAM canvas overlays."""

    geojson_features: list[dict[str, Any]]
    canvas_geometries: list[QgsGeometry]


class _ModelSessionRegistry:
    """Registry for loaded GeoSAM engines so memory can be released explicitly."""

    def __init__(self) -> None:
        self._online_engines: dict[str, Any] = {}
        self._feature_engines: dict[str, Any] = {}

    @staticmethod
    def _key(*parts: str) -> str:
        return "||".join(parts)

    def get_online_engine(self, *, source_path: str, model_id: str) -> Any:
        """Return a cached online engine or create a new one."""
        configure_geosam_qgis_runtime()
        from geosam import RasterDataset
        from geosam.engines import OnlineQueryEngine

        key = self._key(model_id, str(Path(source_path).expanduser()), "online")
        if key not in self._online_engines:
            dataset = RasterDataset(source_path)
            self._online_engines[key] = OnlineQueryEngine(
                dataset,
                create_model_spec(model_id),
            )
        return self._online_engines[key]

    def release_online_engines(
        self,
        *,
        model_id: str | None = None,
        keep_source_path: str | None = None,
    ) -> int:
        """Release cached online engines and return the number removed.

        Parameters
        ----------
        model_id : str | None, optional
            Restrict removal to one model id. When ``None``, all model ids are
            considered.
        keep_source_path : str | None, optional
            Preserve the engine for this source path if present.

        Returns
        -------
        int
            Number of released online engines.
        """
        removed_count = 0
        normalized_keep_source_path = None
        if keep_source_path is not None:
            normalized_keep_source_path = str(Path(keep_source_path).expanduser())

        for key in list(self._online_engines):
            key_model_id, key_source_path, _ = key.split("||", maxsplit=2)
            if model_id is not None and key_model_id != model_id:
                continue
            if (
                normalized_keep_source_path is not None
                and key_source_path == normalized_keep_source_path
            ):
                continue
            _close_runtime_engine(self._online_engines[key])
            del self._online_engines[key]
            removed_count += 1
        return removed_count

    def get_feature_engine(self, *, feature_dir: str | Path, model_id: str) -> Any:
        """Return a cached feature-query engine or create a new one."""
        configure_geosam_qgis_runtime()
        from geosam.engines import FeatureQueryEngine

        manifest_path = resolve_feature_manifest_path(feature_dir)
        key = self._key(model_id, str(manifest_path), "feature")
        if key not in self._feature_engines:
            self._feature_engines[key] = FeatureQueryEngine(
                manifest_path,
                create_model_spec(model_id),
            )
        return self._feature_engines[key]

    def release(self, *, model_id: str | None = None) -> int:
        """Release cached engines and return the number removed."""
        removed_count = 0
        for store_name in ("_online_engines", "_feature_engines"):
            store = getattr(self, store_name)
            for key in list(store):
                if model_id is not None and not key.startswith(f"{model_id}||"):
                    continue
                _close_runtime_engine(store[key])
                del store[key]
                removed_count += 1
        return removed_count


MODEL_SESSIONS = _ModelSessionRegistry()


def _normalize_performance_mode(value: Any) -> PerformanceMode:
    """Normalize a configured performance mode."""
    normalized_value = str(value or "balanced").strip().lower()
    if normalized_value in {"balanced", "fastest", "low_memory"}:
        return normalized_value  # type: ignore[return-value]
    return "balanced"


def get_performance_mode() -> PerformanceMode:
    """Return the configured realtime performance mode."""
    return _normalize_performance_mode(
        load_plugin_settings().get("performance_mode", "balanced")
    )


def _normalize_preview_render_mode(value: Any) -> PreviewRenderMode:
    """Normalize a configured preview render mode."""
    normalized_value = str(value or "pixel_level").strip().lower().replace("-", "_")
    if normalized_value in {"pixel_level", "simplified"}:
        return normalized_value  # type: ignore[return-value]
    return "pixel_level"


def get_preview_render_mode() -> PreviewRenderMode:
    """Return the configured preview render mode."""
    return _normalize_preview_render_mode(
        load_plugin_settings().get("preview_render_mode", "pixel_level")
    )


def _close_runtime_engine(engine: Any) -> None:
    """Best-effort close for cached GeoSAM engine instances."""
    close_method = getattr(engine, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception as exc:
            logger.warning("Failed to close cached GeoSAM engine: %s", exc)


def sanitize_path_component(value: str) -> str:
    """Convert an arbitrary label into a filesystem-safe path component."""
    sanitized = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value.strip()
    )
    sanitized = sanitized.strip("._")
    return sanitized or "unnamed"


def get_layer_cache_directory(layer_name: str) -> Path:
    """Return the cache root for a raster layer name."""
    layer_cache_dir = get_cache_directory() / sanitize_path_component(layer_name)
    layer_cache_dir.mkdir(parents=True, exist_ok=True)
    return layer_cache_dir


def create_model_spec_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_id: str | None = None,
    device: str | None = None,
) -> ModelSpec:
    """Create a model spec from an arbitrary checkpoint path."""
    configure_geosam_qgis_runtime()
    from geosam.runtime import create_model_spec_from_checkpoint as create_runtime_spec

    return create_runtime_spec(
        checkpoint_path,
        model_id=model_id,
        device=device,
    )


def _flush_torch_memory() -> None:
    """Best-effort cleanup for Python and accelerator memory."""
    gc.collect()
    try:
        import torch
    except (ImportError, OSError) as exc:
        logger.warning(
            "Skipping PyTorch memory cleanup because torch failed to load: %s",
            exc,
        )
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def _clear_online_engine_predictor_state(
    engine: Any,
    *,
    keep_features: bool,
) -> None:
    """Clear transient predictor state retained by an online engine.

    Parameters
    ----------
    engine : Any
        Online query engine that may hold an Ultralytics SAM predictor.
    keep_features : bool
        True to preserve ``predictor.features`` as the hot cache for the
        current active image. False to clear all image-specific predictor
        state, including cached features.

    Notes
    -----
    This only clears image-specific predictor state. Model weights remain
    loaded so cached prompt inference can continue using the same engine
    instance.

    """
    predictor = getattr(getattr(engine, "adapter", None), "model", None)
    predictor = getattr(predictor, "predictor", None)
    if predictor is None:
        return

    attributes_to_clear: list[tuple[str, Any]] = [
        ("im", None),
        ("dataset", None),
        ("source_type", None),
        ("batch", None),
        ("results", None),
        ("txt_path", None),
        ("plotted_img", None),
        ("prompts", {}),
        ("vid_writer", {}),
    ]
    if not keep_features:
        attributes_to_clear.insert(0, ("features", None))

    for attribute_name, empty_value in attributes_to_clear:
        if hasattr(predictor, attribute_name):
            setattr(predictor, attribute_name, empty_value)


def _should_trim_online_engines() -> bool:
    """Return whether inactive online engines should be released immediately."""
    return get_performance_mode() != "fastest"


def _should_clear_online_query_state_after_result() -> bool:
    """Return whether realtime queries should clear predictor state after use."""
    return get_performance_mode() == "low_memory"


def release_online_runtime_hot_cache(
    *,
    keep_source_path: str | None = None,
    force: bool = False,
) -> int:
    """Release cached online engines and clear accelerator memory.

    Parameters
    ----------
    keep_source_path : str | None, optional
        Preserve the online engine for this source path when provided.
    force : bool, default=False
        When ``True``, release engines even when the configured performance
        mode prefers keeping them warm.

    Returns
    -------
    int
        Number of released online engines.

    """
    if not force and not _should_trim_online_engines():
        return 0
    removed_count = MODEL_SESSIONS.release_online_engines(
        keep_source_path=keep_source_path,
    )
    if removed_count > 0:
        _flush_torch_memory()
    return removed_count


def release_runtime_models(*, model_id: str | None = None) -> int:
    """Release loaded GeoSAM engines and clear accelerator caches."""
    removed_count = MODEL_SESSIONS.release(model_id=model_id)
    _flush_torch_memory()
    return removed_count


def cleanup_on_plugin_unload() -> dict[str, int]:
    """Release model memory and clear cache when configured."""
    removed_models = release_runtime_models()
    removed_cache_files = 0
    if load_plugin_settings().get("clear_cache_on_plugin_close", True):
        removed_cache_files = clear_cache()
    return {
        "released_models": removed_models,
        "removed_cache_files": removed_cache_files,
    }


def resolve_feature_manifest_path(feature_dir: str | Path) -> Path:
    """Resolve the manifest file path inside a feature folder."""
    configure_geosam_qgis_runtime()
    from geosam.runtime import resolve_feature_manifest_path as resolve_runtime_manifest

    return resolve_runtime_manifest(feature_dir)


def describe_feature_source(feature_dir: str | Path) -> FeatureSourceSummary:
    """Load summary metadata for a GeoSAM feature folder."""
    configure_geosam_qgis_runtime()
    import geopandas as gpd
    import pandas as pd

    manifest_path = resolve_feature_manifest_path(feature_dir)
    if manifest_path.suffix == ".pkl":
        frame = pd.read_pickle(manifest_path)
        frame = frame.set_crs(frame.crs or "EPSG:4326", allow_override=True)
    else:
        frame = gpd.read_parquet(manifest_path)

    if len(frame) == 0:
        msg = f"Feature manifest is empty: {manifest_path}"
        logger.error(msg)
        raise ValueError(msg)

    crs_text = _resolve_feature_crs_text(frame)
    total_bounds = tuple(float(value) for value in frame.total_bounds)
    transform_values = json.loads(frame.iloc[0]["transform"])
    pixel_area = abs(float(transform_values[0]) * float(transform_values[4]))
    checkpoint_path = str(frame.iloc[0].get("checkpoint_path", "")).strip() or None
    model_id = (
        infer_model_id_from_checkpoint_path(checkpoint_path, allow_unknown=True)
        if checkpoint_path is not None
        else None
    )
    if len(total_bounds) != 4:
        msg = f"Unexpected feature extent size for manifest {manifest_path}: {total_bounds!r}"
        logger.error(msg)
        raise ValueError(msg)
    return FeatureSourceSummary(
        manifest_path=manifest_path,
        crs_text=crs_text,
        extent=(
            total_bounds[0],
            total_bounds[1],
            total_bounds[2],
            total_bounds[3],
        ),
        chip_count=len(frame),
        pixel_area=pixel_area,
        model_id=model_id,
        checkpoint_path=checkpoint_path,
    )


def _resolve_feature_crs_text(frame: Any) -> str:
    """Resolve a stable CRS string from a feature manifest frame."""
    if "crs" in frame.columns and len(frame) > 0:
        manifest_crs = frame.iloc[0]["crs"]
        if isinstance(manifest_crs, str) and manifest_crs.strip():
            return manifest_crs.strip()

    frame_crs = frame.crs
    if frame_crs is None:
        msg = "Feature manifest CRS is missing."
        logger.error(msg)
        raise ValueError(msg)

    to_authority = getattr(frame_crs, "to_authority", None)
    if callable(to_authority):
        authority = to_authority()
        if authority is not None:
            return f"{authority[0]}:{authority[1]}"

    to_epsg = getattr(frame_crs, "to_epsg", None)
    if callable(to_epsg):
        epsg_code = to_epsg()
        if epsg_code is not None:
            return f"EPSG:{epsg_code}"

    to_wkt = getattr(frame_crs, "to_wkt", None)
    if callable(to_wkt):
        return str(to_wkt())

    return str(frame_crs)


def layer_extent_rectangle(layer: QgsRasterLayer) -> QgsRectangle:
    """Return the extent of a raster layer."""
    return layer.extent()


def layer_pixel_area(layer: QgsRasterLayer) -> float:
    """Return the approximate area represented by one raster pixel."""
    return abs(
        float(layer.rasterUnitsPerPixelX()) * float(layer.rasterUnitsPerPixelY())
    )


def chip_extent_rectangles_for_source(
    source_path: str | Path,
    *,
    bands: list[int] | None = None,
    crs: str | None = None,
    res: float | None = None,
    extent: tuple[float, float, float, float] | None = None,
    extent_crs: str | None = None,
    chip_size: int = 1024,
    stride: int = 512,
) -> list[tuple[float, float, float, float]]:
    """Return chip extents for a raster source using GeoSAM sampling rules."""
    configure_geosam_qgis_runtime()
    from geosam.runtime import chip_extent_rectangles_for_source as runtime_chip_extents

    return runtime_chip_extents(
        source_path,
        bands=bands,
        crs=crs,
        res=res,
        extent=extent,
        extent_crs=extent_crs,
        chip_size=chip_size,
        stride=stride,
    )


def _raster_layer_source_candidates(layer: QgsRasterLayer) -> list[str]:
    """Return possible raster source strings for a QGIS raster layer."""
    candidates: list[str] = []
    for attribute_name in ("publicSource", "source"):
        attribute = getattr(layer, attribute_name, None)
        if callable(attribute):
            try:
                value = str(attribute())
            except Exception:
                continue
            if value and value not in candidates:
                candidates.append(value)

    provider = layer.dataProvider()
    if provider is not None and hasattr(provider, "dataSourceUri"):
        try:
            value = str(provider.dataSourceUri())
        except Exception:
            value = ""
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def _normalize_local_raster_source(source_candidate: str) -> str | None:
    """Resolve a QGIS raster source candidate to a local file path.

    Parameters
    ----------
    source_candidate : str
        Raster source string reported by QGIS.

    Returns
    -------
    str | None
        Absolute local file path when one can be resolved, otherwise ``None``.
    """
    normalized_source = source_candidate.strip()
    if not normalized_source:
        return None

    direct_path_candidate = normalized_source.split("|", maxsplit=1)[0].strip()
    if direct_path_candidate:
        direct_path = Path(direct_path_candidate).expanduser()
        if direct_path.exists() and direct_path.is_file():
            return str(direct_path)

    parsed_source = urlparse(normalized_source)
    if parsed_source.scheme not in {"", "file"}:
        return None

    if parsed_source.scheme == "file":
        candidate_path = Path(parsed_source.path).expanduser()
    else:
        candidate_path = Path(normalized_source).expanduser()
    if candidate_path.exists() and candidate_path.is_file():
        return str(candidate_path)
    return None


def _layer_source_fingerprint(layer: QgsRasterLayer) -> str:
    """Return a stable fingerprint for matching raster cache entries."""
    provider = layer.dataProvider()
    provider_name = layer.providerType()
    layer_source = "|".join(_raster_layer_source_candidates(layer))
    provider_uri = ""
    if provider is not None and hasattr(provider, "dataSourceUri"):
        try:
            provider_uri = str(provider.dataSourceUri())
        except Exception:
            provider_uri = ""
    return "|".join([layer.name(), provider_name, layer_source, provider_uri])


def _query_to_crs_text(query, crs_text: str) -> Any:
    """Normalize a GeoSAM query into a target CRS string.

    Parameters
    ----------
    query : Any
        GeoSAM query object.
    crs_text : str
        Destination CRS string.

    Returns
    -------
    Any
        Query expressed in ``crs_text`` when conversion is required.
    """
    configure_geosam_qgis_runtime()
    query_crs = getattr(query, "crs", None)
    if query_crs is None or not crs_text:
        return query
    return query if str(query_crs) == crs_text else query.to_crs(crs_text)


def _rectangle_to_bbox(rectangle: QgsRectangle, *, crs_text: str):
    """Convert a QGIS rectangle into a GeoSAM bounding box."""
    configure_geosam_qgis_runtime()
    from geosam import BoundingBox

    return BoundingBox(
        float(rectangle.xMinimum()),
        float(rectangle.yMinimum()),
        float(rectangle.xMaximum()),
        float(rectangle.yMaximum()),
        crs=crs_text,
    )


def _query_in_layer_crs(layer: QgsRasterLayer, query):
    """Normalize a query into the layer CRS."""
    configure_geosam_qgis_runtime()
    crs_text = layer.crs().authid() or layer.crs().toWkt()
    if query.crs == QgsCoordinateReferenceSystem(crs_text):
        return query
    if query.crs is None:
        return query
    return query if str(query.crs) == crs_text else query.to_crs(crs_text)


def _query_is_far_from_chip_edge(query, *, chip_bounds, chip_grid) -> bool:
    """Return whether a query center is inside the safe inner chip region."""
    configure_geosam_qgis_runtime()
    from geosam.query import query_bounds, query_center

    if not chip_bounds.contains(query_bounds(query)):
        return False

    center_x, center_y = query_center(query)
    height, width = chip_grid.shape
    pixel_size_x = abs(float(chip_grid.transform.a))
    pixel_size_y = abs(float(chip_grid.transform.e))
    margin_x_pixels = min(
        max(ONLINE_QUERY_REFRESH_MARGIN_PIXELS, 1), max(width // 2, 1)
    )
    margin_y_pixels = min(
        max(ONLINE_QUERY_REFRESH_MARGIN_PIXELS, 1), max(height // 2, 1)
    )
    margin_x = margin_x_pixels * pixel_size_x
    margin_y = margin_y_pixels * pixel_size_y

    return (
        chip_bounds.left + margin_x <= center_x <= chip_bounds.right - margin_x
        and chip_bounds.bottom + margin_y <= center_y <= chip_bounds.top - margin_y
    )


def _prepare_realtime_raster_source_sample(
    *,
    source_path: str,
    query,
    model_id: str,
) -> PreparedRealtimeRasterSample:
    """Read a realtime raster chip on the main thread for background encoding."""
    configure_geosam_qgis_runtime()
    from geosam.datasets import RasterDataset
    from geosam.engines import _prompt_prediction_kwargs
    from geosam.query import query_bounds, query_center, window_from_center

    model_spec = create_model_spec(model_id)
    dataset = RasterDataset(source_path)
    projected_query = (
        query if str(query.crs) == str(dataset.crs) else query.to_crs(dataset.crs)
    )
    chip_bounds = window_from_center(
        query_center(projected_query),
        model_spec.resolved_imgsz,
        grid=dataset.grid,
    )
    sample = dataset[chip_bounds]
    chip_grid = sample.grid
    return PreparedRealtimeRasterSample(
        source_path=sample.source_path,
        model_image=sample.to_model_image(),
        chip_bounds=chip_bounds,
        chip_grid=chip_grid,
        query=projected_query,
        query_bounds_value=query_bounds(projected_query),
        prompt_kwargs=_prompt_prediction_kwargs(projected_query, chip_grid),
        crs_text=str(chip_grid.crs),
    )


def _prediction_to_prepared_query_result(
    prediction: Any,
    *,
    chip_grid: Any,
    chip_bounds: Any,
    query_bounds_value: Any,
    source_path: str,
    model_type: str,
) -> Any:
    """Convert a prediction using bounds prepared outside the worker thread."""
    configure_geosam_qgis_runtime()
    import numpy as np
    from geosam.engines import QueryResult

    if prediction.masks is None:
        mask_array = np.zeros((0, chip_grid.height, chip_grid.width), dtype=bool)
    else:
        mask_array = prediction.masks.detach().cpu().numpy().astype(bool)
    scores = prediction.scores.detach().cpu().numpy()
    return QueryResult(
        mask_array=mask_array,
        mask_transform=chip_grid.transform,
        mask_crs=chip_grid.crs,
        query_bounds=query_bounds_value,
        chip_bounds=chip_bounds,
        scores=scores,
        source_path=source_path,
        chip_id=None,
        model_type=model_type,
    )


def _realtime_query_cache_directory(layer: QgsRasterLayer, model_id: str) -> Path:
    """Return the persistent realtime-query cache directory for a layer/model."""
    cache_dir = get_layer_cache_directory(layer.name()) / "realtime_query" / model_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _online_layer_raster_cache_directory(layer: QgsRasterLayer) -> Path:
    """Return the raster-export cache directory for a layer."""
    cache_dir = get_layer_cache_directory(layer.name()) / "online_raster"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _persistent_cache_entry_paths(
    layer: QgsRasterLayer,
    model_id: str,
    *,
    chip_bounds,
) -> tuple[Path, Path]:
    """Return persistent cache paths for one encoded realtime chip."""
    cache_dir = _realtime_query_cache_directory(layer, model_id)
    return _persistent_cache_entry_paths_for_directory(
        cache_dir,
        chip_bounds=chip_bounds,
    )


def _persistent_cache_entry_paths_for_directory(
    cache_dir: Path,
    *,
    chip_bounds,
) -> tuple[Path, Path]:
    """Return persistent cache paths for one encoded realtime chip."""
    chip_key = sanitize_path_component(
        "_".join([
            f"{chip_bounds.left:.3f}",
            f"{chip_bounds.bottom:.3f}",
            f"{chip_bounds.right:.3f}",
            f"{chip_bounds.top:.3f}",
        ])
    )
    entry_dir = cache_dir / chip_key
    entry_dir.mkdir(parents=True, exist_ok=True)
    return entry_dir / "encoded.pt", entry_dir / "metadata.json"


def _query_cache_shape_matches_model(model_id: str, query_cache: Any) -> bool:
    """Return whether an in-memory query cache matches the model image size.

    Parameters
    ----------
    model_id : str
        Registered model identifier.
    query_cache : Any
        GeoSAM online query cache payload.

    Returns
    -------
    bool
        ``True`` when the cache chip grid shape matches the effective model
        image size.

    """
    chip_grid = getattr(query_cache, "chip_grid", None)
    if chip_grid is None:
        return False
    try:
        cache_shape = tuple(int(value) for value in chip_grid.shape)
    except Exception:
        return False
    return cache_shape == effective_model_imgsz(model_id)


def _save_persistent_query_cache(
    *,
    layer: QgsRasterLayer,
    model_id: str,
    source_path: str,
    query_cache: Any,
) -> None:
    """Persist a realtime query cache entry to disk."""
    _save_persistent_query_cache_entry(
        cache_directory=_realtime_query_cache_directory(layer, model_id),
        layer_name=layer.name(),
        source_fingerprint=_layer_source_fingerprint(layer),
        model_id=model_id,
        source_path=source_path,
        query_cache=query_cache,
    )


def save_prepared_realtime_query_cache(
    *,
    prepared_query: PreparedRealtimeRasterQuery,
    source_path: str,
    query_cache: Any,
) -> None:
    """Persist a prepared realtime query cache entry to disk.

    Parameters
    ----------
    prepared_query : PreparedRealtimeRasterQuery
        Prepared realtime query metadata captured on the main thread.
    source_path : str
        Raster source used by the finished realtime query.
    query_cache : Any
        Reusable encoded-query cache returned by the background realtime query.

    """
    _save_persistent_query_cache_entry(
        cache_directory=prepared_query.cache_directory,
        layer_name=prepared_query.layer_name,
        source_fingerprint=prepared_query.source_fingerprint,
        model_id=prepared_query.model_id,
        source_path=source_path,
        query_cache=query_cache,
    )


def _save_persistent_query_cache_entry(
    *,
    cache_directory: Path,
    layer_name: str,
    source_fingerprint: str,
    model_id: str,
    source_path: str,
    query_cache: Any,
) -> None:
    """Persist a realtime query cache entry using precomputed cache metadata."""
    settings = load_plugin_settings()
    if not settings.get("cache_enabled", True):
        return
    if (
        query_cache is None
        or query_cache.encoded is None
        or query_cache.chip_grid is None
    ):
        return
    if query_cache.chip_bounds is None:
        return

    crs_text = getattr(query_cache, "crs_text", None)
    if crs_text is None:
        logger.warning(
            "Skipping realtime query cache save because chip CRS text was not "
            "prepared on the main thread."
        )
        return

    encoded_path, metadata_path = _persistent_cache_entry_paths_for_directory(
        cache_directory,
        chip_bounds=query_cache.chip_bounds,
    )
    query_cache.encoded.save(encoded_path)
    metadata = {
        "model_id": model_id,
        "layer_name": layer_name,
        "source_fingerprint": source_fingerprint,
        "source_path": source_path,
        "chip_bounds": [
            float(query_cache.chip_bounds.left),
            float(query_cache.chip_bounds.bottom),
            float(query_cache.chip_bounds.right),
            float(query_cache.chip_bounds.top),
        ],
        "transform": list(query_cache.chip_grid.transform)[:6],
        "shape": list(query_cache.chip_grid.shape),
        "crs": str(crs_text),
        "feature_path": str(encoded_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    cleanup_cache()


def prepare_realtime_raster_query(
    layer: QgsRasterLayer,
    model_id: str,
    query,
    *,
    cache: RealtimeQueryCache | None = None,
) -> PreparedRealtimeRasterQuery | None:
    """Prepare a background query plan when realtime encoding is required.

    Parameters
    ----------
    layer : QgsRasterLayer
        Active realtime raster layer.
    model_id : str
        Selected GeoSAM model identifier.
    query : Any
        GeoSAM query object.
    cache : RealtimeQueryCache | None, optional
        Realtime in-memory cache for the current session.

    Returns
    -------
    PreparedRealtimeRasterQuery | None
        Returns ``None`` when the query can run synchronously using an existing
        in-memory or persistent cache. Returns a prepared plan when a new chip
        must be encoded in a background task.

    """
    configure_geosam_qgis_runtime()
    model_definition = get_model_definition(model_id)
    supports_feature_reuse = bool(model_definition.supports_feature_reuse)
    layer_crs_text = layer.crs().authid() or layer.crs().toWkt()
    source_fingerprint = _layer_source_fingerprint(layer)

    if cache is not None:
        cache_mismatch = cache.layer_id != layer.id() or cache.model_id != model_id
        if cache_mismatch:
            cache.clear()
        elif supports_feature_reuse and cache.query_cache is not None:
            cached_source_path = (
                Path(cache.source_candidate).expanduser()
                if cache.source_candidate is not None
                else None
            )
            if not _query_cache_shape_matches_model(model_id, cache.query_cache):
                cache.query_cache = None
                cache.engine = None
            elif cached_source_path is None or not cached_source_path.exists():
                cache.query_cache = None
                cache.engine = None
            elif not _query_is_far_from_chip_edge(
                _query_in_layer_crs(layer, query),
                chip_bounds=cache.query_cache.chip_bounds,
                chip_grid=cache.query_cache.chip_grid,
            ):
                cache.query_cache = None
                cache.engine = None

    if (
        supports_feature_reuse
        and cache is not None
        and cache.query_cache is not None
        and cache.source_candidate is not None
    ):
        cached_source_path = Path(cache.source_candidate).expanduser()
        if cached_source_path.exists():
            return None

    persistent_cache_hit: PreparedPersistentQueryCacheHit | None = None
    if supports_feature_reuse and (cache is None or cache.query_cache is None):
        persistent_cache_hit = _find_persistent_query_cache(
            layer,
            model_id,
            query,
        )

    source_candidates = [
        source_path
        for source_candidate in _raster_layer_source_candidates(layer)
        if (source_path := _normalize_local_raster_source(source_candidate)) is not None
    ]
    prepared_source_samples: list[PreparedRealtimeRasterSample] = []
    if persistent_cache_hit is None:
        for source_candidate in source_candidates:
            try:
                prepared_source_samples.append(
                    _prepare_realtime_raster_source_sample(
                        source_path=source_candidate,
                        query=query,
                        model_id=model_id,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Failed to prepare realtime raster chip for %s: %s",
                    source_candidate,
                    exc,
                )

    online_export_plan: OnlineTileExportPlan | None = None
    online_raster_cache_directory: Path | None = None
    if persistent_cache_hit is None and len(source_candidates) == 0:
        online_export_plan = prepare_online_raster_export_plan(
            layer,
            query,
            model_id=model_id,
            chip_size=create_model_spec(model_id).resolved_imgsz,
            source_fingerprint=source_fingerprint,
        )
        online_raster_cache_directory = _online_layer_raster_cache_directory(layer)

    return PreparedRealtimeRasterQuery(
        model_id=model_id,
        layer_id=layer.id(),
        layer_name=layer.name(),
        source_fingerprint=source_fingerprint,
        layer_crs_text=layer_crs_text,
        cache_directory=_realtime_query_cache_directory(layer, model_id),
        online_raster_cache_directory=online_raster_cache_directory,
        source_candidates=tuple(source_candidates),
        prepared_source_samples=tuple(prepared_source_samples),
        online_export_plan=online_export_plan,
        supports_feature_reuse=supports_feature_reuse,
        persistent_cache_hit=persistent_cache_hit,
    )


def run_prepared_realtime_raster_query(
    prepared_query: PreparedRealtimeRasterQuery,
    query,
    *,
    progress_callback: RealtimeQueryProgressCallback | None = None,
    is_canceled: RealtimeQueryCancelCallback | None = None,
    qgis_task: Any | None = None,
) -> PreparedRealtimeRasterQueryResult:
    """Execute a prepared realtime raster query without using QGIS layer APIs.

    Parameters
    ----------
    prepared_query : PreparedRealtimeRasterQuery
        Prepared background-query inputs captured on the main thread.
    query : Any
        GeoSAM query object.
    progress_callback : RealtimeQueryProgressCallback | None, optional
        Callback receiving ``(stage_text, progress_percent)`` updates.
    is_canceled : RealtimeQueryCancelCallback | None, optional
        Callback that returns ``True`` when the background task should stop.
    qgis_task : Any | None, optional
        QGIS task object used by GeoSAM for progress, cancellation, logging,
        and QGIS CRS operations inside the worker.

    Returns
    -------
    PreparedRealtimeRasterQueryResult
        Query result and optional in-memory cache payload.

    Raises
    ------
    RuntimeError
        If cancellation is requested before finishing the job.
    ValueError
        If all raster source candidates fail.

    """
    configure_geosam_qgis_runtime(task=qgis_task)
    from geosam.engines import (
        OnlineQueryCache,
        _require_query_crs,
    )
    from geosam.models import build_model_adapter

    def _report_progress(stage_text: str, progress_value: float) -> None:
        if progress_callback is not None:
            progress_callback(stage_text, progress_value)

    def _ensure_not_canceled() -> None:
        if is_canceled is not None and is_canceled():
            msg = "Realtime raster query task was canceled."
            logger.info(msg)
            raise _RealtimeQueryCanceledError(msg)

    _require_query_crs(query)
    errors: list[str] = []
    model_spec = create_model_spec(prepared_query.model_id)

    if prepared_query.supports_feature_reuse:
        _ensure_not_canceled()
        _report_progress("Checking cached encoding", 2.0)
        prepared_cache_hit = prepared_query.persistent_cache_hit
        persistent_cache_hit = _load_prepared_persistent_query_cache(prepared_cache_hit)
        if persistent_cache_hit is not None:
            source_path, query_cache = persistent_cache_hit
            engine = None
            try:
                _ensure_not_canceled()
                _report_progress("Loading segmentation model", 15.0)
                engine = build_model_adapter(model_spec)
                _ensure_not_canceled()
                _report_progress("Running prompt query", 75.0)
                prediction = engine.predict_features(
                    query_cache.encoded,
                    multimask_output=False,
                    **prepared_cache_hit.prompt_kwargs,
                )
                result = _prediction_to_prepared_query_result(
                    prediction,
                    chip_grid=query_cache.chip_grid,
                    chip_bounds=query_cache.chip_bounds,
                    query_bounds_value=prepared_cache_hit.query_bounds_value,
                    source_path=query_cache.source_path,
                    model_type=model_spec.model_type,
                )
                _ensure_not_canceled()
                _report_progress("Finished realtime query", 100.0)
                return PreparedRealtimeRasterQueryResult(
                    source_path=source_path,
                    result=result,
                    query_cache=query_cache,
                )
            except _RealtimeQueryCanceledError:
                raise
            except Exception as exc:
                errors.append(f"persistent-cache {source_path}: {exc}")
            finally:
                _close_runtime_engine(engine)

    prepared_source_samples = list(prepared_query.prepared_source_samples)
    source_candidates = list(prepared_query.source_candidates)
    if (
        len(prepared_source_samples) == 0
        and prepared_query.online_export_plan is not None
        and prepared_query.online_raster_cache_directory is not None
    ):
        try:
            _ensure_not_canceled()
            _report_progress("Preparing online raster export", 3.0)
            exported_source = export_online_tile_raster_plan(
                prepared_query.online_export_plan,
                cache_directory=prepared_query.online_raster_cache_directory,
                layer_name=prepared_query.layer_name,
                progress_callback=_report_progress,
                is_canceled=is_canceled,
            )
            source_candidates.append(exported_source)
            _ensure_not_canceled()
            _report_progress("Preparing exported raster chip", 25.0)
            prepared_source_samples.append(
                _prepare_realtime_raster_source_sample(
                    source_path=exported_source,
                    query=query,
                    model_id=prepared_query.model_id,
                )
            )
        except _RealtimeQueryCanceledError:
            raise
        except (OnlineRasterPreflightError, OnlineRasterExportError) as exc:
            msg = describe_online_export_failure(exc)
            logger.warning("%s", msg)
            raise ValueError(msg) from exc
        except Exception as exc:
            msg = "Failed to prepare exported online raster for realtime query."
            logger.error("%s Details: %s", msg, exc)
            raise ValueError(msg) from exc

    if len(prepared_source_samples) == 0:
        msg = "Failed to prepare a local raster chip for the realtime query."
        logger.error(msg)
        raise ValueError(msg)

    for prepared_sample in prepared_source_samples:
        engine = None
        try:
            _ensure_not_canceled()
            _report_progress("Encoding image", 30.0)
            engine = build_model_adapter(model_spec)
            query_cache = None
            if prepared_query.supports_feature_reuse:
                encoded = engine.encode_image(prepared_sample.model_image)
                _ensure_not_canceled()
                _report_progress("Running prompt query", 75.0)
                prediction = engine.predict_features(
                    encoded,
                    multimask_output=False,
                    **prepared_sample.prompt_kwargs,
                )
                query_cache = OnlineQueryCache(
                    source_path=prepared_sample.source_path,
                    chip_bounds=prepared_sample.chip_bounds,
                    chip_grid=prepared_sample.chip_grid,
                    encoded=encoded,
                )
                query_cache.crs_text = prepared_sample.crs_text
            else:
                _ensure_not_canceled()
                _report_progress("Running prompt query", 75.0)
                prediction = engine.predict_image(
                    prepared_sample.model_image,
                    multimask_output=False,
                    **prepared_sample.prompt_kwargs,
                )

            result = _prediction_to_prepared_query_result(
                prediction,
                chip_grid=prepared_sample.chip_grid,
                chip_bounds=prepared_sample.chip_bounds,
                query_bounds_value=prepared_sample.query_bounds_value,
                source_path=prepared_sample.source_path,
                model_type=model_spec.model_type,
            )
            _ensure_not_canceled()
            _report_progress("Finished realtime query", 100.0)
            return PreparedRealtimeRasterQueryResult(
                source_path=prepared_sample.source_path,
                result=result,
                query_cache=query_cache,
            )
        except _RealtimeQueryCanceledError:
            raise
        except Exception as exc:
            errors.append(f"prepared-sample {prepared_sample.source_path}: {exc}")
            continue
        finally:
            _close_runtime_engine(engine)

    if len(source_candidates) > 0:
        msg = "Failed to encode a prepared realtime raster chip for the current prompt."
        logger.error("%s Details: %s", msg, errors)
        raise ValueError("\n".join([msg, *errors]))

    msg = "Failed to encode a realtime raster chip for the current prompt."
    logger.error("%s Details: %s", msg, errors)
    raise ValueError("\n".join([msg, *errors]))


def _load_persistent_query_cache_for_request(
    *,
    model_id: str,
    cache_directory: Path,
    source_fingerprint: str,
    layer_crs_text: str,
    query,
) -> tuple[str, Any] | None:
    """Load a reusable realtime-query cache entry for a prepared request."""
    cache_hit = _find_persistent_query_cache_for_request(
        model_id=model_id,
        cache_directory=cache_directory,
        source_fingerprint=source_fingerprint,
        layer_crs_text=layer_crs_text,
        query=query,
    )
    return _load_prepared_persistent_query_cache(cache_hit)


def _find_persistent_query_cache_for_request(
    *,
    model_id: str,
    cache_directory: Path,
    source_fingerprint: str,
    layer_crs_text: str,
    query,
) -> PreparedPersistentQueryCacheHit | None:
    """Find reusable realtime-query cache metadata without loading features."""
    configure_geosam_qgis_runtime()
    settings = load_plugin_settings()
    if not settings.get("cache_enabled", True):
        return None

    initialize_rasterio_proj_data()
    from geosam.datasets import GeoGrid
    from geosam.engines import _prompt_prediction_kwargs
    from geosam.query import BoundingBox, query_bounds, query_center
    from rasterio import Affine

    if not cache_directory.exists():
        return None

    projected_query = _query_to_crs_text(query, layer_crs_text)
    center_x, center_y = query_center(projected_query)
    best_match: tuple[float, PreparedPersistentQueryCacheHit] | None = None
    expected_shape = effective_model_imgsz(model_id)
    for metadata_path in sorted(cache_directory.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "Failed to read query cache metadata %s: %s", metadata_path, exc
            )
            continue

        metadata_model_id = str(metadata.get("model_id", "")).strip()
        if metadata_model_id and metadata_model_id != model_id:
            continue

        if metadata.get("source_fingerprint") != source_fingerprint:
            continue

        try:
            metadata_shape = tuple(int(value) for value in metadata["shape"])
        except Exception as exc:
            logger.warning(
                "Skipping realtime query cache with invalid shape metadata %s: %s",
                metadata_path,
                exc,
            )
            continue
        if metadata_shape != expected_shape:
            logger.info(
                "Skipping stale realtime query cache %s with shape %s; expected %s.",
                metadata_path,
                metadata_shape,
                expected_shape,
            )
            continue

        source_path = str(metadata.get("source_path", "")).strip()
        if not source_path:
            continue

        if Path(source_path).suffix and not Path(source_path).exists():
            continue

        chip_bounds = BoundingBox(
            *metadata["chip_bounds"],
            crs=metadata["crs"],
        )
        chip_grid = GeoGrid(
            Affine(*metadata["transform"]),
            metadata_shape,
            metadata["crs"],
        )
        cache_query = (
            projected_query
            if str(projected_query.crs) == str(chip_grid.crs)
            else projected_query.to_crs(chip_grid.crs)
        )
        if not _query_is_far_from_chip_edge(
            cache_query,
            chip_bounds=chip_bounds,
            chip_grid=chip_grid,
        ):
            continue

        encoded_path = Path(metadata["feature_path"])
        if not encoded_path.exists():
            continue

        distance = (chip_bounds.center[0] - center_x) ** 2 + (
            chip_bounds.center[1] - center_y
        ) ** 2
        cache_hit = PreparedPersistentQueryCacheHit(
            source_path=source_path,
            feature_path=encoded_path,
            chip_bounds=chip_bounds,
            chip_grid=chip_grid,
            query=cache_query,
            query_bounds_value=query_bounds(cache_query),
            prompt_kwargs=_prompt_prediction_kwargs(cache_query, chip_grid),
            crs_text=str(metadata["crs"]),
        )
        best_match = (
            (distance, cache_hit)
            if best_match is None
            else min(
                best_match,
                (distance, cache_hit),
                key=lambda item: item[0],
            )
        )

    if best_match is None:
        return None
    return best_match[1]


def _load_prepared_persistent_query_cache(
    cache_hit: PreparedPersistentQueryCacheHit | None,
) -> tuple[str, Any] | None:
    """Load encoded features for a prepared persistent query cache hit."""
    configure_geosam_qgis_runtime()
    if cache_hit is None:
        return None

    from geosam.engines import OnlineQueryCache
    from geosam.models import EncodedImageFeatures

    encoded = EncodedImageFeatures.load(cache_hit.feature_path, map_location="cpu")
    query_cache = OnlineQueryCache(
        source_path=cache_hit.source_path,
        chip_bounds=cache_hit.chip_bounds,
        chip_grid=cache_hit.chip_grid,
        encoded=encoded,
    )
    query_cache.crs_text = cache_hit.crs_text
    return cache_hit.source_path, query_cache


def _load_persistent_query_cache(
    layer: QgsRasterLayer,
    model_id: str,
    query,
) -> tuple[str, Any] | None:
    """Load a reusable realtime-query cache entry when one matches the query."""
    return _load_persistent_query_cache_for_request(
        model_id=model_id,
        cache_directory=_realtime_query_cache_directory(layer, model_id),
        source_fingerprint=_layer_source_fingerprint(layer),
        layer_crs_text=layer.crs().authid() or layer.crs().toWkt(),
        query=query,
    )


def _find_persistent_query_cache(
    layer: QgsRasterLayer,
    model_id: str,
    query,
) -> PreparedPersistentQueryCacheHit | None:
    """Find a reusable realtime-query cache entry without loading features."""
    return _find_persistent_query_cache_for_request(
        model_id=model_id,
        cache_directory=_realtime_query_cache_directory(layer, model_id),
        source_fingerprint=_layer_source_fingerprint(layer),
        layer_crs_text=layer.crs().authid() or layer.crs().toWkt(),
        query=query,
    )


def _export_online_raster_source(layer: QgsRasterLayer, query, *, model_id: str) -> str:
    """Export supported online tiles into the plugin cache as a GeoTIFF."""
    return export_online_tile_raster_source(
        layer,
        query,
        model_id=model_id,
        chip_size=create_model_spec(model_id).resolved_imgsz,
        source_fingerprint=_layer_source_fingerprint(layer),
        cache_directory=_online_layer_raster_cache_directory(layer),
    )


def query_feature_source(
    feature_dir: str | Path,
    model_id: str | None,
    query: BoundingBox | Points | PromptSet,
) -> QueryResult:
    """Run a query against a GeoSAM feature folder."""
    configure_geosam_qgis_runtime()
    resolved_model_id = model_id
    if resolved_model_id is None:
        resolved_model_id = describe_feature_source(feature_dir).model_id
    if resolved_model_id is None:
        msg = (
            "Could not infer the model used to create this feature cache. "
            "Please rebuild the features with a supported GeoSAM checkpoint."
        )
        logger.error(msg)
        raise ValueError(msg)

    engine = MODEL_SESSIONS.get_feature_engine(
        feature_dir=feature_dir,
        model_id=resolved_model_id,
    )
    return engine.query(query)


def query_raster_layer(
    layer: QgsRasterLayer,
    model_id: str,
    query: BoundingBox | Points | PromptSet,
    *,
    cache: RealtimeQueryCache | None = None,
) -> QueryResult:
    """Run a query against a QGIS raster layer."""
    configure_geosam_qgis_runtime()
    from geosam.engines import OnlineQueryCache

    model_definition = get_model_definition(model_id)
    supports_feature_reuse = bool(model_definition.supports_feature_reuse)

    if cache is not None:
        cache_mismatch = cache.layer_id != layer.id() or cache.model_id != model_id
        if cache_mismatch:
            cache.clear()
        elif supports_feature_reuse and cache.query_cache is not None:
            if not _query_cache_shape_matches_model(model_id, cache.query_cache):
                cache.query_cache = None
                cache.engine = None
            elif not _query_is_far_from_chip_edge(
                _query_in_layer_crs(layer, query),
                chip_bounds=cache.query_cache.chip_bounds,
                chip_grid=cache.query_cache.chip_grid,
            ):
                cache.query_cache = None

    if (
        supports_feature_reuse
        and cache is not None
        and cache.query_cache is not None
        and cache.source_candidate is not None
    ):
        cached_source_path = Path(cache.source_candidate).expanduser()
        if cached_source_path.exists():
            release_online_runtime_hot_cache(
                keep_source_path=str(cached_source_path),
            )
            engine = MODEL_SESSIONS.get_online_engine(
                source_path=str(cached_source_path),
                model_id=model_id,
            )
            cache.layer_id = layer.id()
            cache.model_id = model_id
            cache.source_candidate = str(cached_source_path)
            cache.engine = engine
            result = engine.query(query, cache=cache.query_cache)
            if supports_feature_reuse and _should_clear_online_query_state_after_result():
                _clear_online_engine_predictor_state(engine, keep_features=True)
                _flush_torch_memory()
            return result

    source_candidates = [
        source_path
        for source_candidate in _raster_layer_source_candidates(layer)
        if (source_path := _normalize_local_raster_source(source_candidate)) is not None
    ]
    persistent_cache_hit: tuple[str, Any] | None = None
    if supports_feature_reuse and cache is not None and cache.query_cache is None:
        persistent_cache_hit = _load_persistent_query_cache(layer, model_id, query)

    preferred_sources: list[str] = []
    if persistent_cache_hit is not None:
        preferred_sources.append(persistent_cache_hit[0])
    preferred_sources.extend(source_candidates)

    errors: list[str] = []
    for source_candidate in dict.fromkeys(preferred_sources):
        try:
            if cache is not None and cache.source_candidate != source_candidate:
                release_online_runtime_hot_cache(
                    keep_source_path=source_candidate,
                )
            engine = MODEL_SESSIONS.get_online_engine(
                source_path=source_candidate,
                model_id=model_id,
            )
            if cache is not None:
                cache.layer_id = layer.id()
                cache.model_id = model_id
                cache.source_candidate = source_candidate
                cache.engine = engine
                if supports_feature_reuse and cache.query_cache is None:
                    if (
                        persistent_cache_hit is not None
                        and source_candidate == persistent_cache_hit[0]
                    ):
                        cache.query_cache = persistent_cache_hit[1]
                    else:
                        cache.query_cache = OnlineQueryCache()

            result = engine.query(
                query,
                cache=cache.query_cache
                if (cache is not None and supports_feature_reuse)
                else None,
            )
            if (
                supports_feature_reuse
                and cache is not None
                and cache.query_cache is not None
            ):
                _save_persistent_query_cache(
                    layer=layer,
                    model_id=model_id,
                        source_path=source_candidate,
                        query_cache=cache.query_cache,
                    )
            if supports_feature_reuse and _should_clear_online_query_state_after_result():
                _clear_online_engine_predictor_state(engine, keep_features=True)
                _flush_torch_memory()
            return result
        except Exception as exc:
            errors.append(f"{source_candidate}: {exc}")
            if cache is not None and cache.source_candidate == source_candidate:
                cache.engine = None
            continue

    try:
        exported_source = _export_online_raster_source(layer, query, model_id=model_id)
        release_online_runtime_hot_cache(
            keep_source_path=exported_source,
        )
        engine = MODEL_SESSIONS.get_online_engine(
            source_path=exported_source,
            model_id=model_id,
        )
        if cache is not None:
            cache.layer_id = layer.id()
            cache.model_id = model_id
            cache.source_candidate = exported_source
            cache.engine = engine
            cache.query_cache = OnlineQueryCache() if supports_feature_reuse else None
        result = engine.query(
            query,
            cache=cache.query_cache
            if (cache is not None and supports_feature_reuse)
            else None,
        )
        if (
            supports_feature_reuse
            and cache is not None
            and cache.query_cache is not None
        ):
            _save_persistent_query_cache(
                layer=layer,
                model_id=model_id,
                source_path=exported_source,
                query_cache=cache.query_cache,
            )
        if supports_feature_reuse and _should_clear_online_query_state_after_result():
            _clear_online_engine_predictor_state(engine, keep_features=True)
            _flush_torch_memory()
        return result
    except (OnlineRasterPreflightError, OnlineRasterExportError) as exc:
        error_prefix = (
            "online-export-preflight"
            if isinstance(exc, OnlineRasterPreflightError)
            else "online-export-failed"
        )
        errors.append(f"{error_prefix}: {exc}")
        msg = describe_online_export_failure(exc)
        MessageTool.MessageLog("\n".join([msg, *errors]), level="warning")
        logger.warning("%s Details: %s", msg, errors)
        raise ValueError(msg) from exc
    except Exception as exc:
        errors.append(f"online-export: {exc}")

    msg = (
        "Failed to open the selected raster layer with GeoSAM. "
        "The plugin also failed to export a local cache for the current online view."
    )
    MessageTool.MessageLog("\n".join([msg, *errors]), level="warning")
    logger.error("%s Details: %s", msg, errors)
    raise ValueError(msg)


def _preview_simplify_tolerance(result: QueryResult) -> float:
    """Return a simplified-preview tolerance in source CRS units."""
    transform = result.mask_transform
    return max(abs(float(transform.a)), abs(float(transform.e))) * 1.5


def query_result_to_render_payload(
    result: QueryResult,
    *,
    render_mode: PreviewRenderMode | None = None,
) -> QueryResultRenderPayload:
    """Convert a GeoSAM query result into canvas and GeoJSON render payloads."""
    configure_geosam_qgis_runtime()
    from shapely.geometry import mapping

    from geosam import MaskVectorizer

    resolved_render_mode = (
        get_preview_render_mode() if render_mode is None else render_mode
    )
    simplify_tolerance = (
        _preview_simplify_tolerance(result)
        if resolved_render_mode == "simplified"
        else None
    )
    properties = {
        "score": float(result.scores[0]) if result.scores.size > 0 else None,
    }
    vectorizer = MaskVectorizer.from_query_result(result, properties=properties)
    to_geometries = getattr(vectorizer, "to_geometries", None)
    if callable(to_geometries):
        shapely_geometries = to_geometries(
            simplify_tolerance=simplify_tolerance,
            preserve_topology=True,
        )
    else:
        frame = vectorizer.to_geodataframe()
        shapely_geometries = list(frame.geometry)
        if simplify_tolerance is not None:
            shapely_geometries = [
                geometry.simplify(simplify_tolerance, preserve_topology=True)
                for geometry in shapely_geometries
                if geometry is not None and not geometry.is_empty
            ]
    geojson_features = [
        {
            "type": "Feature",
            "properties": dict(properties),
            "geometry": mapping(geometry),
        }
        for geometry in shapely_geometries
    ]
    canvas_geometries: list[QgsGeometry] = []
    for geometry in shapely_geometries:
        qgs_geometry = QgsGeometry()
        qgs_geometry.fromWkb(geometry.wkb)
        canvas_geometries.append(qgs_geometry)
    return QueryResultRenderPayload(
        geojson_features=geojson_features,
        canvas_geometries=canvas_geometries,
    )


def query_result_to_geojson_features(
    result: QueryResult,
    *,
    render_mode: PreviewRenderMode | None = None,
) -> list[dict[str, Any]]:
    """Convert a GeoSAM query result into GeoJSON features."""
    return query_result_to_render_payload(
        result,
        render_mode=render_mode,
    ).geojson_features
