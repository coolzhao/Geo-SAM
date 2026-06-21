"""Online tile export helpers for Geo-SAM live encoding raster queries."""

from __future__ import annotations

import hashlib
import logging
import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from qgis.PyQt.QtGui import QImage
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsProviderRegistry,
    QgsRasterLayer,
    QgsRectangle,
)

from .geosam_backend import configure_geosam_qgis_runtime
from .plugin_settings import rasterio_proj_data_environment

logger = logging.getLogger(__name__)

OnlineTileExportProgressCallback = Callable[[str, float], None]
OnlineTileExportCancelCallback = Callable[[], bool]

WEB_MERCATOR_CRS_TEXT = "EPSG:3857"
WEB_MERCATOR_HALF_WORLD = 20037508.342789244
WEB_MERCATOR_TILE_SIZE = 256
WEB_MERCATOR_INITIAL_RESOLUTION = 2.0 * WEB_MERCATOR_HALF_WORLD / WEB_MERCATOR_TILE_SIZE
ONLINE_QUERY_REFRESH_MARGIN_PIXELS = 128
ONLINE_TILE_EXPORT_MAX_DIMENSION = 4096
ONLINE_TILE_EXPORT_MAX_TILE_COUNT = 64
_DATACLASS_SLOTS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}
_FROZEN_DATACLASS_KWARGS = {"frozen": True, **_DATACLASS_SLOTS_KWARGS}


class OnlineRasterPreflightError(ValueError):
    """Raised when the current online raster view is unsuitable for export."""


class OnlineRasterExportError(RuntimeError):
    """Raised when the plugin cannot export a local raster from an online view."""


class OnlineRasterUnsupportedProviderError(OnlineRasterExportError):
    """Raised when the active online provider cannot be exported via tiles."""


class OnlineRasterTileMetadataError(OnlineRasterExportError):
    """Raised when tile metadata cannot be derived from the active layer."""


class OnlineRasterTileDownloadError(OnlineRasterExportError):
    """Raised when one or more online tiles cannot be downloaded or decoded."""


@dataclass(**_FROZEN_DATACLASS_KWARGS)
class OnlineTileProviderInfo:
    """Resolved tile-provider metadata for an online raster layer."""

    provider_kind: Literal["xyz", "wmts"]
    provider_type: str
    source_uri: str
    url_template: str
    crs_text: str
    min_zoom: int
    max_zoom: int
    tile_size: int
    y_origin: Literal["xyz", "tms"] = "xyz"
    tile_matrix_set: str | None = None
    layer_name: str | None = None
    style_name: str | None = None
    image_format: str | None = None


@dataclass(**_FROZEN_DATACLASS_KWARGS)
class OnlineTileExportPlan:
    """Prepared tile export for a supported online raster source."""

    provider: OnlineTileProviderInfo
    model_id: str
    chip_size: tuple[int, int]
    pixel_size: tuple[float, float]
    query_extent_layer_crs: QgsRectangle
    export_extent_layer_crs: QgsRectangle
    export_extent_tile_crs: QgsRectangle
    tile_zoom: int
    tile_column_range: tuple[int, int]
    tile_row_range: tuple[int, int]
    output_width: int
    output_height: int
    output_crs_text: str
    source_fingerprint: str


def online_export_failure_message(export_error: Exception) -> str:
    """Return a user-facing message for online raster export failures."""
    if isinstance(export_error, OnlineRasterPreflightError):
        return (
            "The selected online query area is too large for one Geo-SAM export. "
            "Zoom in further or draw a smaller bounding box."
        )
    if isinstance(export_error, OnlineRasterUnsupportedProviderError):
        return (
            "This online raster source is not currently supported for Geo-SAM tile export. "
            "Geo-SAM currently supports XYZ and WMTS tiled providers only."
        )
    if isinstance(export_error, OnlineRasterTileMetadataError):
        return (
            "Geo-SAM could not derive tile coordinates from the active online layer. "
            "Please verify the layer is a standard XYZ or WMTS tile source."
        )
    if isinstance(export_error, OnlineRasterTileDownloadError):
        return (
            "Geo-SAM failed to download or decode the online tiles for the current view. "
            "Please try again or use another online tile provider."
        )
    return (
        "GeoSAM could not prepare a local raster cache from the current online map view. "
        "Zoom in further and try again."
    )


def export_online_raster_source(
    layer: QgsRasterLayer,
    query: Any,
    *,
    model_id: str,
    chip_size: tuple[int, int],
    source_fingerprint: str,
    cache_directory: Path,
) -> str:
    """Export supported online tiles into the cache as a local GeoTIFF."""
    export_plan = prepare_online_raster_export_plan(
        layer,
        query,
        model_id=model_id,
        chip_size=chip_size,
        source_fingerprint=source_fingerprint,
    )
    return export_online_raster_plan(
        export_plan,
        cache_directory=cache_directory,
        layer_name=layer.name(),
    )


def prepare_online_raster_export_plan(
    layer: QgsRasterLayer,
    query: Any,
    *,
    model_id: str,
    chip_size: tuple[int, int],
    source_fingerprint: str,
) -> OnlineTileExportPlan:
    """Prepare an online raster export plan on the main thread.

    Parameters
    ----------
    layer : QgsRasterLayer
        Active online raster layer.
    query : Any
        GeoSAM query object.
    model_id : str
        Selected GeoSAM model identifier.
    chip_size : tuple[int, int]
        GeoSAM chip size for the selected model.
    source_fingerprint : str
        Stable fingerprint used to namespace exported caches.

    Returns
    -------
    OnlineTileExportPlan
        Prepared plan describing the online tile export.
    """
    return _prepare_online_tile_export(
        layer,
        query,
        model_id=model_id,
        chip_size=chip_size,
        source_fingerprint=source_fingerprint,
    )


def export_online_raster_plan(
    export_plan: OnlineTileExportPlan,
    *,
    cache_directory: Path,
    layer_name: str,
    progress_callback: OnlineTileExportProgressCallback | None = None,
    is_canceled: OnlineTileExportCancelCallback | None = None,
) -> str:
    """Export a prepared online tile plan into the cache as a GeoTIFF.

    Parameters
    ----------
    export_plan : OnlineTileExportPlan
        Prepared export plan captured on the main thread.
    cache_directory : Path
        Cache directory used to store the exported GeoTIFF.
    layer_name : str
        Human-readable layer name used in log messages.
    progress_callback : OnlineTileExportProgressCallback | None, optional
        Callback receiving ``(stage_text, progress_percent)`` updates.
    is_canceled : OnlineTileExportCancelCallback | None, optional
        Callback returning ``True`` when export should stop.

    Returns
    -------
    str
        Exported GeoTIFF path.
    """
    file_stem = _online_raster_cache_stem(export_plan)
    destination_path = cache_directory / f"{file_stem}.tif"
    if destination_path.exists():
        return str(destination_path)

    legacy_destination_path = _find_legacy_model_scoped_online_raster_cache(
        export_plan,
        cache_directory=cache_directory,
    )
    if legacy_destination_path is not None:
        return str(legacy_destination_path)

    return str(
        _export_online_tiles_to_geotiff(
            layer_name,
            export_plan=export_plan,
            destination_path=destination_path,
            progress_callback=progress_callback,
            is_canceled=is_canceled,
        )
    )


def _online_raster_cache_stem(export_plan: OnlineTileExportPlan) -> str:
    """Return the model-independent online raster cache stem.

    Parameters
    ----------
    export_plan : OnlineTileExportPlan
        Prepared export plan that identifies the source tile coverage.

    Returns
    -------
    str
        Filesystem-safe cache stem for the exported GeoTIFF.

    Notes
    -----
    Online raster GeoTIFFs store source imagery, not model features. They can
    be reused across GeoSAM models when the tile source, zoom, and tile range
    are identical. Encoded feature caches remain model-scoped elsewhere.
    """
    return _sanitize_path_component(
        "_".join([
            export_plan.source_fingerprint,
            export_plan.provider.provider_kind,
            f"z{export_plan.tile_zoom}",
            f"c{export_plan.tile_column_range[0]}-{export_plan.tile_column_range[1]}",
            f"r{export_plan.tile_row_range[0]}-{export_plan.tile_row_range[1]}",
        ])
    )


def _find_legacy_model_scoped_online_raster_cache(
    export_plan: OnlineTileExportPlan,
    *,
    cache_directory: Path,
) -> Path | None:
    """Return an existing pre-migration online raster cache path.

    Parameters
    ----------
    export_plan : OnlineTileExportPlan
        Prepared export plan that identifies the source tile coverage.
    cache_directory : Path
        Directory containing exported online raster GeoTIFFs.

    Returns
    -------
    Path | None
        Existing model-scoped cache file when one matches the same source tile
        coverage, otherwise ``None``.
    """
    if not cache_directory.exists():
        return None

    legacy_stem_suffix = _online_raster_cache_stem(export_plan)
    legacy_pattern = f"*_{legacy_stem_suffix}.tif"
    for legacy_path in sorted(cache_directory.glob(legacy_pattern)):
        if legacy_path.is_file():
            return legacy_path
    return None


def _sanitize_path_component(value: str) -> str:
    """Return a filesystem-friendly path component."""
    sanitized = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value.strip()
    )
    return sanitized or "unnamed"


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


def _provider_source_uri(layer: QgsRasterLayer) -> str:
    """Return the most informative source URI available for a raster layer."""
    provider = layer.dataProvider()
    if provider is not None and hasattr(provider, "dataSourceUri"):
        try:
            provider_uri = str(provider.dataSourceUri())
        except Exception:
            provider_uri = ""
        if provider_uri:
            return provider_uri
    for candidate in _raster_layer_source_candidates(layer):
        if candidate:
            return candidate
    return ""


def _manual_decode_source_uri(source_uri: str) -> dict[str, str]:
    """Decode a QGIS-style query string while preserving embedded tile URLs."""
    known_keys = {
        "authcfg",
        "contextualWMSLegend",
        "crs",
        "dpiMode",
        "featureCount",
        "format",
        "layers",
        "password",
        "referer",
        "style",
        "styles",
        "tileMatrixSet",
        "tilePixelRatio",
        "type",
        "url",
        "username",
        "version",
        "zmax",
        "zmin",
    }
    decoded: dict[str, str] = {}
    active_key: str | None = None
    for raw_segment in unquote(source_uri).split("&"):
        if "=" in raw_segment:
            key, value = raw_segment.split("=", maxsplit=1)
        else:
            key, value = raw_segment, ""
        key = key.strip()
        if active_key == "url" and key not in known_keys:
            decoded["url"] = f"{decoded['url']}&{raw_segment}"
            continue
        decoded[key] = value
        active_key = key
    return {key: value for key, value in decoded.items() if key}


def _decoded_provider_uri(layer: QgsRasterLayer) -> dict[str, str]:
    """Decode provider metadata for an online raster layer."""
    source_uri = _provider_source_uri(layer)
    if not source_uri:
        return {}

    provider_name = layer.providerType()
    try:
        metadata = QgsProviderRegistry.instance().providerMetadata(provider_name)
    except Exception:
        metadata = None
    if metadata is not None and hasattr(metadata, "decodeUri"):
        try:
            decoded_uri = metadata.decodeUri(source_uri)
        except Exception:
            decoded_uri = None
        else:
            if isinstance(decoded_uri, dict) and len(decoded_uri) > 0:
                return {
                    str(key): "" if value is None else str(value)
                    for key, value in decoded_uri.items()
                }

    return _manual_decode_source_uri(source_uri)


def _transform_rectangle(
    rectangle: QgsRectangle,
    *,
    source_crs: QgsCoordinateReferenceSystem,
    destination_crs: QgsCoordinateReferenceSystem,
) -> QgsRectangle:
    """Transform a rectangle between CRS definitions."""
    if source_crs == destination_crs:
        return QgsRectangle(rectangle)
    transform = QgsCoordinateTransform(
        source_crs,
        destination_crs,
        QgsProject.instance().transformContext(),
    )
    return transform.transformBoundingBox(rectangle)


def _query_in_layer_crs(layer: QgsRasterLayer, query: Any) -> Any:
    """Normalize a query into the layer CRS."""
    configure_geosam_qgis_runtime()
    crs_text = layer.crs().authid() or layer.crs().toWkt()
    if query.crs == QgsCoordinateReferenceSystem(crs_text):
        return query
    if query.crs is None:
        return query
    return query if str(query.crs) == crs_text else query.to_crs(crs_text)


def _rectangle_for_query_bounds(layer: QgsRasterLayer, query: Any) -> QgsRectangle:
    """Return query bounds as a rectangle in the layer CRS."""
    configure_geosam_qgis_runtime()
    from geosam.query import query_bounds

    projected_query = _query_in_layer_crs(layer, query)
    bounds = query_bounds(projected_query)
    return QgsRectangle(
        float(bounds.left),
        float(bounds.bottom),
        float(bounds.right),
        float(bounds.top),
    )


def _resolve_pixel_size(layer: QgsRasterLayer) -> tuple[float, float]:
    """Return raster pixel size in layer units."""
    pixel_size_x = abs(float(layer.rasterUnitsPerPixelX()))
    pixel_size_y = abs(float(layer.rasterUnitsPerPixelY()))
    if pixel_size_x > 0 and pixel_size_y > 0:
        return pixel_size_x, pixel_size_y

    extent = layer.extent()
    width = max(int(layer.width()), 1)
    height = max(int(layer.height()), 1)
    return (
        abs(float(extent.width())) / width,
        abs(float(extent.height())) / height,
    )


def _resolve_online_tile_provider(layer: QgsRasterLayer) -> OnlineTileProviderInfo:
    """Resolve supported tile-provider metadata from a raster layer."""
    provider_type = layer.providerType()
    source_uri = _provider_source_uri(layer)
    decoded_uri = _decoded_provider_uri(layer)
    source_uri_lower = source_uri.lower()
    type_value = decoded_uri.get("type", "").lower()
    url_template = decoded_uri.get("url", "").strip() or source_uri.strip()

    if type_value == "xyz" or (
        "{x}" in url_template and "{y}" in url_template and "{z}" in url_template
    ):
        try:
            min_zoom = int(decoded_uri.get("zmin", "0") or 0)
        except ValueError:
            min_zoom = 0
        try:
            max_zoom = int(decoded_uri.get("zmax", "22") or 22)
        except ValueError:
            max_zoom = 22
        y_origin: Literal["xyz", "tms"] = "tms" if "{-y}" in url_template else "xyz"
        return OnlineTileProviderInfo(
            provider_kind="xyz",
            provider_type=provider_type,
            source_uri=source_uri,
            url_template=url_template,
            crs_text=WEB_MERCATOR_CRS_TEXT,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            tile_size=WEB_MERCATOR_TILE_SIZE,
            y_origin=y_origin,
        )

    wmts_like = (
        "tilematrixset" in {key.lower() for key in decoded_uri}
        or "service=wmts" in source_uri_lower
        or "request=gettile" in source_uri_lower
    )
    if not wmts_like:
        msg = (
            "This online raster source is not currently supported for tile export. "
            "Geo-SAM currently supports XYZ and WMTS tiled providers only."
        )
        logger.error(msg)
        raise OnlineRasterUnsupportedProviderError(msg)

    tile_matrix_set = decoded_uri.get("tileMatrixSet", "").strip()
    if tile_matrix_set == "":
        msg = "Geo-SAM could not determine the WMTS tile matrix set from the active layer."
        logger.error(msg)
        raise OnlineRasterTileMetadataError(msg)

    if tile_matrix_set.lower() not in {"googlemapscompatible", "webmercatorquad", "epsg:3857"}:
        msg = (
            "Geo-SAM currently supports WMTS tile export only for Web Mercator "
            "matrix sets such as GoogleMapsCompatible."
        )
        logger.error("%s Matrix set=%s", msg, tile_matrix_set)
        raise OnlineRasterUnsupportedProviderError(msg)

    return OnlineTileProviderInfo(
        provider_kind="wmts",
        provider_type=provider_type,
        source_uri=source_uri,
        url_template=url_template,
        crs_text=decoded_uri.get("crs", WEB_MERCATOR_CRS_TEXT),
        min_zoom=0,
        max_zoom=22,
        tile_size=WEB_MERCATOR_TILE_SIZE,
        tile_matrix_set=tile_matrix_set,
        layer_name=decoded_uri.get("layers") or decoded_uri.get("layer"),
        style_name=decoded_uri.get("styles") or decoded_uri.get("style"),
        image_format=decoded_uri.get("format"),
    )


def _web_mercator_zoom_for_pixel_size(
    pixel_size: tuple[float, float],
    *,
    min_zoom: int,
    max_zoom: int,
) -> int:
    """Estimate a Web Mercator zoom level from the current pixel size."""
    reference_resolution = max(float(pixel_size[0]), float(pixel_size[1]))
    if reference_resolution <= 0:
        msg = "Geo-SAM could not derive a valid zoom level from the current raster resolution."
        logger.error(msg)
        raise OnlineRasterTileMetadataError(msg)
    zoom = int(round(math.log2(WEB_MERCATOR_INITIAL_RESOLUTION / reference_resolution)))
    return max(min(zoom, max_zoom), min_zoom)


def _web_mercator_resolution_for_zoom(zoom: int) -> float:
    """Return the ground resolution for one Web Mercator zoom level."""
    return WEB_MERCATOR_INITIAL_RESOLUTION / (2**zoom)


def _export_extent_for_query(
    query_extent: QgsRectangle,
    *,
    pixel_size: tuple[float, float],
    chip_size: tuple[int, int],
) -> QgsRectangle:
    """Build an export extent that fully contains the active query bounds."""
    chip_height, chip_width = chip_size
    minimum_width = pixel_size[0] * chip_width
    minimum_height = pixel_size[1] * chip_height
    margin_x = pixel_size[0] * ONLINE_QUERY_REFRESH_MARGIN_PIXELS
    margin_y = pixel_size[1] * ONLINE_QUERY_REFRESH_MARGIN_PIXELS
    center_x = float(query_extent.center().x())
    center_y = float(query_extent.center().y())
    target_width = max(float(query_extent.width()) + 2.0 * margin_x, minimum_width)
    target_height = max(float(query_extent.height()) + 2.0 * margin_y, minimum_height)
    return QgsRectangle(
        center_x - target_width / 2.0,
        center_y - target_height / 2.0,
        center_x + target_width / 2.0,
        center_y + target_height / 2.0,
    )


def _web_mercator_tile_bounds(
    tile_x: int,
    tile_y: int,
    zoom: int,
    *,
    tile_size: int,
) -> tuple[float, float, float, float]:
    """Return Web Mercator bounds for one XYZ tile."""
    resolution = WEB_MERCATOR_INITIAL_RESOLUTION / (2**zoom)
    tile_span = tile_size * resolution
    left = -WEB_MERCATOR_HALF_WORLD + tile_x * tile_span
    right = left + tile_span
    top = WEB_MERCATOR_HALF_WORLD - tile_y * tile_span
    bottom = top - tile_span
    return left, bottom, right, top


def _web_mercator_tile_range(
    rectangle: QgsRectangle,
    *,
    zoom: int,
    tile_size: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the tile column and row range covering a Web Mercator extent."""
    resolution = WEB_MERCATOR_INITIAL_RESOLUTION / (2**zoom)
    tile_span = tile_size * resolution
    max_index = 2**zoom - 1
    min_column = int(
        math.floor((float(rectangle.xMinimum()) + WEB_MERCATOR_HALF_WORLD) / tile_span)
    )
    max_column = int(
        math.floor((float(rectangle.xMaximum()) + WEB_MERCATOR_HALF_WORLD) / tile_span)
    )
    min_row = int(
        math.floor((WEB_MERCATOR_HALF_WORLD - float(rectangle.yMaximum())) / tile_span)
    )
    max_row = int(
        math.floor((WEB_MERCATOR_HALF_WORLD - float(rectangle.yMinimum())) / tile_span)
    )
    return (
        (max(min_column, 0), min(max_column, max_index)),
        (max(min_row, 0), min(max_row, max_index)),
    )


def _prepare_online_tile_export(
    layer: QgsRasterLayer,
    query: Any,
    *,
    model_id: str,
    chip_size: tuple[int, int],
    source_fingerprint: str,
) -> OnlineTileExportPlan:
    """Prepare a tile export plan for a supported online raster layer."""
    provider = _resolve_online_tile_provider(layer)
    query_extent = _rectangle_for_query_bounds(layer, query)
    current_pixel_size = _resolve_pixel_size(layer)
    layer_crs = layer.crs()
    tile_crs = QgsCoordinateReferenceSystem(provider.crs_text)
    if not tile_crs.isValid():
        msg = "Geo-SAM could not determine a valid tile CRS for the active online layer."
        logger.error(msg)
        raise OnlineRasterTileMetadataError(msg)
    if tile_crs.authid() not in {WEB_MERCATOR_CRS_TEXT, "EPSG:900913"}:
        msg = (
            "Geo-SAM currently supports online tile export only for Web Mercator "
            "XYZ/WMTS layers."
        )
        logger.error("%s Tile CRS=%s", msg, tile_crs.authid() or tile_crs.toWkt())
        raise OnlineRasterUnsupportedProviderError(msg)

    current_zoom_level = _web_mercator_zoom_for_pixel_size(
        current_pixel_size,
        min_zoom=provider.min_zoom,
        max_zoom=provider.max_zoom,
    )
    usable_chip_height = max(chip_size[0] - 2 * ONLINE_QUERY_REFRESH_MARGIN_PIXELS, 1)
    usable_chip_width = max(chip_size[1] - 2 * ONLINE_QUERY_REFRESH_MARGIN_PIXELS, 1)
    required_resolution = max(
        float(query_extent.width()) / usable_chip_width,
        float(query_extent.height()) / usable_chip_height,
        max(current_pixel_size[0], current_pixel_size[1]),
    )
    max_zoom_for_query = int(
        math.floor(
            math.log2(WEB_MERCATOR_INITIAL_RESOLUTION / max(required_resolution, 1e-9))
        )
    )
    zoom_level = max(
        provider.min_zoom,
        min(current_zoom_level, max_zoom_for_query, provider.max_zoom),
    )
    export_pixel_size = (
        _web_mercator_resolution_for_zoom(zoom_level),
        _web_mercator_resolution_for_zoom(zoom_level),
    )
    export_extent_layer_crs = _export_extent_for_query(
        query_extent,
        pixel_size=export_pixel_size,
        chip_size=chip_size,
    )
    export_extent_tile_crs = _transform_rectangle(
        export_extent_layer_crs,
        source_crs=layer_crs,
        destination_crs=tile_crs,
    )
    tile_column_range, tile_row_range = _web_mercator_tile_range(
        export_extent_tile_crs,
        zoom=zoom_level,
        tile_size=provider.tile_size,
    )
    tile_count_x = tile_column_range[1] - tile_column_range[0] + 1
    tile_count_y = tile_row_range[1] - tile_row_range[0] + 1
    output_width = tile_count_x * provider.tile_size
    output_height = tile_count_y * provider.tile_size
    if (
        output_width > ONLINE_TILE_EXPORT_MAX_DIMENSION
        or output_height > ONLINE_TILE_EXPORT_MAX_DIMENSION
        or tile_count_x * tile_count_y > ONLINE_TILE_EXPORT_MAX_TILE_COUNT
    ):
        msg = (
            "The selected online query area is too large for one Geo-SAM tile export. "
            "Zoom in further or draw a smaller bounding box."
        )
        logger.error(
            "%s width=%s height=%s tile_count=%s",
            msg,
            output_width,
            output_height,
            tile_count_x * tile_count_y,
        )
        raise OnlineRasterPreflightError(msg)

    return OnlineTileExportPlan(
        provider=provider,
        model_id=model_id,
        chip_size=chip_size,
        pixel_size=export_pixel_size,
        query_extent_layer_crs=query_extent,
        export_extent_layer_crs=export_extent_layer_crs,
        export_extent_tile_crs=export_extent_tile_crs,
        tile_zoom=zoom_level,
        tile_column_range=tile_column_range,
        tile_row_range=tile_row_range,
        output_width=output_width,
        output_height=output_height,
        output_crs_text=provider.crs_text,
        source_fingerprint=hashlib.sha256(source_fingerprint.encode("utf-8")).hexdigest()[:16],
    )


def _tile_request_url(
    provider: OnlineTileProviderInfo,
    *,
    zoom: int,
    tile_column: int,
    tile_row: int,
) -> str:
    """Build a tile request URL for an XYZ or WMTS provider."""
    if provider.provider_kind == "xyz":
        rendered_url = provider.url_template.replace("{x}", str(tile_column))
        rendered_url = rendered_url.replace(
            "{y}",
            str(tile_row if provider.y_origin == "xyz" else (2**zoom - 1 - tile_row)),
        )
        rendered_url = rendered_url.replace("{-y}", str(2**zoom - 1 - tile_row))
        rendered_url = rendered_url.replace("{z}", str(zoom))
        return rendered_url

    matrix_id = str(zoom)
    rendered_url = provider.url_template
    if "{TileMatrix}" in rendered_url or "{TileRow}" in rendered_url:
        rendered_url = rendered_url.replace(
            "{TileMatrixSet}", provider.tile_matrix_set or ""
        )
        rendered_url = rendered_url.replace("{TileMatrix}", matrix_id)
        rendered_url = rendered_url.replace("{TileRow}", str(tile_row))
        rendered_url = rendered_url.replace("{TileCol}", str(tile_column))
        return rendered_url

    split_url = urlsplit(rendered_url)
    query_items = dict(parse_qsl(split_url.query, keep_blank_values=True))
    query_items.update({
        "SERVICE": query_items.get("SERVICE", "WMTS"),
        "REQUEST": "GetTile",
        "VERSION": query_items.get("VERSION", "1.0.0"),
        "TILEMATRIXSET": provider.tile_matrix_set or "",
        "TILEMATRIX": matrix_id,
        "TILEROW": str(tile_row),
        "TILECOL": str(tile_column),
    })
    if provider.layer_name:
        query_items["LAYER"] = provider.layer_name
    if provider.style_name is not None:
        query_items["STYLE"] = provider.style_name
    if provider.image_format:
        query_items["FORMAT"] = provider.image_format
    return urlunsplit(
        (
            split_url.scheme,
            split_url.netloc,
            split_url.path,
            urlencode(query_items),
            split_url.fragment,
        )
    )


def _download_tile_array(
    provider: OnlineTileProviderInfo,
    *,
    zoom: int,
    tile_column: int,
    tile_row: int,
) -> Any:
    """Download one online tile and return it as an RGB array."""
    import numpy as np

    tile_url = _tile_request_url(
        provider,
        zoom=zoom,
        tile_column=tile_column,
        tile_row=tile_row,
    )
    request = urllib.request.Request(
        tile_url,
        headers={"User-Agent": "Geo-SAM/QGIS Tile Export"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            tile_bytes = response.read()
    except Exception as exc:
        msg = f"Failed to download online tile z={zoom} x={tile_column} y={tile_row}."
        logger.error("%s %s", msg, exc)
        raise OnlineRasterTileDownloadError(msg) from exc

    image = QImage()
    if not image.loadFromData(tile_bytes):
        msg = f"Failed to decode online tile z={zoom} x={tile_column} y={tile_row}."
        logger.error(msg)
        raise OnlineRasterTileDownloadError(msg)

    rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
    if rgb_image.width() != provider.tile_size or rgb_image.height() != provider.tile_size:
        msg = (
            "Geo-SAM received an unexpected tile size from the active online provider. "
            f"Expected {provider.tile_size}x{provider.tile_size}, "
            f"got {rgb_image.width()}x{rgb_image.height()}."
        )
        logger.error(msg)
        raise OnlineRasterTileDownloadError(msg)

    bits = rgb_image.bits()
    bits.setsize(rgb_image.sizeInBytes())
    row_bytes = rgb_image.bytesPerLine()
    tile_array = np.frombuffer(bits, dtype=np.uint8).reshape(
        (rgb_image.height(), row_bytes)
    )
    return tile_array[:, : rgb_image.width() * 3].reshape(
        (rgb_image.height(), rgb_image.width(), 3)
    ).copy()


def _write_tile_mosaic_geotiff(
    mosaic_array: Any,
    *,
    bounds: tuple[float, float, float, float],
    crs_text: str,
    destination_path: Path,
) -> Path:
    """Write an RGB tile mosaic to a georeferenced GeoTIFF."""
    with rasterio_proj_data_environment():
        import rasterio
        from rasterio.transform import from_bounds

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        height, width, _ = mosaic_array.shape
        transform = from_bounds(*bounds, width=width, height=height)
        with rasterio.open(
            destination_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=3,
            dtype=mosaic_array.dtype,
            crs=crs_text,
            transform=transform,
        ) as dataset:
            dataset.write(mosaic_array.transpose(2, 0, 1))
    return destination_path


def _export_online_tiles_to_geotiff(
    layer_name: str,
    *,
    export_plan: OnlineTileExportPlan,
    destination_path: Path,
    progress_callback: OnlineTileExportProgressCallback | None = None,
    is_canceled: OnlineTileExportCancelCallback | None = None,
) -> Path:
    """Download and mosaic online tiles into a GeoTIFF cache entry."""
    import numpy as np

    tile_count_x = export_plan.tile_column_range[1] - export_plan.tile_column_range[0] + 1
    tile_count_y = export_plan.tile_row_range[1] - export_plan.tile_row_range[0] + 1
    total_tile_count = tile_count_x * tile_count_y
    tile_size = export_plan.provider.tile_size
    mosaic_array = np.zeros(
        (tile_count_y * tile_size, tile_count_x * tile_size, 3),
        dtype=np.uint8,
    )
    downloaded_tile_count = 0
    for row_index, tile_row in enumerate(
        range(export_plan.tile_row_range[0], export_plan.tile_row_range[1] + 1)
    ):
        for column_index, tile_column in enumerate(
            range(export_plan.tile_column_range[0], export_plan.tile_column_range[1] + 1)
        ):
            if is_canceled is not None and is_canceled():
                msg = "Online raster export task was canceled."
                logger.info(msg)
                raise OnlineRasterExportError(msg)
            tile_array = _download_tile_array(
                export_plan.provider,
                zoom=export_plan.tile_zoom,
                tile_column=tile_column,
                tile_row=tile_row,
            )
            row_offset = row_index * tile_size
            column_offset = column_index * tile_size
            mosaic_array[
                row_offset : row_offset + tile_size,
                column_offset : column_offset + tile_size,
                :,
            ] = tile_array
            downloaded_tile_count += 1
            if progress_callback is not None:
                progress_callback(
                    "Downloading online tiles",
                    float(downloaded_tile_count / total_tile_count * 20.0),
                )

    left, _, _, top = _web_mercator_tile_bounds(
        export_plan.tile_column_range[0],
        export_plan.tile_row_range[0],
        export_plan.tile_zoom,
        tile_size=tile_size,
    )
    _, bottom, right, _ = _web_mercator_tile_bounds(
        export_plan.tile_column_range[1],
        export_plan.tile_row_range[1],
        export_plan.tile_zoom,
        tile_size=tile_size,
    )
    logger.info(
        "Exporting online tiles for layer '%s' kind=%s zoom=%s cols=%s rows=%s",
        layer_name,
        export_plan.provider.provider_kind,
        export_plan.tile_zoom,
        export_plan.tile_column_range,
        export_plan.tile_row_range,
    )
    return _write_tile_mosaic_geotiff(
        mosaic_array,
        bounds=(left, bottom, right, top),
        crs_text=export_plan.output_crs_text,
        destination_path=destination_path,
    )
