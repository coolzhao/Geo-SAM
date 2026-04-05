"""GeoSAM-backed runtime helpers for the QGIS plugin."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from qgis.core import QgsRasterLayer, QgsRectangle

from .messageTool import MessageTool

if TYPE_CHECKING:
    from geosam.engines import QueryResult
    from geosam.models import ModelSpec
    from geosam.query import BoundingBox, Points, PromptSet

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PLUGIN_ROOT / "ui" / "config"
SETTINGS_DEFAULT_PATH = CONFIG_DIR / "default.json"
SETTINGS_USER_PATH = CONFIG_DIR / "user.json"
DEFAULT_MODEL_REPOSITORY = "https://github.com/Fanchengyan/geosam-models"
DEFAULT_MODEL_DIR = PLUGIN_ROOT / "models"
DEFAULT_CACHE_DIR = PLUGIN_ROOT / ".cache"
HELP_LINKS = {
    "Documentation": "https://geo-sam.readthedocs.io/en/latest/",
    "GitHub": "https://github.com/coolzhao/Geo-SAM",
    "Report Bug": "https://github.com/coolzhao/Geo-SAM/issues",
    "Discussions": "https://github.com/coolzhao/Geo-SAM/discussions",
}


@dataclass(slots=True, frozen=True)
class ModelDefinition:
    """Supported downloadable model definition."""

    model_id: str
    label: str
    model_type: Literal["sam", "sam2", "sam3"]
    filename: str
    supports_feature_reuse: bool = True


@dataclass(slots=True, frozen=True)
class FeatureSourceSummary:
    """Summary metadata for a cached feature source."""

    manifest_path: Path
    crs_text: str
    extent: tuple[float, float, float, float]
    chip_count: int
    pixel_area: float


MODEL_DEFINITIONS: tuple[ModelDefinition, ...] = (
    ModelDefinition("sam_b", "SAM Base", "sam", "sam_b.pt"),
    ModelDefinition("sam_l", "SAM Large", "sam", "sam_l.pt"),
    ModelDefinition("sam2_t", "SAM2 Tiny", "sam2", "sam2_t.pt"),
    ModelDefinition("sam2_s", "SAM2 Small", "sam2", "sam2_s.pt"),
    ModelDefinition("sam2_b", "SAM2 Base", "sam2", "sam2_b.pt"),
    ModelDefinition("sam2_l", "SAM2 Large", "sam2", "sam2_l.pt"),
    ModelDefinition("sam2.1_t", "SAM2.1 Tiny", "sam2", "sam2.1_t.pt"),
    ModelDefinition("sam2.1_s", "SAM2.1 Small", "sam2", "sam2.1_s.pt"),
    ModelDefinition("sam2.1_b", "SAM2.1 Base", "sam2", "sam2.1_b.pt"),
    ModelDefinition("sam2.1_l", "SAM2.1 Large", "sam2", "sam2.1_l.pt"),
    ModelDefinition("sam3", "SAM3", "sam3", "sam3.pt", supports_feature_reuse=False),
    ModelDefinition(
        "sam3.1_multiplex",
        "SAM3.1 Multiplex",
        "sam3",
        "sam3.1_multiplex.pt",
        supports_feature_reuse=False,
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_plugin_settings() -> dict[str, Any]:
    """Load effective plugin settings."""
    settings = _load_json(SETTINGS_DEFAULT_PATH)
    settings.update(_load_json(SETTINGS_USER_PATH))
    settings.setdefault("model_repo_url", DEFAULT_MODEL_REPOSITORY)
    settings.setdefault("model_store_dir", str(DEFAULT_MODEL_DIR))
    settings.setdefault("cache_enabled", True)
    settings.setdefault("cache_dir", str(DEFAULT_CACHE_DIR))
    settings.setdefault("cache_max_size_mb", 2048)
    settings.setdefault("selected_model_id", "")
    settings.setdefault("show_boundary", True)
    settings.setdefault("default_minimum_pixels", 0)
    return settings


def save_plugin_settings(updates: dict[str, Any]) -> dict[str, Any]:
    """Persist plugin settings."""
    settings = load_plugin_settings()
    settings.update(updates)
    defaults = _load_json(SETTINGS_DEFAULT_PATH)
    user_settings = {
        key: value
        for key, value in settings.items()
        if defaults.get(key) != value
        or key in {
            "model_repo_url",
            "model_store_dir",
            "cache_enabled",
            "cache_dir",
            "cache_max_size_mb",
            "selected_model_id",
        }
    }
    SETTINGS_USER_PATH.write_text(
        json.dumps(user_settings, indent=4),
        encoding="utf-8",
    )
    return settings


def get_model_definition(model_id: str) -> ModelDefinition:
    """Return a model definition by id."""
    for definition in MODEL_DEFINITIONS:
        if definition.model_id == model_id:
            return definition
    raise KeyError(model_id)


def get_model_directory() -> Path:
    """Return the local model directory."""
    settings = load_plugin_settings()
    model_dir = Path(settings["model_store_dir"]).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_cache_directory() -> Path:
    """Return the local cache directory."""
    settings = load_plugin_settings()
    cache_dir = Path(settings["cache_dir"]).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_checkpoint_path(model_id: str) -> Path:
    """Return the local checkpoint path for a model."""
    definition = get_model_definition(model_id)
    return get_model_directory() / definition.filename


def get_model_display_items() -> list[tuple[str, str]]:
    """Return selectable model entries for UI comboboxes."""
    return [(definition.model_id, definition.label) for definition in MODEL_DEFINITIONS]


def get_model_status_rows() -> list[dict[str, Any]]:
    """Return model download status rows."""
    rows: list[dict[str, Any]] = []
    for definition in MODEL_DEFINITIONS:
        checkpoint_path = get_model_checkpoint_path(definition.model_id)
        rows.append({
            "model_id": definition.model_id,
            "label": definition.label,
            "model_type": definition.model_type,
            "downloaded": checkpoint_path.exists(),
            "path": checkpoint_path,
        })
    return rows


def create_model_spec(model_id: str) -> ModelSpec:
    """Create a :class:`geosam.models.ModelSpec` for the selected model."""
    from geosam.models import ModelSpec

    definition = get_model_definition(model_id)
    checkpoint_path = get_model_checkpoint_path(model_id)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    return ModelSpec(
        model_type=definition.model_type,
        checkpoint_path=checkpoint_path,
        supports_feature_reuse=definition.supports_feature_reuse,
    )


def infer_model_id_from_checkpoint_path(
    checkpoint_path: str | Path,
    *,
    fallback_model_id: str | None = None,
) -> str:
    """Infer a registered model id from a checkpoint file path."""
    checkpoint_name = Path(checkpoint_path).name.lower()
    for definition in MODEL_DEFINITIONS:
        if checkpoint_name == definition.filename.lower():
            return definition.model_id
    for definition in MODEL_DEFINITIONS:
        stem = Path(definition.filename).stem.lower()
        if stem in checkpoint_name:
            return definition.model_id
    if fallback_model_id is not None:
        return fallback_model_id
    msg = (
        "Could not infer a GeoSAM model id from the checkpoint path. "
        "Please select a supported model explicitly."
    )
    raise ValueError(msg)


def create_model_spec_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_id: str | None = None,
    device: str | None = None,
) -> ModelSpec:
    """Create a model spec from an arbitrary checkpoint path."""
    from geosam.models import ModelSpec

    resolved_model_id = infer_model_id_from_checkpoint_path(
        checkpoint_path,
        fallback_model_id=model_id,
    )
    definition = get_model_definition(resolved_model_id)
    return ModelSpec(
        model_type=definition.model_type,
        checkpoint_path=checkpoint_path,
        device=device,
        supports_feature_reuse=definition.supports_feature_reuse,
    )


def dependency_status() -> dict[str, bool]:
    """Return import availability for required runtime dependencies."""
    modules = [
        "geosam",
        "torch",
        "ultralytics",
        "rasterio",
        "geopandas",
        "pyarrow",
    ]
    return {
        module_name: importlib.util.find_spec(module_name) is not None
        for module_name in modules
    }


def install_dependencies() -> tuple[bool, str]:
    """Install GeoSAM plugin dependencies with pip."""
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "geosam",
        "torch",
        "torchvision",
        "ultralytics",
        "rasterio",
        "geopandas",
        "pyarrow",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(
        part for part in [result.stdout.strip(), result.stderr.strip()] if part
    )
    return result.returncode == 0, output


def open_url(url: str) -> None:
    """Open a URL in the desktop browser."""
    QDesktopServices.openUrl(QUrl(url))


def open_path(path: Path) -> None:
    """Open a local path in the desktop file manager."""
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.expanduser().resolve())))


def _resolve_model_download_source(repository: str, filename: str) -> str:
    """Resolve a model source URL or local path."""
    repository_path = Path(repository).expanduser()
    if repository_path.exists():
        return str(repository_path / filename)

    normalized_repository = repository.rstrip("/")
    if (
        "github.com" in normalized_repository
        and "raw.githubusercontent.com" not in normalized_repository
    ):
        normalized_repository = normalized_repository.replace(
            "https://github.com/",
            "https://raw.githubusercontent.com/",
        ).replace(
            "http://github.com/",
            "https://raw.githubusercontent.com/",
        )
        normalized_repository = f"{normalized_repository}/main"
    return f"{normalized_repository}/{filename}"


def download_model(model_id: str) -> Path:
    """Download or copy a model checkpoint into the local store."""
    definition = get_model_definition(model_id)
    settings = load_plugin_settings()
    target_path = get_model_checkpoint_path(model_id)
    if target_path.exists():
        return target_path

    source = _resolve_model_download_source(
        str(settings["model_repo_url"]),
        definition.filename,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if Path(source).exists():
        shutil.copy2(source, target_path)
        return target_path

    download_target = target_path.with_suffix(f"{target_path.suffix}.download")
    urllib.request.urlretrieve(source, download_target)
    download_target.replace(target_path)
    return target_path


def delete_model(model_id: str) -> None:
    """Delete a locally downloaded model checkpoint."""
    checkpoint_path = get_model_checkpoint_path(model_id)
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def get_cache_size_bytes(path: Path | None = None) -> int:
    """Return the current cache size in bytes."""
    cache_dir = get_cache_directory() if path is None else path
    if not cache_dir.exists():
        return 0
    return sum(
        file_path.stat().st_size
        for file_path in cache_dir.rglob("*")
        if file_path.is_file()
    )


def cleanup_cache() -> int:
    """Trim cache files until the configured maximum size is respected."""
    settings = load_plugin_settings()
    cache_dir = get_cache_directory()
    if not settings.get("cache_enabled", True):
        return 0

    max_size_bytes = int(settings["cache_max_size_mb"]) * 1024 * 1024
    current_size = get_cache_size_bytes(cache_dir)
    removed_count = 0
    if current_size <= max_size_bytes:
        return removed_count

    files = sorted(
        (file_path for file_path in cache_dir.rglob("*") if file_path.is_file()),
        key=lambda file_path: file_path.stat().st_mtime,
    )
    for file_path in files:
        file_size = file_path.stat().st_size
        file_path.unlink()
        current_size -= file_size
        removed_count += 1
        if current_size <= max_size_bytes:
            break
    return removed_count


def resolve_feature_manifest_path(feature_dir: str | Path) -> Path:
    """Resolve the manifest file path inside a feature folder."""
    from geosam.engines import FeatureQueryEngine

    return FeatureQueryEngine.resolve_manifest_path(feature_dir)


def describe_feature_source(feature_dir: str | Path) -> FeatureSourceSummary:
    """Load summary metadata for a GeoSAM feature folder."""
    import geopandas as gpd
    import pandas as pd

    manifest_path = resolve_feature_manifest_path(feature_dir)
    if manifest_path.suffix == ".pkl":
        frame = pd.read_pickle(manifest_path)
        frame = frame.set_crs(frame.crs or "EPSG:4326", allow_override=True)
    else:
        frame = gpd.read_parquet(manifest_path)

    crs_text = _resolve_feature_crs_text(frame)
    total_bounds = tuple(float(value) for value in frame.total_bounds)
    transform_values = json.loads(frame.iloc[0]["transform"])
    pixel_area = abs(float(transform_values[0]) * float(transform_values[4]))
    return FeatureSourceSummary(
        manifest_path=manifest_path,
        crs_text=crs_text,
        extent=total_bounds,
        chip_count=len(frame),
        pixel_area=pixel_area,
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
    from geosam import BoundingBox, RasterDataset
    from geosam.query.prompts import normalize_chip_size

    dataset = RasterDataset(
        source_path,
        indexes=bands,
        crs=crs,
        res=res,
    )
    roi_bounds = dataset.bounds
    if extent is not None:
        extent_bounds = BoundingBox(
            extent[0],
            extent[1],
            extent[2],
            extent[3],
            crs=extent_crs or dataset.crs,
        )
        if extent_bounds.crs == dataset.crs:
            roi_bounds = extent_bounds
        else:
            roi_bounds = extent_bounds.to_crs(dataset.crs)
        intersection = roi_bounds & dataset.bounds
        if intersection is None:
            return []
        roi_bounds = intersection

    roi_grid = dataset.grid.to_view(roi_bounds)
    chip_height, chip_width = normalize_chip_size(chip_size)
    stride_height, stride_width = normalize_chip_size(stride)

    def window_starts(size: int, tile_size: int, tile_stride: int) -> list[int]:
        if tile_size >= size:
            return [0]
        starts = list(range(0, max(size - tile_size, 0) + 1, tile_stride))
        last_start = size - tile_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    rectangles: list[tuple[float, float, float, float]] = []
    row_starts = window_starts(roi_grid.height, chip_height, stride_height)
    col_starts = window_starts(roi_grid.width, chip_width, stride_width)
    for row_start in row_starts:
        for col_start in col_starts:
            chip_grid = roi_grid.window(
                row_off=row_start,
                col_off=col_start,
                height=min(chip_height, roi_grid.height),
                width=min(chip_width, roi_grid.width),
            )
            bounds = chip_grid.bounds
            rectangles.append((bounds.left, bounds.bottom, bounds.right, bounds.top))
    return rectangles


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


def query_feature_source(
    feature_dir: str | Path,
    model_id: str,
    query: BoundingBox | Points | PromptSet,
) -> QueryResult:
    """Run a query against a GeoSAM feature folder."""
    from geosam.engines import FeatureQueryEngine

    engine = FeatureQueryEngine(feature_dir, create_model_spec(model_id))
    return engine.query(query)


def query_raster_layer(
    layer: QgsRasterLayer,
    model_id: str,
    query: BoundingBox | Points | PromptSet,
) -> QueryResult:
    """Run a query against a QGIS raster layer."""
    from geosam import OnlineQueryEngine, RasterDataset

    errors: list[str] = []
    for source_candidate in _raster_layer_source_candidates(layer):
        try:
            dataset = RasterDataset(source_candidate)
            engine = OnlineQueryEngine(dataset, create_model_spec(model_id))
            return engine.query(query)
        except Exception as exc:
            errors.append(f"{source_candidate}: {exc}")

    msg = (
        "Failed to open the selected raster layer with GeoSAM. "
        "Only file-backed or GDAL-readable raster sources are supported right now."
    )
    MessageTool.MessageLog("\n".join([msg, *errors]), level="warning")
    raise ValueError(msg)


def query_result_to_geojson_features(result: QueryResult) -> list[dict[str, Any]]:
    """Convert a GeoSAM query result into GeoJSON features."""
    from geosam import MaskVectorizer

    properties = {
        "score": float(result.scores[0]) if result.scores.size > 0 else None,
    }
    payload = MaskVectorizer.from_query_result(result).to_geojson(
        properties=properties,
    )
    return list(payload.get("features", []))
