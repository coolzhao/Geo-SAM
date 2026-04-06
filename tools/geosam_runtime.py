"""GeoSAM-backed runtime helpers for the QGIS plugin."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeAlias
from urllib.parse import urlparse

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsProject,
    QgsRasterFileWriter,
    QgsRasterLayer,
    QgsRasterPipe,
    QgsRectangle,
)

from .messageTool import MessageTool

if TYPE_CHECKING:
    from geosam.engines import QueryResult
    from geosam.models import ModelSpec
    from geosam.query import BoundingBox, Points, PromptSet

logger = logging.getLogger(__name__)

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PLUGIN_ROOT / "ui" / "config"
SETTINGS_DEFAULT_PATH = CONFIG_DIR / "default.json"
SETTINGS_USER_PATH = CONFIG_DIR / "user.json"
DEFAULT_MODEL_DIR = PLUGIN_ROOT / "models"
DEFAULT_CACHE_DIR = PLUGIN_ROOT / ".cache"
LOCAL_GEOSAM_REPOSITORY = Path("/Users/fancy/Documents/GitHub/geosam")
HELP_LINKS = {
    "Documentation": "https://geo-sam.readthedocs.io/en/latest/",
    "GitHub": "https://github.com/coolzhao/Geo-SAM",
    "Report Bug": "https://github.com/coolzhao/Geo-SAM/issues",
    "Discussions": "https://github.com/coolzhao/Geo-SAM/discussions",
}
ONLINE_QUERY_REFRESH_MARGIN_PIXELS = 128

try:
    from geosam.runtime import (
        DEFAULT_MODEL_REPOSITORY as RUNTIME_DEFAULT_MODEL_REPOSITORY,
        MODEL_DEFINITIONS as RUNTIME_MODEL_DEFINITIONS,
        ModelDefinition as RuntimeModelDefinition,
    )
except ModuleNotFoundError:
    DEFAULT_MODEL_REPOSITORY = "https://github.com/Fanchengyan/geosam-models"

    @dataclass(slots=True, frozen=True)
    class ModelDefinition:
        """Supported downloadable model definition."""

        model_id: str
        label: str
        model_type: Literal["sam", "sam2", "sam3"]
        filename: str
        supports_feature_reuse: bool = True


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
        ModelDefinition(
            "sam3",
            "SAM3",
            "sam3",
            "sam3.pt",
            supports_feature_reuse=False,
        ),
        ModelDefinition(
            "sam3.1_multiplex",
            "SAM3.1 Multiplex",
            "sam3",
            "sam3.1_multiplex.pt",
            supports_feature_reuse=False,
        ),
    )
else:
    DEFAULT_MODEL_REPOSITORY = RUNTIME_DEFAULT_MODEL_REPOSITORY
    ModelDefinition = RuntimeModelDefinition  # type: ignore[misc]
    MODEL_DEFINITIONS = RUNTIME_MODEL_DEFINITIONS


@dataclass(slots=True, frozen=True)
class FeatureSourceSummary:
    """Summary metadata for a cached GeoSAM feature source."""

    manifest_path: Path
    crs_text: str
    extent: tuple[float, float, float, float]
    chip_count: int
    pixel_area: float
    model_id: str | None = None
    checkpoint_path: str | None = None


@dataclass(slots=True)
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


RealtimeQueryProgressCallback: TypeAlias = Callable[[str, float], None]
RealtimeQueryCancelCallback: TypeAlias = Callable[[], bool]


class _RealtimeQueryCanceledError(RuntimeError):
    """Internal exception used to stop a background realtime query task."""


@dataclass(slots=True, frozen=True)
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
    cache_directory : Path
        Directory where realtime query caches are persisted.
    source_candidates : tuple[str, ...]
        Candidate raster sources that can be opened without touching QGIS APIs.
    supports_feature_reuse : bool
        Whether the selected model supports reusable encoded features.

    """

    model_id: str
    layer_id: str
    layer_name: str
    source_fingerprint: str
    cache_directory: Path
    source_candidates: tuple[str, ...]
    supports_feature_reuse: bool


@dataclass(slots=True)
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
            del self._online_engines[key]
            removed_count += 1
        return removed_count

    def get_feature_engine(self, *, feature_dir: str | Path, model_id: str) -> Any:
        """Return a cached feature-query engine or create a new one."""
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
                del store[key]
                removed_count += 1
        return removed_count


MODEL_SESSIONS = _ModelSessionRegistry()


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_path_component(value: str) -> str:
    """Convert an arbitrary label into a filesystem-safe path component."""
    sanitized = "".join(
        character
        if character.isalnum() or character in {"-", "_", "."}
        else "_"
        for character in value.strip()
    )
    sanitized = sanitized.strip("._")
    return sanitized or "unnamed"


def load_plugin_settings() -> dict[str, Any]:
    """Load effective plugin settings."""
    settings = _default_plugin_settings()
    settings.update(_load_json(SETTINGS_USER_PATH))
    return settings


def _default_plugin_settings() -> dict[str, Any]:
    """Return the effective default plugin settings."""
    settings = _load_json(SETTINGS_DEFAULT_PATH)
    settings.setdefault("model_repo_url", DEFAULT_MODEL_REPOSITORY)
    settings.setdefault("model_store_dir", str(DEFAULT_MODEL_DIR))
    settings.setdefault("cache_enabled", True)
    settings.setdefault("cache_dir", str(DEFAULT_CACHE_DIR))
    settings.setdefault("cache_max_size_mb", 2048)
    settings.setdefault("clear_cache_on_plugin_close", True)
    settings.setdefault("selected_model_id", "")
    settings.setdefault("show_boundary", True)
    settings.setdefault("default_minimum_pixels", 0)
    return settings


def save_plugin_settings(updates: dict[str, Any]) -> dict[str, Any]:
    """Persist plugin settings."""
    settings = load_plugin_settings()
    settings.update(updates)
    defaults = _default_plugin_settings()
    user_settings = {
        key: value
        for key, value in settings.items()
        if defaults.get(key) != value
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
    logger.error("Unknown GeoSAM model id requested: %s", model_id)
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


def get_layer_cache_directory(layer_name: str) -> Path:
    """Return the cache root for a raster layer name."""
    layer_cache_dir = get_cache_directory() / sanitize_path_component(layer_name)
    layer_cache_dir.mkdir(parents=True, exist_ok=True)
    return layer_cache_dir


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
    """Create a GeoSAM model spec for the selected model."""
    from geosam.runtime import create_model_spec as create_runtime_model_spec

    checkpoint_path = get_model_checkpoint_path(model_id)
    if not checkpoint_path.exists():
        logger.error("Model checkpoint is missing: %s", checkpoint_path)
        raise FileNotFoundError(checkpoint_path)
    return create_runtime_model_spec(model_id, checkpoint_path)


def infer_model_id_from_checkpoint_path(
    checkpoint_path: str | Path,
    *,
    fallback_model_id: str | None = None,
    allow_unknown: bool = False,
) -> str | None:
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
    if allow_unknown:
        logger.warning(
            "Could not infer a GeoSAM model id from checkpoint path: %s",
            checkpoint_path,
        )
        return None
    msg = (
        "Could not infer a GeoSAM model id from the checkpoint path. "
        "Please select a supported model explicitly."
    )
    logger.error(msg)
    raise ValueError(msg)


def create_model_spec_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_id: str | None = None,
    device: str | None = None,
) -> ModelSpec:
    """Create a model spec from an arbitrary checkpoint path."""
    from geosam.runtime import create_model_spec_from_checkpoint as create_runtime_spec

    return create_runtime_spec(
        checkpoint_path,
        model_id=model_id,
        device=device,
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
    geosam_requirement = (
        str(LOCAL_GEOSAM_REPOSITORY)
        if LOCAL_GEOSAM_REPOSITORY.exists()
        else "geosam"
    )
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        geosam_requirement,
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
    release_runtime_models(model_id=model_id)
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


def clear_cache() -> int:
    """Delete all cached files and return the number removed."""
    cache_dir = get_cache_directory()
    if not cache_dir.exists():
        return 0
    removed_count = 0
    for path in sorted(cache_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
            removed_count += 1
        elif path.is_dir():
            path.rmdir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return removed_count


def _flush_torch_memory() -> None:
    """Best-effort cleanup for Python and accelerator memory."""
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


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
    from geosam.runtime import resolve_feature_manifest_path as resolve_runtime_manifest

    return resolve_runtime_manifest(feature_dir)


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


def _rectangle_to_bbox(rectangle: QgsRectangle, *, crs_text: str):
    """Convert a QGIS rectangle into a GeoSAM bounding box."""
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
    crs_text = layer.crs().authid() or layer.crs().toWkt()
    if query.crs == QgsCoordinateReferenceSystem(crs_text):
        return query
    if query.crs is None:
        return query
    return query if str(query.crs) == crs_text else query.to_crs(crs_text)


def _query_is_far_from_chip_edge(query, *, chip_bounds, chip_grid) -> bool:
    """Return whether a query center is inside the safe inner chip region."""
    from geosam.query import query_bounds, query_center

    if not chip_bounds.contains(query_bounds(query)):
        return False

    center_x, center_y = query_center(query)
    height, width = chip_grid.shape
    pixel_size_x = abs(float(chip_grid.transform.a))
    pixel_size_y = abs(float(chip_grid.transform.e))
    margin_x_pixels = min(max(ONLINE_QUERY_REFRESH_MARGIN_PIXELS, 1), max(width // 2, 1))
    margin_y_pixels = min(max(ONLINE_QUERY_REFRESH_MARGIN_PIXELS, 1), max(height // 2, 1))
    margin_x = margin_x_pixels * pixel_size_x
    margin_y = margin_y_pixels * pixel_size_y

    return (
        chip_bounds.left + margin_x <= center_x <= chip_bounds.right - margin_x
        and chip_bounds.bottom + margin_y <= center_y <= chip_bounds.top - margin_y
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
        "_".join(
            [
                f"{chip_bounds.left:.3f}",
                f"{chip_bounds.bottom:.3f}",
                f"{chip_bounds.right:.3f}",
                f"{chip_bounds.top:.3f}",
            ]
        )
    )
    entry_dir = cache_dir / chip_key
    entry_dir.mkdir(parents=True, exist_ok=True)
    return entry_dir / "encoded.pt", entry_dir / "metadata.json"


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
    if query_cache is None or query_cache.encoded is None or query_cache.chip_grid is None:
        return
    if query_cache.chip_bounds is None:
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
        "crs": query_cache.chip_grid.crs.to_string(),
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
    model_definition = get_model_definition(model_id)
    supports_feature_reuse = bool(model_definition.supports_feature_reuse)

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
            if cached_source_path is None or not cached_source_path.exists():
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

    if supports_feature_reuse and cache is not None and cache.query_cache is None:
        persistent_cache_hit = _load_persistent_query_cache(layer, model_id, query)
        if persistent_cache_hit is not None:
            cache.layer_id = layer.id()
            cache.model_id = model_id
            cache.source_candidate = persistent_cache_hit[0]
            cache.engine = None
            cache.query_cache = persistent_cache_hit[1]
            return None

    source_candidates = [
        source_path
        for source_candidate in _raster_layer_source_candidates(layer)
        if (source_path := _normalize_local_raster_source(source_candidate)) is not None
    ]
    if len(source_candidates) == 0:
        source_candidates = [_export_online_raster_source(layer, query, model_id=model_id)]

    return PreparedRealtimeRasterQuery(
        model_id=model_id,
        layer_id=layer.id(),
        layer_name=layer.name(),
        source_fingerprint=_layer_source_fingerprint(layer),
        cache_directory=_realtime_query_cache_directory(layer, model_id),
        source_candidates=tuple(source_candidates),
        supports_feature_reuse=supports_feature_reuse,
    )


def run_prepared_realtime_raster_query(
    prepared_query: PreparedRealtimeRasterQuery,
    query,
    *,
    progress_callback: RealtimeQueryProgressCallback | None = None,
    is_canceled: RealtimeQueryCancelCallback | None = None,
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
    from geosam.datasets import RasterDataset
    from geosam.engines import (
        OnlineQueryCache,
        OnlineQueryEngine,
        _prediction_to_result,
        _prompt_prediction_kwargs,
        _require_query_crs,
    )
    from geosam.query import query_bounds, query_center, window_from_center

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

    for source_candidate in prepared_query.source_candidates:
        try:
            _ensure_not_canceled()
            _report_progress("Opening raster source", 5.0)
            dataset = RasterDataset(source_candidate)
            engine = OnlineQueryEngine(dataset, model_spec)

            if not prepared_query.supports_feature_reuse:
                _report_progress("Running realtime query", 35.0)
                result = engine.query(query)
                _report_progress("Finished realtime query", 100.0)
                return PreparedRealtimeRasterQueryResult(
                    source_path=source_candidate,
                    result=result,
                    query_cache=None,
                )

            projected_query = (
                query if query.crs == dataset.crs else query.to_crs(dataset.crs)
            )
            chip_bounds = window_from_center(
                query_center(projected_query),
                model_spec.resolved_imgsz,
                grid=dataset.grid,
            )
            sample = dataset[chip_bounds]
            chip_grid = sample.grid

            _ensure_not_canceled()
            _report_progress("Encoding image", 30.0)
            encoded = engine.adapter.encode_image(sample.to_model_image())

            _ensure_not_canceled()
            _report_progress("Running prompt query", 75.0)
            prediction = engine.adapter.predict_features(
                encoded,
                multimask_output=False,
                **_prompt_prediction_kwargs(query, chip_grid),
            )

            query_cache = OnlineQueryCache(
                source_path=sample.source_path,
                chip_bounds=chip_bounds,
                chip_grid=chip_grid,
                encoded=encoded,
            )

            _ensure_not_canceled()
            _report_progress("Saving encoded cache", 90.0)
            _save_persistent_query_cache_entry(
                cache_directory=prepared_query.cache_directory,
                layer_name=prepared_query.layer_name,
                source_fingerprint=prepared_query.source_fingerprint,
                model_id=prepared_query.model_id,
                source_path=sample.source_path,
                query_cache=query_cache,
            )

            result = _prediction_to_result(
                prediction,
                sample_grid=chip_grid,
                query_bounds_value=query_bounds(projected_query),
                source_path=sample.source_path,
                chip_id=None,
                model_type=model_spec.model_type,
            )
            _report_progress("Finished realtime query", 100.0)
            return PreparedRealtimeRasterQueryResult(
                source_path=sample.source_path,
                result=result,
                query_cache=query_cache,
            )
        except _RealtimeQueryCanceledError:
            raise
        except Exception as exc:
            errors.append(f"{source_candidate}: {exc}")
            continue

    msg = "Failed to encode a realtime raster chip for the current prompt."
    logger.error("%s Details: %s", msg, errors)
    raise ValueError("\n".join([msg, *errors]))


def _load_persistent_query_cache(
    layer: QgsRasterLayer,
    model_id: str,
    query,
) -> tuple[str, Any] | None:
    """Load a reusable realtime-query cache entry when one matches the query."""
    settings = load_plugin_settings()
    if not settings.get("cache_enabled", True):
        return None

    from geosam.datasets import GeoGrid
    from geosam.engines import OnlineQueryCache
    from geosam.models import EncodedImageFeatures
    from geosam.query import BoundingBox, query_center
    from rasterio import Affine

    cache_dir = _realtime_query_cache_directory(layer, model_id)
    if not cache_dir.exists():
        return None

    projected_query = _query_in_layer_crs(layer, query)
    center_x, center_y = query_center(projected_query)
    expected_fingerprint = _layer_source_fingerprint(layer)

    best_match: tuple[float, str, Any] | None = None
    for metadata_path in sorted(cache_dir.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read query cache metadata %s: %s", metadata_path, exc)
            continue

        if metadata.get("source_fingerprint") != expected_fingerprint:
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
            tuple(metadata["shape"]),
            metadata["crs"],
        )
        if not _query_is_far_from_chip_edge(
            projected_query,
            chip_bounds=chip_bounds,
            chip_grid=chip_grid,
        ):
            continue

        encoded_path = Path(metadata["feature_path"])
        if not encoded_path.exists():
            continue

        encoded = EncodedImageFeatures.load(encoded_path, map_location="cpu")
        distance = (chip_bounds.center[0] - center_x) ** 2 + (
            chip_bounds.center[1] - center_y
        ) ** 2
        query_cache = OnlineQueryCache(
            source_path=source_path,
            chip_bounds=chip_bounds,
            chip_grid=chip_grid,
            encoded=encoded,
        )
        best_match = (distance, source_path, query_cache) if best_match is None else min(
            best_match,
            (distance, source_path, query_cache),
            key=lambda item: item[0],
        )

    if best_match is None:
        return None
    return best_match[1], best_match[2]


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


def _chip_extent_for_online_layer(layer: QgsRasterLayer, query, *, chip_size: tuple[int, int]) -> QgsRectangle:
    """Build a chip extent in layer CRS for an online raster query."""
    from geosam.query import query_center

    projected_query = _query_in_layer_crs(layer, query)
    center_x, center_y = query_center(projected_query)
    pixel_size_x, pixel_size_y = _resolve_pixel_size(layer)
    chip_height, chip_width = chip_size
    half_width = pixel_size_x * chip_width / 2.0
    half_height = pixel_size_y * chip_height / 2.0
    extent = QgsRectangle(
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )
    if extent.isNull() or extent.isEmpty():
        return layer.extent()
    return extent.intersect(layer.extent())


def _export_raster_extent(
    layer: QgsRasterLayer,
    *,
    extent: QgsRectangle,
    width: int,
    height: int,
    destination_path: Path,
) -> Path:
    """Export a raster extent to a GeoTIFF using QGIS raster IO."""
    provider = layer.dataProvider()
    if provider is None:
        msg = "The selected raster layer does not provide a raster data provider."
        logger.error(msg)
        raise ValueError(msg)

    provider_clone = provider.clone()
    if provider_clone is None:
        msg = "Failed to clone the raster provider for cache export."
        logger.error(msg)
        raise RuntimeError(msg)

    pipe = QgsRasterPipe()
    if not pipe.set(provider_clone):
        msg = "Failed to initialize a raster pipe for cache export."
        logger.error(msg)
        raise RuntimeError(msg)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    writer = QgsRasterFileWriter(str(destination_path))
    writer.setOutputFormat("GTiff")
    result = writer.writeRaster(
        pipe,
        int(width),
        int(height),
        extent,
        layer.crs(),
        QgsProject.instance().transformContext(),
    )
    if result != Qgis.RasterFileWriterResult.NoError:
        msg = f"Failed to export online raster cache. QGIS writer result={result!r}"
        logger.error(msg)
        raise RuntimeError(msg)
    return destination_path


def _export_online_raster_source(layer: QgsRasterLayer, query, *, model_id: str) -> str:
    """Export the current online-layer chip into the plugin cache."""
    chip_size = create_model_spec(model_id).resolved_imgsz
    chip_extent = _chip_extent_for_online_layer(layer, query, chip_size=chip_size)
    source_fingerprint = hashlib.sha256(
        _layer_source_fingerprint(layer).encode("utf-8")
    ).hexdigest()[:16]
    file_stem = sanitize_path_component(
        "_".join(
            [
                model_id,
                source_fingerprint,
                f"{chip_extent.xMinimum():.3f}",
                f"{chip_extent.yMinimum():.3f}",
                f"{chip_extent.xMaximum():.3f}",
                f"{chip_extent.yMaximum():.3f}",
            ]
        )
    )
    destination_path = _online_layer_raster_cache_directory(layer) / f"{file_stem}.tif"
    if destination_path.exists():
        return str(destination_path)

    return str(
        _export_raster_extent(
            layer,
            extent=chip_extent,
            width=chip_size[1],
            height=chip_size[0],
            destination_path=destination_path,
        )
    )


def query_feature_source(
    feature_dir: str | Path,
    model_id: str | None,
    query: BoundingBox | Points | PromptSet,
) -> QueryResult:
    """Run a query against a GeoSAM feature folder."""
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
    from geosam.engines import OnlineQueryCache

    model_definition = get_model_definition(model_id)
    supports_feature_reuse = bool(model_definition.supports_feature_reuse)

    if cache is not None:
        cache_mismatch = cache.layer_id != layer.id() or cache.model_id != model_id
        if cache_mismatch:
            cache.clear()
        elif (
            supports_feature_reuse
            and cache.query_cache is not None
            and not _query_is_far_from_chip_edge(
                _query_in_layer_crs(layer, query),
                chip_bounds=cache.query_cache.chip_bounds,
                chip_grid=cache.query_cache.chip_grid,
            )
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
            MODEL_SESSIONS.release_online_engines(
                model_id=model_id,
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
            return engine.query(query, cache=cache.query_cache)

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
                MODEL_SESSIONS.release_online_engines(
                    model_id=model_id,
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
                    if persistent_cache_hit is not None and source_candidate == persistent_cache_hit[0]:
                        cache.query_cache = persistent_cache_hit[1]
                    else:
                        cache.query_cache = OnlineQueryCache()

            result = engine.query(
                query,
                cache=cache.query_cache if (cache is not None and supports_feature_reuse) else None,
            )
            if supports_feature_reuse and cache is not None and cache.query_cache is not None:
                _save_persistent_query_cache(
                    layer=layer,
                    model_id=model_id,
                    source_path=source_candidate,
                    query_cache=cache.query_cache,
                )
            return result
        except Exception as exc:
            errors.append(f"{source_candidate}: {exc}")
            if cache is not None and cache.source_candidate == source_candidate:
                cache.engine = None
            continue

    try:
        exported_source = _export_online_raster_source(layer, query, model_id=model_id)
        MODEL_SESSIONS.release_online_engines(
            model_id=model_id,
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
            cache=cache.query_cache if (cache is not None and supports_feature_reuse) else None,
        )
        if supports_feature_reuse and cache is not None and cache.query_cache is not None:
            _save_persistent_query_cache(
                layer=layer,
                model_id=model_id,
                source_path=exported_source,
                query_cache=cache.query_cache,
            )
        return result
    except Exception as exc:
        errors.append(f"online-export: {exc}")

    msg = (
        "Failed to open the selected raster layer with GeoSAM. "
        "The plugin also failed to export a local cache for the current online view."
    )
    MessageTool.MessageLog("\n".join([msg, *errors]), level="warning")
    logger.error("%s Details: %s", msg, errors)
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
