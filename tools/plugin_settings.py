"""Plugin settings, dependency, and cache helpers for Geo-SAM."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Literal

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices

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
DEPENDENCY_DISTRIBUTIONS: dict[str, str] = {
    "geosam": "geosam",
    "torch": "torch",
    "ultralytics": "ultralytics",
    "rasterio": "rasterio",
    "geopandas": "geopandas",
    "pyarrow": "pyarrow",
}
PerformanceMode = Literal["balanced", "fastest", "low_memory"]
PreviewRenderMode = Literal["light", "exact"]
PERFORMANCE_MODE_VALUES: tuple[PerformanceMode, ...] = (
    "balanced",
    "fastest",
    "low_memory",
)
PREVIEW_RENDER_MODE_VALUES: tuple[PreviewRenderMode, ...] = ("light", "exact")


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Parameters
    ----------
    path : Path
        JSON file path.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object, or an empty dictionary when the file is missing.

    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_plugin_settings() -> dict[str, Any]:
    """Load effective plugin settings.

    Returns
    -------
    dict[str, Any]
        Effective settings after merging defaults with user overrides.

    """
    settings = _default_plugin_settings()
    settings.update(_load_json(SETTINGS_USER_PATH))
    if settings.get("performance_mode") not in PERFORMANCE_MODE_VALUES:
        settings["performance_mode"] = "balanced"
    if settings.get("preview_render_mode") not in PREVIEW_RENDER_MODE_VALUES:
        settings["preview_render_mode"] = "light"
    return settings


def _default_plugin_settings() -> dict[str, Any]:
    """Return the effective default plugin settings.

    Returns
    -------
    dict[str, Any]
        Default settings resolved from the shipped configuration file.

    """
    settings = _load_json(SETTINGS_DEFAULT_PATH)
    settings.setdefault("model_store_dir", str(DEFAULT_MODEL_DIR))
    settings.setdefault("cache_enabled", True)
    settings.setdefault("cache_dir", str(DEFAULT_CACHE_DIR))
    settings.setdefault("cache_max_size_mb", 2048)
    settings.setdefault("clear_cache_on_plugin_close", True)
    settings.setdefault("selected_model_id", "")
    settings.setdefault("show_boundary", True)
    settings.setdefault("default_minimum_pixels", 0)
    settings.setdefault("performance_mode", "balanced")
    settings.setdefault("preview_render_mode", "light")
    return settings


def save_plugin_settings(updates: dict[str, Any]) -> dict[str, Any]:
    """Persist plugin settings.

    Parameters
    ----------
    updates : dict[str, Any]
        Setting values to merge into the current configuration.

    Returns
    -------
    dict[str, Any]
        Effective settings after the update is applied.

    """
    settings = load_plugin_settings()
    settings.update(updates)
    defaults = _default_plugin_settings()
    user_settings = {
        key: settings[key]
        for key in defaults
        if settings.get(key) != defaults[key]
    }
    SETTINGS_USER_PATH.write_text(
        json.dumps(user_settings, indent=4),
        encoding="utf-8",
    )
    return settings


def get_model_directory() -> Path:
    """Return the local model directory.

    Returns
    -------
    Path
        Existing model directory path.

    """
    settings = load_plugin_settings()
    model_dir = Path(settings["model_store_dir"]).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_cache_directory() -> Path:
    """Return the local cache directory.

    Returns
    -------
    Path
        Existing cache directory path.

    """
    settings = load_plugin_settings()
    cache_dir = Path(settings["cache_dir"]).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def dependency_status() -> dict[str, bool]:
    """Return installed-package availability for required dependencies.

    Returns
    -------
    dict[str, bool]
        Mapping from module name to installation status.

    Notes
    -----
    This check intentionally uses distribution metadata instead of
    ``importlib.util.find_spec`` so the result reflects the current environment
    after install or uninstall operations in the same QGIS session.

    """
    status: dict[str, bool] = {}
    for module_name, distribution_name in DEPENDENCY_DISTRIBUTIONS.items():
        try:
            importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            status[module_name] = False
        else:
            status[module_name] = True
    return status


def _is_relative_to(path: Path, base_path: Path) -> bool:
    """Return whether ``path`` is located inside ``base_path``.

    Parameters
    ----------
    path : Path
        Candidate path.
    base_path : Path
        Expected ancestor path.

    Returns
    -------
    bool
        ``True`` when ``path`` is inside ``base_path``.

    """
    try:
        path.relative_to(base_path)
    except ValueError:
        return False
    return True


def _iter_python_interpreter_candidates() -> list[Path]:
    """Return candidate Python executables for the active QGIS environment.

    Returns
    -------
    list[Path]
        Ordered executable candidates, with duplicates removed.

    """
    prefix_path = Path(sys.prefix).expanduser().resolve()
    candidate_paths: list[Path] = []

    for relative_path in (
        "bin/python",
        "bin/python3",
        "bin/python3.11",
        "python",
        "python3",
        "python3.11",
    ):
        candidate_paths.append(prefix_path / relative_path)

    for raw_path in sys.path:
        if not raw_path:
            continue
        try:
            resolved_path = Path(raw_path).expanduser().resolve()
        except OSError:
            continue
        if not _is_relative_to(resolved_path, prefix_path):
            continue

        for parent_path in (resolved_path, *resolved_path.parents):
            if parent_path == prefix_path.parent:
                break
            if parent_path.name not in {"bin", "Scripts"}:
                continue
            for child_path in sorted(parent_path.glob("python*")):
                candidate_paths.append(child_path)

    unique_candidates: list[Path] = []
    seen_paths: set[Path] = set()
    for candidate_path in candidate_paths:
        try:
            resolved_candidate = candidate_path.expanduser().resolve()
        except OSError:
            continue
        if resolved_candidate in seen_paths:
            continue
        seen_paths.add(resolved_candidate)
        unique_candidates.append(resolved_candidate)
    return unique_candidates


def resolve_python_interpreter() -> Path:
    """Resolve the Python interpreter for the active QGIS runtime environment.

    Returns
    -------
    Path
        Resolved Python interpreter path that belongs to ``sys.prefix``.

    Raises
    ------
    RuntimeError
        Raised when no suitable Python interpreter can be found.

    """
    prefix_path = Path(sys.prefix).expanduser().resolve()
    executable_path = Path(sys.executable).expanduser().resolve()
    if executable_path.is_file() and executable_path.name.startswith("python"):
        return executable_path

    for candidate_path in _iter_python_interpreter_candidates():
        if not candidate_path.is_file():
            continue
        if not os.access(candidate_path, os.X_OK):
            continue
        if not candidate_path.name.startswith("python"):
            continue
        if not _is_relative_to(candidate_path, prefix_path):
            continue
        return candidate_path

    msg = (
        "Could not resolve a Python interpreter from the current QGIS runtime. "
        f"sys.executable={sys.executable!r}, sys.prefix={sys.prefix!r}"
    )
    logger.error(msg)
    raise RuntimeError(msg)


def get_dependency_install_command() -> list[str]:
    """Build the pip command used to install plugin dependencies.

    Returns
    -------
    list[str]
        Command arguments suitable for ``subprocess`` execution.

    Raises
    ------
    RuntimeError
        Raised when the current QGIS runtime Python interpreter cannot be
        resolved.

    """
    geosam_requirement = (
        str(LOCAL_GEOSAM_REPOSITORY) if LOCAL_GEOSAM_REPOSITORY.exists() else "geosam"
    )
    python_interpreter = resolve_python_interpreter()
    return [
        str(python_interpreter),
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


def install_dependencies(
    log_callback: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Install GeoSAM plugin dependencies with pip.

    Parameters
    ----------
    log_callback : Callable[[str], None] | None, optional
        Callback that receives incremental install log lines.

    Returns
    -------
    tuple[bool, str]
        Installation success flag and combined command output.

    """
    try:
        command = get_dependency_install_command()
    except RuntimeError as exc:
        logger.error("Dependency installation aborted: %s", exc)
        return False, str(exc)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []

    if process.stdout is not None:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            output_lines.append(line)
            if log_callback is not None:
                log_callback(line)

    return_code = process.wait()
    output = "\n".join(line for line in output_lines if line.strip())
    return return_code == 0, output


def open_url(url: str) -> None:
    """Open a URL in the desktop browser.

    Parameters
    ----------
    url : str
        Target URL.

    """
    QDesktopServices.openUrl(QUrl(url))


def open_path(path: Path) -> None:
    """Open a local path in the desktop file manager.

    Parameters
    ----------
    path : Path
        Target local path.

    """
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.expanduser().resolve())))


def get_cache_size_bytes(path: Path | None = None) -> int:
    """Return the current cache size in bytes.

    Parameters
    ----------
    path : Path | None, optional
        Cache directory to inspect. When omitted, the configured cache
        directory is used.

    Returns
    -------
    int
        Total cache size in bytes.

    """
    cache_dir = get_cache_directory() if path is None else path
    if not cache_dir.exists():
        return 0
    return sum(
        file_path.stat().st_size
        for file_path in cache_dir.rglob("*")
        if file_path.is_file()
    )


def cleanup_cache() -> int:
    """Trim cache files until the configured maximum size is respected.

    Returns
    -------
    int
        Number of removed cache files.

    """
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
    """Delete all cached files and return the number removed.

    Returns
    -------
    int
        Number of removed cache files.

    """
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
