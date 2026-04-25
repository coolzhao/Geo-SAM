"""Plugin settings, dependency, and cache helpers for Geo-SAM."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, TypedDict

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
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "rasterio": "rasterio",
    "geopandas": "geopandas",
    "pyarrow": "pyarrow",
    "geosam": "geosam",
}
DEPENDENCY_INSTALL_REQUIREMENTS: dict[str, str] = {
    "geosam": "geosam",
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "rasterio": "rasterio",
    "geopandas": "geopandas",
    "pyarrow": "pyarrow",
}
PerformanceMode = Literal["balanced", "fastest", "low_memory"]
PreviewRenderMode = Literal["pixel_level", "simplified"]
PERFORMANCE_MODE_VALUES: tuple[PerformanceMode, ...] = (
    "balanced",
    "fastest",
    "low_memory",
)
PREVIEW_RENDER_MODE_VALUES: tuple[PreviewRenderMode, ...] = (
    "pixel_level",
    "simplified",
)
PROJ_DATA_ENV_KEYS = ("PROJ_DATA", "PROJ_LIB")


class DependencyStatusRow(TypedDict):
    """Dependency status information shown in the settings dialog.

    Attributes
    ----------
    package : str
        Import or package name shown to users.
    distribution : str
        Installed Python distribution name.
    installed : bool
        Whether the distribution can be found in the active environment.
    version : str
        Installed distribution version, or an empty string when missing.

    """

    package: str
    distribution: str
    installed: bool
    version: str


def initialize_rasterio_proj_data() -> None:
    """Point rasterio at its bundled PROJ database when available.

    Rasterio wheels can bundle a newer PROJ database than the active QGIS
    process exposes through ``PROJ_LIB``. Importing rasterio while ``PROJ_LIB``
    points at an older database can make valid EPSG codes fail to resolve.
    This helper resolves rasterio's bundled ``proj_data`` directory, imports
    rasterio with that path during initialization when needed, and explicitly
    updates rasterio's PROJ search path. The explicit search-path update also
    fixes sessions where rasterio was imported before this helper ran.

    Returns
    -------
    None
        The function mutates only import state and temporarily mutates
        ``os.environ``.

    Notes
    -----
    The helper is intentionally a no-op when rasterio is unavailable. Call it
    immediately before rasterio operations in lazy rasterio code paths.

    """
    rasterio_module = sys.modules.get("rasterio")
    if rasterio_module is not None:
        rasterio_file = getattr(rasterio_module, "__file__", None)
        if rasterio_file is None:
            return
        rasterio_package_path = Path(rasterio_file).resolve().parent
    else:
        rasterio_spec = importlib.util.find_spec("rasterio")
        if rasterio_spec is None or rasterio_spec.submodule_search_locations is None:
            return
        rasterio_package_path = Path(rasterio_spec.submodule_search_locations[0])

    rasterio_proj_data_path = rasterio_package_path / "proj_data"
    if not (rasterio_proj_data_path / "proj.db").is_file():
        return

    original_environment = {
        env_key: os.environ.get(env_key) for env_key in PROJ_DATA_ENV_KEYS
    }
    rasterio_proj_data_text = str(rasterio_proj_data_path)
    try:
        for env_key in PROJ_DATA_ENV_KEYS:
            os.environ[env_key] = rasterio_proj_data_text
        rasterio_module = importlib.import_module("rasterio")
        rasterio_env_module = importlib.import_module("rasterio.env")
        set_proj_data_search_path = getattr(
            rasterio_env_module,
            "set_proj_data_search_path",
            None,
        )
        if set_proj_data_search_path is None:
            logger.warning(
                "Rasterio does not expose set_proj_data_search_path; "
                "cannot override PROJ data path."
            )
            return
        set_proj_data_search_path(rasterio_proj_data_text)
        logger.debug(
            "Initialized rasterio PROJ data path for %s from %s.",
            getattr(rasterio_module, "__file__", "rasterio"),
            rasterio_proj_data_path,
        )
    finally:
        for env_key, env_value in original_environment.items():
            if env_value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = env_value


def _resolve_rasterio_proj_data_path() -> Path | None:
    """Resolve rasterio's bundled PROJ data directory.

    Returns
    -------
    Path | None
        Path to rasterio's bundled ``proj_data`` directory, or ``None`` when
        rasterio is unavailable or does not bundle ``proj.db``.

    """
    rasterio_module = sys.modules.get("rasterio")
    if rasterio_module is not None:
        rasterio_file = getattr(rasterio_module, "__file__", None)
        if rasterio_file is None:
            return None
        rasterio_package_path = Path(rasterio_file).resolve().parent
    else:
        rasterio_spec = importlib.util.find_spec("rasterio")
        if rasterio_spec is None or rasterio_spec.submodule_search_locations is None:
            return None
        rasterio_package_path = Path(rasterio_spec.submodule_search_locations[0])

    rasterio_proj_data_path = rasterio_package_path / "proj_data"
    if not (rasterio_proj_data_path / "proj.db").is_file():
        return None
    return rasterio_proj_data_path


@contextmanager
def rasterio_proj_data_environment() -> Iterator[None]:
    """Temporarily expose rasterio's bundled PROJ database to rasterio.

    Yields
    ------
    None
        Control returns to the caller while ``PROJ_DATA`` and ``PROJ_LIB`` point
        at rasterio's matching ``proj_data`` directory.

    Notes
    -----
    Rasterio's ``Env`` wrapper can re-read ``os.environ`` during operations such
    as :func:`rasterio.open`. Keeping the environment override active for the
    full operation prevents it from falling back to an incompatible QGIS or
    conda ``proj.db``.

    """
    rasterio_proj_data_path = _resolve_rasterio_proj_data_path()
    if rasterio_proj_data_path is None:
        yield
        return

    original_environment = {
        env_key: os.environ.get(env_key) for env_key in PROJ_DATA_ENV_KEYS
    }
    rasterio_proj_data_text = str(rasterio_proj_data_path)
    try:
        for env_key in PROJ_DATA_ENV_KEYS:
            os.environ[env_key] = rasterio_proj_data_text
        initialize_rasterio_proj_data()
        yield
    finally:
        for env_key, env_value in original_environment.items():
            if env_value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = env_value


def _normalize_preview_render_mode(value: Any) -> PreviewRenderMode:
    """Normalize a configured vectorization mode.

    Parameters
    ----------
    value : Any
        Raw persisted value, including legacy preview-mode variants.

    Returns
    -------
    PreviewRenderMode
        Normalized vectorization mode.

    Notes
    -----
    Legacy values are kept compatible so existing user settings continue to
    work after the UI label and option names change.

    """
    normalized_value = str(value or "simplified").strip().lower().replace("-", "_")
    legacy_value_mapping: dict[str, PreviewRenderMode] = {
        "exact": "pixel_level",
        "light": "simplified",
        "pixel_level": "pixel_level",
        "simplified": "simplified",
    }
    return legacy_value_mapping.get(normalized_value, "simplified")


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Parameters
    ----------
    path : Path
        JSON file path.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object, or an empty dictionary when the user settings file
        is missing.

    """
    if not path.exists():
        if path == SETTINGS_USER_PATH:
            path.write_text("{}\n", encoding="utf-8")
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
    settings["preview_render_mode"] = _normalize_preview_render_mode(
        settings.get("preview_render_mode")
    )
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
    settings["preview_render_mode"] = _normalize_preview_render_mode(
        settings.get("preview_render_mode")
    )
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
    settings["preview_render_mode"] = _normalize_preview_render_mode(
        settings.get("preview_render_mode")
    )
    defaults = _default_plugin_settings()
    user_settings = {
        key: settings[key] for key in defaults if settings.get(key) != defaults[key]
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


def dependency_status_rows() -> list[DependencyStatusRow]:
    """Return dependency status rows with installed versions.

    Returns
    -------
    list[DependencyStatusRow]
        Ordered dependency rows for the settings dialog table.

    """
    rows: list[DependencyStatusRow] = []
    for module_name, distribution_name in DEPENDENCY_DISTRIBUTIONS.items():
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            rows.append(
                {
                    "package": module_name,
                    "distribution": distribution_name,
                    "installed": False,
                    "version": "",
                }
            )
        else:
            rows.append(
                {
                    "package": module_name,
                    "distribution": distribution_name,
                    "installed": True,
                    "version": distribution.version,
                }
            )
    return rows


def _dependency_install_requirement(module_name: str) -> str:
    """Return the pip requirement used to install a dependency.

    Parameters
    ----------
    module_name : str
        Dependency module name.

    Returns
    -------
    str
        Pip requirement name or path.

    """
    if module_name == "geosam" and LOCAL_GEOSAM_REPOSITORY.exists():
        return str(LOCAL_GEOSAM_REPOSITORY)
    return DEPENDENCY_INSTALL_REQUIREMENTS[module_name]


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
    executable_path = Path(sys.executable).expanduser().resolve()
    candidate_paths: list[Path] = []

    for relative_path in (
        "bin/python",
        "bin/python3",
        "bin/python3.11",
        "python",
        "python3",
        "python3.11",
        "python.exe",
        "python3.exe",
    ):
        candidate_paths.append(prefix_path / relative_path)

    candidate_paths.extend(_iter_qgis_app_python_candidates(executable_path))

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


def _iter_qgis_app_python_candidates(executable_path: Path) -> list[Path]:
    """Return Python executables bundled beside Windows QGIS launchers.

    Parameters
    ----------
    executable_path : Path
        Active QGIS process executable path.

    Returns
    -------
    list[Path]
        Candidate Python executable paths under the detected QGIS root.

    Notes
    -----
    Windows QGIS exposes ``sys.executable`` as launchers such as
    ``qgis-ltr-bin.exe``. The matching interpreter lives under
    ``apps/Python*/python.exe`` within the same QGIS installation root.

    """
    if executable_path.name.lower().startswith("python"):
        return []

    candidate_roots: list[Path] = []
    for parent_path in executable_path.parents:
        if parent_path.name.lower() != "bin":
            continue
        candidate_roots.append(parent_path.parent)
        break

    candidate_paths: list[Path] = []
    for root_path in candidate_roots:
        python_apps_path = root_path / "apps"
        for python_dir in sorted(python_apps_path.glob("Python*")):
            candidate_paths.extend(
                [
                    python_dir / "python.exe",
                    python_dir / "python3.exe",
                    python_dir / "python",
                    python_dir / "python3",
                ]
            )
    return candidate_paths


def _is_python_interpreter_candidate(candidate_path: Path) -> bool:
    """Return whether a path is an executable Python interpreter candidate.

    Parameters
    ----------
    candidate_path : Path
        Candidate executable path.

    Returns
    -------
    bool
        ``True`` when the path looks runnable as Python.

    """
    candidate_name = candidate_path.name.lower()
    if not candidate_name.startswith("python"):
        return False
    if not candidate_path.is_file():
        return False
    if os.name == "nt":
        return True
    return os.access(candidate_path, os.X_OK)


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
    executable_path = Path(sys.executable).expanduser().resolve()
    if _is_python_interpreter_candidate(executable_path):
        return executable_path

    for candidate_path in _iter_python_interpreter_candidates():
        if not _is_python_interpreter_candidate(candidate_path):
            continue
        return candidate_path

    msg = (
        "Could not resolve a Python interpreter from the current QGIS runtime. "
        f"sys.executable={sys.executable!r}, sys.prefix={sys.prefix!r}"
    )
    logger.error(msg)
    raise RuntimeError(msg)


def get_dependency_install_command(
    module_names: Iterable[str] | None = None,
) -> list[str]:
    """Build the pip command used to install plugin dependencies.

    Parameters
    ----------
    module_names : Iterable[str] | None, optional
        Dependency package names to install. When omitted, all known plugin
        dependencies are installed.

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
    if module_names is None:
        selected_module_names = list(DEPENDENCY_INSTALL_REQUIREMENTS)
    else:
        selected_module_names = [
            module_name
            for module_name in module_names
            if module_name in DEPENDENCY_INSTALL_REQUIREMENTS
        ]
    if not selected_module_names:
        msg = "No known dependencies were selected for installation."
        logger.error(msg)
        raise RuntimeError(msg)

    python_interpreter = resolve_python_interpreter()
    return [
        str(python_interpreter),
        "-m",
        "pip",
        "install",
        *[
            _dependency_install_requirement(module_name)
            for module_name in selected_module_names
        ],
    ]


def format_dependency_install_command(command: Iterable[str]) -> str:
    """Return a readable dependency installation command.

    Parameters
    ----------
    command : Iterable[str]
        Command arguments produced by :func:`get_dependency_install_command`.

    Returns
    -------
    str
        Shell-readable command text for logs.

    """
    return subprocess.list2cmdline(list(command))


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
