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
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, TypedDict

from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtGui import QDesktopServices

from .dependency_path import register_plugin_managed_dependency_path

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
    "shapely": "shapely",
    "pyproj": "pyproj",
    "geosam": "geosam",
}
DEPENDENCY_INSTALL_REQUIREMENTS: dict[str, str] = {
    "geosam": "geosam",
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics>=8.4.33",
    "rasterio": "rasterio>=1.4.3",
    "geopandas": "geopandas>=1.0.1",
    "shapely": "shapely>=2.0.7",
    "pyproj": "pyproj>=3.6.1",
}
PLUGIN_RUNTIME_DEPENDENCY_NAMES: tuple[str, ...] = tuple(DEPENDENCY_DISTRIBUTIONS)
RUNTIME_PROVIDED_DEPENDENCY_NAMES: tuple[str, ...] = (
    "rasterio",
    "geopandas",
    "shapely",
    "pyproj",
)
SEGMENTATION_RUNTIME_DEPENDENCY_NAMES: tuple[str, ...] = (
    "geosam",
    "rasterio",
    "geopandas",
    "shapely",
    "pyproj",
)
QGIS_RUNTIME_CONSTRAINT_DISTRIBUTIONS: tuple[str, ...] = (
    "numpy",
    "scipy",
    "pandas",
    "geopandas",
    "rasterio",
    "shapely",
    "pyproj",
    "pyogrio",
    "python-dateutil",
    "networkx",
    "requests",
    "pyyaml",
    "packaging",
)
PIP_RESOLVER_CONSTRAINT_REQUIREMENTS: tuple[str, ...] = (
    "contourpy>=1.2.0",
    "matplotlib>=3.8.0",
)
TORCH_RESOLVER_CONSTRAINT_REQUIREMENTS_PY39: tuple[str, ...] = (
    "torch==2.8.0",
    "torchvision==0.23.0",
)
TORCH_RESOLVER_CONSTRAINT_REQUIREMENTS_PY310_PLUS: tuple[str, ...] = (
    "torch==2.11.0",
    "torchvision==0.26.0",
)
PIP_ONLY_BINARY_DISTRIBUTIONS: tuple[str, ...] = (
    "contourpy",
    "kiwisolver",
    "matplotlib",
    "numpy",
    "opencv-python",
    "pillow",
    "torch",
    "torchvision",
)
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
DependencyInstallCommand = list[str]
DependencyInstallPlan = list[DependencyInstallCommand]
DependencyState = Literal[
    "installed_qgis_runtime",
    "installed_plugin_managed",
    "missing_installable",
    "missing_required",
]


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
    state : DependencyState
        Resolved dependency source and availability state.
    installable : bool
        Whether the plugin installer should attempt to install this dependency.
    source : str
        Short source or install-provider note for the settings dialog.

    """

    package: str
    distribution: str
    installed: bool
    version: str
    state: DependencyState
    installable: bool
    source: str


def dependency_state(module_name: str, installed: bool) -> DependencyState:
    """Return the resolved dependency state for a module.

    Parameters
    ----------
    module_name : str
        Dependency module name.
    installed : bool
        Whether the dependency is installed in the active environment.

    Returns
    -------
    DependencyState
        Normalized dependency state used by the settings dialog and runtime
        preflight checks.

    """
    if installed:
        if module_name in RUNTIME_PROVIDED_DEPENDENCY_NAMES:
            return "installed_qgis_runtime"
        return "installed_plugin_managed"
    if module_name in DEPENDENCY_INSTALL_REQUIREMENTS:
        return "missing_installable"
    return "missing_required"


def dependency_source_text(module_name: str, state: DependencyState) -> str:
    """Return the settings source label for a dependency state.

    Parameters
    ----------
    module_name : str
        Dependency module name.
    state : DependencyState
        Resolved dependency state.

    Returns
    -------
    str
        Source text shown in the settings dependency table.

    """
    if state == "installed_plugin_managed":
        return "Geo-SAM managed"
    if module_name in RUNTIME_PROVIDED_DEPENDENCY_NAMES:
        if module_name in DEPENDENCY_INSTALL_REQUIREMENTS:
            return "QGIS runtime or Geo-SAM"
        return "QGIS runtime"
    return "Geo-SAM managed"


def dependency_status_text(state: DependencyState) -> str:
    """Return the display status text for a dependency state.

    Parameters
    ----------
    state : DependencyState
        Resolved dependency state.

    Returns
    -------
    str
        Human-readable status text for the settings dependency table.

    """
    if state in {"installed_qgis_runtime", "installed_plugin_managed"}:
        return "Installed"
    return "Missing"


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
        Raw persisted value.

    Returns
    -------
    PreviewRenderMode
        Normalized vectorization mode.

    """
    normalized_value = str(value or "pixel_level").strip().lower().replace("-", "_")
    if normalized_value in PREVIEW_RENDER_MODE_VALUES:
        return normalized_value  # type: ignore[return-value]
    return "pixel_level"


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
    settings.setdefault("image_source_mode", "live_encoding")
    settings.setdefault("default_minimum_pixels", 0)
    settings.setdefault("performance_mode", "balanced")
    settings.setdefault("preview_mode", True)
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
    register_plugin_managed_dependency_path()
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
    register_plugin_managed_dependency_path()
    rows: list[DependencyStatusRow] = []
    for module_name, distribution_name in DEPENDENCY_DISTRIBUTIONS.items():
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            state = dependency_state(module_name, installed=False)
            rows.append(
                {
                    "package": module_name,
                    "distribution": distribution_name,
                    "installed": False,
                    "version": "",
                    "state": state,
                    "installable": state == "missing_installable",
                    "source": dependency_source_text(module_name, state),
                }
            )
        else:
            state = dependency_state(module_name, installed=True)
            rows.append(
                {
                    "package": module_name,
                    "distribution": distribution_name,
                    "installed": True,
                    "version": distribution.version,
                    "state": state,
                    "installable": False,
                    "source": dependency_source_text(module_name, state),
                }
            )
    return rows


def missing_runtime_dependencies(module_names: Iterable[str]) -> list[str]:
    """Return missing runtime dependency module names in input order.

    Parameters
    ----------
    module_names : Iterable[str]
        Dependency module names to validate.

    Returns
    -------
    list[str]
        Missing dependency module names without duplicates.

    Notes
    -----
    The check uses distribution metadata so the result reflects the active
    QGIS Python environment after in-session installs.

    """
    known_status = dependency_status()
    missing_module_names: list[str] = []
    seen_module_names: set[str] = set()
    for module_name in module_names:
        if module_name in seen_module_names:
            continue
        seen_module_names.add(module_name)

        installed = known_status.get(module_name)
        if installed is None:
            distribution_name = DEPENDENCY_DISTRIBUTIONS.get(module_name, module_name)
            try:
                importlib.metadata.distribution(distribution_name)
            except importlib.metadata.PackageNotFoundError:
                installed = False
            else:
                installed = True
        if not installed:
            missing_module_names.append(module_name)
    return missing_module_names


def missing_segmentation_runtime_dependencies() -> list[str]:
    """Return missing dependencies required for segmentation queries.

    Returns
    -------
    list[str]
        Missing dependency module names required before GeoSAM query creation.

    """
    return missing_runtime_dependencies(SEGMENTATION_RUNTIME_DEPENDENCY_NAMES)


def missing_plugin_runtime_dependencies() -> list[str]:
    """Return missing dependencies managed by the Geo-SAM installer.

    Returns
    -------
    list[str]
        Missing module names shown on the Dependencies page.

    """
    return missing_runtime_dependencies(PLUGIN_RUNTIME_DEPENDENCY_NAMES)


def format_missing_dependencies_message(
    module_names: Iterable[str],
    *,
    action_label: str = "this action",
) -> str:
    """Return a user-facing message for missing runtime dependencies.

    Parameters
    ----------
    module_names : Iterable[str]
        Missing dependency module names.
    action_label : str, optional
        Action name inserted into the message body.

    Returns
    -------
    str
        Formatted QMessageBox-friendly message text.

    Raises
    ------
    ValueError
        Raised when no dependency names were provided.

    """
    unique_module_names = list(dict.fromkeys(module_names))
    if not unique_module_names:
        msg = "Cannot format a dependency message without dependency names."
        logger.error(msg)
        raise ValueError(msg)

    if len(unique_module_names) == 1:
        dependency_text = unique_module_names[0]
    elif len(unique_module_names) == 2:
        dependency_text = " and ".join(unique_module_names)
    else:
        dependency_text = ", ".join(unique_module_names[:-1])
        dependency_text = f"{dependency_text}, and {unique_module_names[-1]}"

    return (
        f"Geo-SAM needs {dependency_text} for {action_label}.\n\n"
        "Open Geo-SAM Settings and install the missing dependencies first."
    )


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


def _selected_dependency_module_names(
    module_names: Iterable[str] | None = None,
) -> list[str]:
    """Return known dependency module names selected for installation.

    Parameters
    ----------
    module_names : Iterable[str] | None, optional
        Dependency package names to install. When omitted, all known plugin
        dependencies are selected.

    Returns
    -------
    list[str]
        Known dependency module names in installation order.

    Raises
    ------
    RuntimeError
        Raised when no known dependencies were selected.

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
    return selected_module_names


def _create_qgis_runtime_constraints_file() -> Path | None:
    """Create a temporary pip constraints file for bundled QGIS packages.

    Returns
    -------
    Path | None
        Temporary constraints file path, or ``None`` when no protected runtime
        distributions are installed in the active environment.

    Raises
    ------
    RuntimeError
        Raised when the constraints file cannot be written.

    Notes
    -----
    The constraints are generated from the currently installed package
    versions. This preserves whichever versions the active QGIS runtime ships,
    including future runtimes that may bundle newer major versions.

    """
    constraint_lines: list[str] = []
    installed_versions: dict[str, str] = {}
    for distribution_name in QGIS_RUNTIME_CONSTRAINT_DISTRIBUTIONS:
        resolved_version = _runtime_constraint_version(distribution_name)
        if resolved_version is None:
            continue
        installed_versions[distribution_name] = resolved_version
        constraint_lines.append(f"{distribution_name}=={resolved_version}")

    constraint_lines.extend(_qgis_runtime_resolver_constraints(installed_versions))

    if not constraint_lines:
        return None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="geo_sam_qgis_runtime_",
            suffix=".constraints.txt",
            delete=False,
        ) as constraints_file:
            constraints_file.write("\n".join(constraint_lines))
            constraints_file.write("\n")
            return Path(constraints_file.name)
    except OSError as exc:
        msg = "Could not create QGIS runtime dependency constraints file."
        logger.error("%s Error: %s", msg, exc)
        raise RuntimeError(msg) from exc


def _qgis_runtime_resolver_constraints(installed_versions: dict[str, str]) -> list[str]:
    """Return extra pip constraints for QGIS runtime compatibility.

    Parameters
    ----------
    installed_versions : dict[str, str]
        Installed protected distribution versions keyed by distribution name.

    Returns
    -------
    list[str]
        Additional requirement constraints that prevent pip resolver backtracking
        into incompatible or source-only transitive dependency versions.

    Notes
    -----
    These constraints do not pin QGIS packages. They only guide transitive
    dependencies toward versions compatible with the active QGIS runtime.

    """
    constraints = list(PIP_RESOLVER_CONSTRAINT_REQUIREMENTS)
    constraints.extend(_torch_resolver_constraints())
    numpy_major_version = _distribution_major_version(installed_versions.get("numpy"))
    if numpy_major_version is not None and numpy_major_version < 2:
        constraints.append("opencv-python==4.11.0.86")
    constraints.extend(_setuptools_resolver_constraints())
    return constraints


def _torch_resolver_constraints() -> list[str]:
    """Return PyTorch constraints compatible with the active Python runtime.

    Returns
    -------
    list[str]
        Torch and torchvision constraints for pip resolution.

    Notes
    -----
    PyTorch 2.11 does not publish Python 3.9 wheels. QGIS environments based
    on Python 3.9 therefore need the latest known PyTorch pair with cp39
    wheels.

    """
    if sys.version_info < (3, 10):
        return list(TORCH_RESOLVER_CONSTRAINT_REQUIREMENTS_PY39)
    return list(TORCH_RESOLVER_CONSTRAINT_REQUIREMENTS_PY310_PLUS)


def _distribution_major_version(version_text: str | None) -> int | None:
    """Return the leading integer version component.

    Parameters
    ----------
    version_text : str | None
        Distribution version text.

    Returns
    -------
    int | None
        Major version number, or ``None`` when it cannot be parsed.

    """
    if version_text is None:
        return None
    major_text = version_text.split(".", maxsplit=1)[0]
    try:
        return int(major_text)
    except ValueError:
        logger.debug(
            "Could not parse distribution major version from %s.", version_text
        )
        return None


def _runtime_constraint_version(distribution_name: str) -> str | None:
    """Return the effective version used for a runtime constraint line.

    Parameters
    ----------
    distribution_name : str
        Distribution name written into the pip constraints file.

    Returns
    -------
    str | None
        Resolved version text, or ``None`` when the distribution is unavailable.

    Notes
    -----
    Some QGIS Python environments expose placeholder distribution metadata such
    as ``python-dateutil==0.0.0`` while the importable module is a valid
    release. This helper normalizes those cases so pip receives a satisfiable
    constraint set.

    """
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None

    distribution_version = distribution.version
    if distribution_name != "python-dateutil" or distribution_version != "0.0.0":
        return distribution_version

    try:
        dateutil_module = importlib.import_module("dateutil")
    except ImportError:
        logger.warning(
            "python-dateutil distribution metadata reports version %s, and the "
            "dateutil module could not be imported for a fallback version.",
            distribution_version,
        )
        return distribution_version

    module_version = getattr(dateutil_module, "__version__", None)
    if not module_version:
        logger.warning(
            "python-dateutil distribution metadata reports version %s, and the "
            "dateutil module does not expose __version__ for fallback.",
            distribution_version,
        )
        return distribution_version

    logger.info(
        "Using python-dateutil module version %s instead of placeholder "
        "distribution version %s in runtime constraints.",
        module_version,
        distribution_version,
    )
    return str(module_version)


def _setuptools_resolver_constraints() -> list[str]:
    """Return setuptools constraints compatible with recent torch builds.

    Returns
    -------
    list[str]
        Additional setuptools constraints for pip resolution.

    Notes
    -----
    Recent torch releases require ``setuptools<82``. Some QGIS Python runtimes
    currently bundle ``setuptools==82.0.0`` or newer, which would make a
    generated exact runtime pin impossible to satisfy. The plugin does not rely
    on the QGIS-bundled setuptools version at runtime, so dependency installs
    should prefer the torch-compatible range instead of preserving that exact
    version.

    """
    try:
        setuptools_version = importlib.metadata.version("setuptools")
    except importlib.metadata.PackageNotFoundError:
        return ["setuptools<82"]

    setuptools_major_version = _distribution_major_version(setuptools_version)
    if setuptools_major_version is None:
        logger.debug(
            "Could not determine setuptools major version from %s; using "
            "torch-compatible upper bound.",
            setuptools_version,
        )
        return ["setuptools<82"]
    if setuptools_major_version >= 82:
        logger.info(
            "QGIS runtime provides setuptools %s; using setuptools<82 for "
            "torch compatibility during dependency installation.",
            setuptools_version,
        )
        return ["setuptools<82"]
    return [f"setuptools=={setuptools_version}"]


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
    prefix_path = Path(sys.prefix).expanduser()
    executable_path = Path(sys.executable).expanduser()
    candidate_paths: list[Path] = []

    candidate_paths.extend(_iter_prefix_python_candidates(prefix_path))
    candidate_paths.extend(_iter_platform_qgis_python_candidates(executable_path))

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

    return _unique_python_candidates(candidate_paths)


def _iter_prefix_python_candidates(prefix_path: Path) -> list[Path]:
    """Return Python executables below and near ``sys.prefix``.

    Parameters
    ----------
    prefix_path : Path
        Active Python prefix path.

    Returns
    -------
    list[Path]
        Candidate executable paths for virtualenv, conda, and system prefixes.

    Notes
    -----
    Conda-based QGIS installs normally expose the interpreter in
    ``sys.prefix/bin`` or ``sys.prefix/Scripts``. The parent scan is restricted
    to the active prefix name so it can cover relocated prefixes without
    accidentally selecting another conda environment.

    """
    candidate_paths: list[Path] = []
    candidate_directories = [
        prefix_path / "bin",
        prefix_path / "Scripts",
        prefix_path,
    ]
    for directory_path in candidate_directories:
        candidate_paths.extend(_iter_python_files(directory_path))

    parent_path = prefix_path.parent
    prefix_name = prefix_path.name
    for directory_name in ("bin", "Scripts"):
        glob_pattern = f"{prefix_name}*/{directory_name}/python*"
        candidate_paths.extend(sorted(parent_path.glob(glob_pattern)))

    return candidate_paths


def _iter_python_files(directory_path: Path) -> list[Path]:
    """Return Python-like files in a directory.

    Parameters
    ----------
    directory_path : Path
        Directory to scan.

    Returns
    -------
    list[Path]
        Ordered candidate files whose names begin with ``python``.

    """
    return sorted(directory_path.glob("python*"))


def _unique_python_candidates(candidate_paths: Iterable[Path]) -> list[Path]:
    """Return unique Python candidates without resolving symlink wrappers.

    Parameters
    ----------
    candidate_paths : Iterable[Path]
        Raw candidate executable paths.

    Returns
    -------
    list[Path]
        Candidate paths with duplicate textual paths removed.

    Notes
    -----
    The macOS QGIS app exposes ``Contents/MacOS/python`` as a symlink wrapper
    that sets ``PYTHONHOME`` before running the real binary. Resolving that
    symlink would break direct execution, so this helper keeps the original
    executable path.

    """
    unique_candidates: list[Path] = []
    seen_paths: set[str] = set()
    for candidate_path in candidate_paths:
        try:
            expanded_candidate = candidate_path.expanduser()
        except OSError:
            continue
        if not expanded_candidate.is_absolute():
            expanded_candidate = Path.cwd() / expanded_candidate
        candidate_key = str(expanded_candidate)
        if candidate_key in seen_paths:
            continue
        seen_paths.add(candidate_key)
        unique_candidates.append(expanded_candidate)
    return unique_candidates


def _iter_platform_qgis_python_candidates(executable_path: Path) -> list[Path]:
    """Return platform-specific QGIS Python executable candidates.

    Parameters
    ----------
    executable_path : Path
        Active QGIS process executable path.

    Returns
    -------
    list[Path]
        Candidate Python executable paths under the detected QGIS root.

    """
    if sys.platform == "win32":
        return _iter_windows_qgis_python_candidates(executable_path)
    if sys.platform == "darwin":
        return _iter_macos_qgis_python_candidates(executable_path)
    return _iter_linux_qgis_python_candidates(executable_path)


def _iter_windows_qgis_python_candidates(executable_path: Path) -> list[Path]:
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


def _iter_macos_qgis_python_candidates(executable_path: Path) -> list[Path]:
    """Return Python executables for official macOS QGIS app bundles.

    Parameters
    ----------
    executable_path : Path
        Active QGIS process executable path.

    Returns
    -------
    list[Path]
        Candidate Python executable paths under ``QGIS.app``.

    Notes
    -----
    The official macOS app uses ``Contents/MacOS/python`` as a relocatable
    wrapper. Prefer that path over versioned binaries because the wrapper sets
    ``PYTHONHOME`` correctly before Python starts.

    """
    contents_path: Path | None = None
    for parent_path in (executable_path, *executable_path.parents):
        if parent_path.name != "Contents":
            continue
        contents_path = parent_path
        break
    if contents_path is None:
        return []

    macos_path = contents_path / "MacOS"
    candidate_paths = [
        macos_path / "python",
        macos_path / "python3",
    ]
    candidate_paths.extend(_iter_python_files(macos_path))
    return candidate_paths


def _iter_linux_qgis_python_candidates(executable_path: Path) -> list[Path]:
    """Return Python executables near Linux QGIS launchers.

    Parameters
    ----------
    executable_path : Path
        Active QGIS process executable path.

    Returns
    -------
    list[Path]
        Candidate Python executable paths near the launcher.

    """
    candidate_paths: list[Path] = []
    for parent_path in executable_path.parents:
        if parent_path.name not in {"bin", "sbin"}:
            continue
        candidate_paths.extend(_iter_python_files(parent_path))
        break
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
    executable_path = Path(sys.executable).expanduser()
    if sys.platform == "darwin":
        for candidate_path in _iter_macos_qgis_python_candidates(executable_path):
            if not _is_python_interpreter_candidate(candidate_path):
                continue
            return candidate_path

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
) -> DependencyInstallCommand:
    """Build the first pip command used to install plugin dependencies.

    Parameters
    ----------
    module_names : Iterable[str] | None, optional
        Dependency package names to install. When omitted, all known plugin
        dependencies are installed.

    Returns
    -------
    DependencyInstallCommand
        Command arguments suitable for ``subprocess`` execution.

    Raises
    ------
    RuntimeError
        Raised when the current QGIS runtime Python interpreter cannot be
        resolved.

    """
    return get_dependency_install_commands(module_names)[0]


def get_dependency_install_commands(
    module_names: Iterable[str] | None = None,
) -> DependencyInstallPlan:
    """Build pip commands used to install plugin dependencies.

    Parameters
    ----------
    module_names : Iterable[str] | None, optional
        Dependency package names to install. When omitted, all known plugin
        dependencies are installed.

    Returns
    -------
    DependencyInstallPlan
        Command argument lists suitable for ``subprocess`` execution.

    Raises
    ------
    RuntimeError
        Raised when the current QGIS runtime Python interpreter cannot be
        resolved or no known dependencies were selected.

    Notes
    -----
    The local ``geosam`` package is installed in a separate ``--no-deps`` step.
    QGIS already owns geospatial runtime packages such as rasterio, geopandas,
    and pyproj; resolving the local package dependencies through pip can force
    pip to reconcile wheel metadata that does not match QGIS' bundled runtime.

    """
    selected_module_names = _selected_dependency_module_names(module_names)
    dependency_target_path = register_plugin_managed_dependency_path(create=True)
    python_interpreter = resolve_python_interpreter()
    constraints_path = _create_qgis_runtime_constraints_file()
    commands: DependencyInstallPlan = []
    resolver_module_names = [
        module_name for module_name in selected_module_names if module_name != "geosam"
    ]

    if resolver_module_names:
        command = _dependency_install_command_base(
            python_interpreter,
            constraints_path,
            dependency_target_path,
        )
        command.extend(
            [
                _dependency_install_requirement(module_name)
                for module_name in resolver_module_names
            ]
        )
        commands.append(command)

    if "geosam" in selected_module_names:
        commands.append(
            [
                str(python_interpreter),
                "-m",
                "pip",
                "install",
                "--target",
                str(dependency_target_path),
                "--upgrade",
                "--no-deps",
                _dependency_install_requirement("geosam"),
            ]
        )

    return commands


def format_dependency_install_commands(commands: Iterable[Iterable[str]]) -> str:
    """Return readable dependency installation commands.

    Parameters
    ----------
    commands : Iterable[Iterable[str]]
        Command argument lists produced by :func:`get_dependency_install_commands`.

    Returns
    -------
    str
        Shell-readable command text for logs, one command per line.

    """
    return "\n".join(format_dependency_install_command(command) for command in commands)


def _dependency_install_command_base(
    python_interpreter: Path,
    constraints_path: Path | None,
    dependency_target_path: Path,
) -> DependencyInstallCommand:
    """Build the shared pip install command prefix.

    Parameters
    ----------
    python_interpreter : Path
        Python interpreter used by the active QGIS runtime.
    constraints_path : Path | None
        Runtime constraints file path, or ``None`` when no constraints exist.
    dependency_target_path : Path
        Plugin-managed dependency directory used as the pip ``--target`` path.

    Returns
    -------
    DependencyInstallCommand
        Command prefix for dependency-resolving pip installs.

    """
    command: DependencyInstallCommand = [
        str(python_interpreter),
        "-m",
        "pip",
        "install",
        "--target",
        str(dependency_target_path),
        "--upgrade",
        "--upgrade-strategy",
        "only-if-needed",
        "--only-binary",
        ",".join(PIP_ONLY_BINARY_DISTRIBUTIONS),
    ]
    if constraints_path is not None:
        command.extend(["--constraint", str(constraints_path)])
    return command


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
        commands = get_dependency_install_commands()
    except RuntimeError as exc:
        logger.error("Dependency installation aborted: %s", exc)
        return False, str(exc)

    output_lines: list[str] = []

    for command in commands:
        if log_callback is not None:
            log_callback(f"Command: {format_dependency_install_command(command)}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                output_lines.append(line)
                if log_callback is not None:
                    log_callback(line)

        return_code = process.wait()
        if return_code != 0:
            output = "\n".join(line for line in output_lines if line.strip())
            return False, output

    output = "\n".join(line for line in output_lines if line.strip())
    register_plugin_managed_dependency_path()
    return True, output


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


def trim_cache_if_needed() -> int:
    """Delete the oldest half of cache files when the max size is exceeded.

    Checks the current cache size against the configured maximum.  When the
    limit is reached, files are sorted by modification time (oldest first)
    and the older half is removed.

    Returns
    -------
    int
        Number of removed cache files, or 0 when no trimming was needed.

    """
    settings = load_plugin_settings()
    cache_dir = get_cache_directory()
    if not settings.get("cache_enabled", True):
        return 0

    max_size_bytes = int(settings["cache_max_size_mb"]) * 1024 * 1024
    current_size = get_cache_size_bytes(cache_dir)
    if current_size <= max_size_bytes:
        return 0

    files = sorted(
        (p for p in cache_dir.rglob("*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        return 0

    # Delete the oldest half of the files (by count).
    remove_count = max(1, len(files) // 2)
    removed = 0
    for file_path in files[:remove_count]:
        try:
            file_path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


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
