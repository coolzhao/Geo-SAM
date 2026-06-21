"""Plugin-managed dependency path helpers for Geo-SAM."""

from __future__ import annotations

import hashlib
import importlib
import logging
import platform
import shutil
import sys
import sysconfig
from pathlib import Path
from typing import TypedDict

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DEPENDENCY_ROOT = PLUGIN_ROOT / ".deps"
PLUGIN_RUNTIME_DEPENDENCIES_ROOT = PLUGIN_DEPENDENCY_ROOT / "runtimes"
LEGACY_PLUGIN_MANAGED_SITE_PACKAGES = PLUGIN_DEPENDENCY_ROOT / "python"

logger = logging.getLogger(__name__)


class DependencyPathStats(TypedDict):
    """Summary information for a plugin-managed dependency path.

    Attributes
    ----------
    path : Path
        Dependency directory path.
    exists : bool
        Whether the directory exists.
    file_count : int
        Number of files contained in the directory.
    size_bytes : int
        Total size of files contained in the directory.

    """

    path: Path
    exists: bool
    file_count: int
    size_bytes: int


def get_plugin_managed_site_packages(*, create: bool = False) -> Path:
    """Return the plugin-managed dependency directory for this runtime.

    Parameters
    ----------
    create : bool, optional
        Whether to create the dependency directory before returning it.

    Returns
    -------
    Path
        Directory used as the pip ``--target`` location for plugin-managed
        Python packages in the active QGIS Python runtime.

    """
    dependency_path = (
        PLUGIN_RUNTIME_DEPENDENCIES_ROOT / _runtime_dependency_key() / "python"
    )
    if create:
        dependency_path.mkdir(parents=True, exist_ok=True)
    return dependency_path


def register_plugin_managed_dependency_path(*, create: bool = False) -> Path:
    """Expose plugin-managed dependencies to the active Python runtime.

    Parameters
    ----------
    create : bool, optional
        Whether to create the dependency directory before registering it.

    Returns
    -------
    Path
        Registered dependency directory path.

    """
    dependency_path = get_plugin_managed_site_packages(create=create)
    dependency_path_text = str(dependency_path)
    if dependency_path_text not in sys.path:
        sys.path.insert(0, dependency_path_text)
        importlib.invalidate_caches()
    return dependency_path


def get_plugin_managed_dependency_stats(path: Path) -> DependencyPathStats:
    """Return size and file-count information for a dependency directory.

    Parameters
    ----------
    path : Path
        Dependency directory to inspect.

    Returns
    -------
    DependencyPathStats
        Summary information for the dependency directory.

    """
    exists = path.exists()
    file_count = 0
    size_bytes = 0
    if exists:
        for child_path in path.rglob("*"):
            if not child_path.is_file():
                continue
            file_count += 1
            try:
                size_bytes += child_path.stat().st_size
            except OSError as exc:
                logger.warning(
                    "Could not stat dependency file %s while computing size: %s",
                    child_path,
                    exc,
                )
    return {
        "path": path,
        "exists": exists,
        "file_count": file_count,
        "size_bytes": size_bytes,
    }


def iter_plugin_managed_runtime_site_packages() -> list[Path]:
    """Return existing plugin-managed dependency directories for all runtimes.

    Returns
    -------
    list[Path]
        Existing runtime-specific dependency directories.

    """
    if not PLUGIN_RUNTIME_DEPENDENCIES_ROOT.exists():
        return []
    dependency_paths: list[Path] = []
    for runtime_path in sorted(PLUGIN_RUNTIME_DEPENDENCIES_ROOT.iterdir()):
        site_packages_path = runtime_path / "python"
        if site_packages_path.exists():
            dependency_paths.append(site_packages_path)
    return dependency_paths


def iter_all_plugin_managed_site_packages(*, include_legacy: bool = True) -> list[Path]:
    """Return all plugin-managed dependency directories.

    Parameters
    ----------
    include_legacy : bool, optional
        Whether to include the old shared ``.deps/python`` directory.

    Returns
    -------
    list[Path]
        Existing plugin-managed dependency directories.

    """
    dependency_paths = iter_plugin_managed_runtime_site_packages()
    if include_legacy and LEGACY_PLUGIN_MANAGED_SITE_PACKAGES.exists():
        dependency_paths.append(LEGACY_PLUGIN_MANAGED_SITE_PACKAGES)
    return dependency_paths


def clear_current_plugin_managed_site_packages() -> int:
    """Delete dependencies installed for the active runtime.

    Returns
    -------
    int
        Number of files removed.

    """
    dependency_path = get_plugin_managed_site_packages()
    return _remove_dependency_path(dependency_path)


def clear_all_plugin_managed_site_packages() -> int:
    """Delete all plugin-managed dependencies, including legacy installs.

    Returns
    -------
    int
        Number of files removed.

    """
    removed_count = 0
    for dependency_path in iter_all_plugin_managed_site_packages(include_legacy=True):
        removed_count += _remove_dependency_path(dependency_path)
    _remove_empty_directory(PLUGIN_RUNTIME_DEPENDENCIES_ROOT)
    _remove_empty_directory(PLUGIN_DEPENDENCY_ROOT)
    return removed_count


def _runtime_dependency_key() -> str:
    """Return a stable dependency directory key for the active runtime.

    Returns
    -------
    str
        Runtime-specific dependency directory name.

    """
    python_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    platform_tag = _safe_path_part(sysconfig.get_platform())
    architecture_tag = _safe_path_part(platform.machine() or "unknown")
    identity_parts = [
        sys.prefix,
        getattr(sys, "base_prefix", ""),
        sys.executable,
        python_tag,
        platform_tag,
        architecture_tag,
    ]
    runtime_hash = hashlib.sha256("\0".join(identity_parts).encode("utf-8"))
    return f"{python_tag}-{platform_tag}-{architecture_tag}-{runtime_hash.hexdigest()[:12]}"


def _safe_path_part(value: str) -> str:
    """Return a filesystem-safe path component.

    Parameters
    ----------
    value : str
        Raw path component text.

    Returns
    -------
    str
        Sanitized path component text.

    """
    sanitized = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "-"
        for character in value
    ).strip("-._")
    return sanitized or "unknown"


def _remove_dependency_path(dependency_path: Path) -> int:
    """Remove one plugin-managed dependency directory.

    Parameters
    ----------
    dependency_path : Path
        Dependency directory to delete.

    Returns
    -------
    int
        Number of files removed.

    """
    dependency_path_text = str(dependency_path)
    sys.path[:] = [path_text for path_text in sys.path if path_text != dependency_path_text]
    importlib.invalidate_caches()
    if not dependency_path.exists():
        return 0
    stats = get_plugin_managed_dependency_stats(dependency_path)
    try:
        shutil.rmtree(dependency_path)
    except OSError as exc:
        logger.error(
            "Could not remove plugin-managed dependency directory %s: %s",
            dependency_path,
            exc,
        )
        raise
    _remove_empty_directory(dependency_path.parent)
    return stats["file_count"]


def _remove_empty_directory(path: Path) -> None:
    """Remove a directory when it exists and is empty.

    Parameters
    ----------
    path : Path
        Directory path to remove.

    """
    try:
        path.rmdir()
    except FileNotFoundError:
        return
    except OSError:
        return
