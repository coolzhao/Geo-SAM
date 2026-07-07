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
from typing import TYPE_CHECKING, TypedDict

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DEPENDENCY_ROOT = PLUGIN_ROOT / ".deps"
PLUGIN_RUNTIME_DEPENDENCIES_ROOT = PLUGIN_DEPENDENCY_ROOT / "runtimes"
LEGACY_PLUGIN_MANAGED_SITE_PACKAGES = PLUGIN_DEPENDENCY_ROOT / "python"

if TYPE_CHECKING:
    from .runtime_profile import TorchProfile

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


class DependencyPathStatsDedup(TypedDict):
    """Aggregated disk-usage stats across multiple runtime folders.

    Hardlinked files (identical ``(st_dev, st_ino)`` pair, common on
    Linux where uv uses ``--link-mode=hardlink``) are counted only once
    in :attr:`dedup_size_bytes`. Symlinks are skipped entirely, so a
    symlinked target is never double-counted.

    On macOS, uv uses APFS ``clonefile`` instead of hardlinks. Clones
    have *distinct* inodes and ``st_nlink == 1``, so they cannot be
    deduplicated by user-space tools (including this function). On that
    platform :attr:`dedup_size_bytes` equals :attr:`logical_size_bytes`,
    and :attr:`shared_bytes` is ``0`` even though real on-disk usage may
    be substantially lower. Callers should surface that caveat to the
    user rather than pretending to know the real figure.

    Attributes
    ----------
    logical_size_bytes : int
        Sum of ``st_size`` over every regular file seen (the figure the
        naive per-folder scan would report).
    dedup_size_bytes : int
        Sum of ``st_size`` over unique ``(st_dev, st_ino)`` pairs. This
        is the best user-space estimate of real disk usage on
        Linux/Windows; on macOS it is an upper bound.
    shared_bytes : int
        ``logical_size_bytes - dedup_size_bytes``. Bytes "saved" by
        hardlinks/symlinks that the naive scan would have double-counted.
    file_count : int
        Total number of regular (non-symlink) files seen.
    unique_file_count : int
        Number of unique ``(st_dev, st_ino)`` pairs.
    symlink_count : int
        Number of symlinks encountered and skipped.

    """

    logical_size_bytes: int
    dedup_size_bytes: int
    shared_bytes: int
    file_count: int
    unique_file_count: int
    symlink_count: int


def get_plugin_managed_site_packages(
    *,
    create: bool = False,
    profile: TorchProfile | None = None,
) -> Path:
    """Return the plugin-managed dependency directory for this runtime.

    Parameters
    ----------
    create : bool, optional
        Whether to create the dependency directory before returning it.
    profile : TorchProfile | None, optional
        Torch profile to use. When ``None``, the currently active profile
        for this runtime is used.

    Returns
    -------
    Path
        Directory used as the pip ``--target`` location for plugin-managed
        Python packages in the active QGIS Python runtime.

    Notes
    -----
    Legacy directory migration is performed once at plugin startup via
    :func:`register_plugin_managed_dependency_path`, not on every call.

    """
    from .runtime_profile import active_profile, profile_target_path

    key = _runtime_dependency_key()

    if profile is None:
        profile = active_profile(key)
    dependency_path = profile_target_path(key, profile)

    if create:
        dependency_path.mkdir(parents=True, exist_ok=True)
    return dependency_path


def register_plugin_managed_dependency_path(*, create: bool = False) -> Path:
    """Expose plugin-managed dependencies to the active Python runtime.

    Performs a one-time legacy directory migration (if needed) before
    registering the active profile's target directory on ``sys.path``.

    Parameters
    ----------
    create : bool, optional
        Whether to create the dependency directory before registering it.

    Returns
    -------
    Path
        Registered dependency directory path.

    """
    from .runtime_profile import active_profile, migrate_legacy_profile, profile_target_path

    key = _runtime_dependency_key()
    migrate_legacy_profile(key)

    profile = active_profile(key)
    dependency_path = profile_target_path(key, profile)
    if create:
        dependency_path.mkdir(parents=True, exist_ok=True)

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


def get_plugin_managed_dependency_stats_dedup(
    paths: list[Path],
) -> DependencyPathStatsDedup:
    """Aggregate disk usage across runtime folders, deduplicating hardlinks.

    Walks every path in *paths* once, groups regular files by their
    ``(st_dev, st_ino)`` pair, and sums ``st_size`` over unique inodes
    only. Hardlinked files (the default uv layout on Linux/Windows when
    ``--link-mode=hardlink`` is used within one filesystem) are therefore
    counted exactly once, even if they appear under several runtime
    profile folders. Symlinks are skipped so symlinked targets are never
    double-counted.

    On macOS, uv installs via APFS ``clonefile``. Clones carry distinct
    inodes and ``st_nlink == 1``, so they are *not* detected as shared
    by this function. The returned :attr:`dedup_size_bytes` is therefore
    an **upper bound** on macOS; the real on-disk usage may be lower
    (often dramatically, e.g. two full torch installs sharing one set of
    blocks). Callers should communicate this caveat to the user instead
    of presenting the dedup figure as ground truth on macOS.

    Parameters
    ----------
    paths : list[Path]
        Runtime dependency directories to scan. Non-existent paths are
        silently skipped.

    Returns
    -------
    DependencyPathStatsDedup
        Aggregated statistics with hardlink/symlink deduplication applied.

    """
    seen: dict[tuple[int, int], int] = {}
    logical_size_bytes = 0
    file_count = 0
    symlink_count = 0
    for path in paths:
        if not path.exists():
            continue
        for child_path in path.rglob("*"):
            if child_path.is_symlink():
                symlink_count += 1
                continue
            if not child_path.is_file():
                continue
            try:
                stat_result = child_path.stat()
            except OSError as exc:
                logger.warning(
                    "Could not stat dependency file %s while computing "
                    "deduplicated size: %s",
                    child_path,
                    exc,
                )
                continue
            file_count += 1
            file_size = stat_result.st_size
            logical_size_bytes += file_size
            inode_key = (stat_result.st_dev, stat_result.st_ino)
            seen.setdefault(inode_key, file_size)
    dedup_size_bytes = sum(seen.values())
    return {
        "logical_size_bytes": logical_size_bytes,
        "dedup_size_bytes": dedup_size_bytes,
        "shared_bytes": logical_size_bytes - dedup_size_bytes,
        "file_count": file_count,
        "unique_file_count": len(seen),
        "symlink_count": symlink_count,
    }


def iter_plugin_managed_runtime_site_packages() -> list[Path]:
    """Return existing plugin-managed dependency directories for all runtimes.

    Scans both the new profile-based layout (``<key>/<profile>/python``) and
    the legacy flat layout (``<key>/python``).

    Returns
    -------
    list[Path]
        Existing runtime-specific dependency directories.

    """
    if not PLUGIN_RUNTIME_DEPENDENCIES_ROOT.exists():
        return []
    dependency_paths: list[Path] = []
    for runtime_path in sorted(PLUGIN_RUNTIME_DEPENDENCIES_ROOT.iterdir()):
        if not runtime_path.is_dir():
            continue
        # New profile-based layout: <key>/<profile>/python
        for child in sorted(runtime_path.iterdir()):
            if child.is_dir() and child.name in {
                "auto", "cpu", "cu128", "cu130", "custom",
            }:
                site_packages = child / "python"
                if site_packages.exists():
                    dependency_paths.append(site_packages)
        # Legacy flat layout: <key>/python
        legacy_site_packages = runtime_path / "python"
        if legacy_site_packages.exists():
            dependency_paths.append(legacy_site_packages)
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
    """Delete dependencies installed for the active runtime profile.

    Returns
    -------
    int
        Number of files removed.

    """
    dependency_path = get_plugin_managed_site_packages()
    return _remove_dependency_path(dependency_path)


def clear_profile_site_packages(key: str, profile: TorchProfile) -> int:
    """Delete dependencies installed for a specific profile.

    Parameters
    ----------
    key : str
        Runtime dependency key.
    profile : TorchProfile
        Torch profile to clear.

    Returns
    -------
    int
        Number of files removed.

    """
    from .runtime_profile import profile_target_path

    dependency_path = profile_target_path(key, profile)
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


def _clear_metadata_cache() -> None:
    """Clear :mod:`importlib.metadata` internal caches.

    :func:`importlib.invalidate_caches` only clears the import machinery's
    meta-path finder caches — it does **not** necessarily clear
    :mod:`importlib.metadata`'s own internal ``FastPath`` / ``_cache``
    state.  After modifying ``sys.path`` (e.g. swapping the active torch
    profile), stale metadata caches can cause
    :func:`importlib.metadata.distribution` to return outdated results.

    This helper attempts the public ``invalidate_caches`` API (Python 3.12+)
    and falls back to clearing the private ``MetadataPathFinder._cache``
    attribute on older interpreters.

    """
    try:
        import importlib.metadata as _meta

        invalidate = getattr(_meta, "invalidate_caches", None)
        if callable(invalidate):
            invalidate()
            return
        # Python 3.9–3.11: clear the private cache attribute directly.
        finder_cls = getattr(_meta, "MetadataPathFinder", None)
        if finder_cls is not None and hasattr(finder_cls, "_cache"):
            finder_cls._cache = None  # type: ignore[attr-defined]
    except Exception:
        logger.debug("Could not clear importlib.metadata cache", exc_info=True)


def swap_dependency_path(old_path: Path, new_path: Path) -> None:
    """Replace one dependency path with another in ``sys.path``.

    Parameters
    ----------
    old_path : Path
        Dependency directory to remove from ``sys.path``.
    new_path : Path
        Dependency directory to insert at position 0 of ``sys.path``.

    """
    old_text = str(old_path)
    new_text = str(new_path)

    sys.path[:] = [p for p in sys.path if p != old_text]
    if new_text not in sys.path:
        sys.path.insert(0, new_text)
    importlib.invalidate_caches()
    _clear_metadata_cache()
    logger.debug("Swapped dependency path: %s → %s", old_path, new_path)


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
