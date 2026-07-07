"""Runtime profile management for Geo-SAM dependency installation.

This module manages torch backend profiles (auto, cpu, cu128, cu130, custom)
and their corresponding target directories for dependency installation.
"""

from __future__ import annotations

import logging
import platform
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .dependency_path import (
    PLUGIN_RUNTIME_DEPENDENCIES_ROOT,
    _runtime_dependency_key,
    swap_dependency_path,
)

logger = logging.getLogger(__name__)

TorchProfile = Literal["auto", "cpu", "cu128", "cu130", "custom"]
PROFILE_VALUES: tuple[TorchProfile, ...] = ("auto", "cpu", "cu128", "cu130", "custom")
DEFAULT_PROFILE: TorchProfile = "auto"

PROFILE_DISPLAY_NAMES: dict[TorchProfile, str] = {
    "auto": "Auto",
    "cpu": "CPU",
    "cu128": "NVIDIA CUDA 12.8",
    "cu130": "NVIDIA CUDA 13.0",
    "custom": "Custom",
}


def profiles_for_platform() -> list[TorchProfile]:
    """Return available torch profiles for the current platform.

    Returns
    -------
    list[TorchProfile]
        List of profile keys available on this platform. On macOS, CUDA
        profiles (cu128, cu130) are excluded.

    """
    if platform.system() == "Darwin":
        return ["auto", "cpu", "custom"]
    return list(PROFILE_VALUES)


def profile_display_name(profile: TorchProfile) -> str:
    """Return the human-readable display name for a profile.

    Parameters
    ----------
    profile : TorchProfile
        Profile key.

    Returns
    -------
    str
        Display name for the profile.

    """
    return PROFILE_DISPLAY_NAMES.get(profile, profile)


def profile_target_path(key: str, profile: TorchProfile) -> Path:
    """Return the target directory for a specific profile.

    Parameters
    ----------
    key : str
        Runtime dependency key from :func:`_runtime_dependency_key`.
    profile : TorchProfile
        Torch profile identifier.

    Returns
    -------
    Path
        Target directory path for pip/uv ``--target``.

    """
    return PLUGIN_RUNTIME_DEPENDENCIES_ROOT / key / profile / "python"


def active_profile(key: str | None = None) -> TorchProfile:
    """Return the currently active torch profile for a runtime.

    Parameters
    ----------
    key : str | None, optional
        Runtime dependency key. When ``None``, the current runtime key is used.

    Returns
    -------
    TorchProfile
        Active profile identifier, defaults to :data:`DEFAULT_PROFILE`.

    """
    if key is None:
        key = _runtime_dependency_key()
    profile_file = PLUGIN_RUNTIME_DEPENDENCIES_ROOT / key / "active_profile.txt"
    if profile_file.exists():
        text = profile_file.read_text(encoding="utf-8").strip()
        if text in PROFILE_VALUES:
            return text  # type: ignore[return-value]
        logger.warning(
            "Invalid active profile %r in %s, using default %r",
            text,
            profile_file,
            DEFAULT_PROFILE,
        )
    return DEFAULT_PROFILE


def set_active_profile(key: str, profile: TorchProfile) -> None:
    """Persist the active torch profile for a runtime.

    Parameters
    ----------
    key : str
        Runtime dependency key.
    profile : TorchProfile
        Profile to activate.

    """
    profile_dir = PLUGIN_RUNTIME_DEPENDENCIES_ROOT / key
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_file = profile_dir / "active_profile.txt"
    profile_file.write_text(profile + "\n", encoding="utf-8")
    logger.debug("Set active profile for %s to %s", key, profile)


def migrate_legacy_profile(key: str) -> bool:
    """Migrate legacy ``<key>/python`` directory to the default profile.

    This is an idempotent one-time migration for existing installations that
    used the old flat ``<key>/python`` layout. The legacy directory is moved
    into ``<key>/<DEFAULT_PROFILE>/python`` (i.e. ``auto``) and the active
    profile is pinned to :data:`DEFAULT_PROFILE`, so the persisted active
    profile and the combo-box default (``torch_profile`` setting) agree —
    previously this hard-coded ``cpu``, which left "Active: cpu | Selected:
    auto" after upgrade.

    Parameters
    ----------
    key : str
        Runtime dependency key.

    Returns
    -------
    bool
        ``True`` if migration was performed, ``False`` if already migrated or
        no legacy directory exists.

    """
    legacy_dir = PLUGIN_RUNTIME_DEPENDENCIES_ROOT / key / "python"
    default_dir = profile_target_path(key, DEFAULT_PROFILE)

    if not legacy_dir.exists():
        return False
    if default_dir.exists():
        # Already migrated; remove legacy if it's a different directory
        if legacy_dir.resolve() != default_dir.resolve():
            logger.warning(
                "Legacy directory %s exists alongside %s, removing legacy",
                legacy_dir,
                default_dir,
            )
            shutil.rmtree(legacy_dir, ignore_errors=True)
        return False

    # Perform migration into the default profile (auto) so active and default
    # stay consistent.
    default_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        legacy_dir.rename(default_dir)
        set_active_profile(key, DEFAULT_PROFILE)
        logger.info(
            "Migrated legacy dependency directory %s → %s",
            legacy_dir,
            default_dir,
        )
        return True
    except OSError as exc:
        logger.error("Failed to migrate legacy directory %s: %s", legacy_dir, exc)
        return False


def custom_index_path(key: str) -> Path:
    """Return the path to the custom index URL file for a runtime.

    Parameters
    ----------
    key : str
        Runtime dependency key.

    Returns
    -------
    Path
        Path to the custom index URL text file.

    """
    return PLUGIN_RUNTIME_DEPENDENCIES_ROOT / key / "custom" / "custom_index.txt"


def get_custom_index_url(key: str | None = None) -> str | None:
    """Return the custom PyPI index URL for the custom profile.

    Parameters
    ----------
    key : str | None, optional
        Runtime dependency key. When ``None``, the current runtime key is used.

    Returns
    -------
    str | None
        Custom index URL, or ``None`` if not configured.

    """
    if key is None:
        key = _runtime_dependency_key()
    index_file = custom_index_path(key)
    if index_file.exists():
        url = index_file.read_text(encoding="utf-8").strip()
        if url:
            return url
    return None


def set_custom_index_url(key: str, url: str) -> None:
    """Persist a custom PyPI index URL for the custom profile.

    Parameters
    ----------
    key : str
        Runtime dependency key.
    url : str
        Custom PyPI index URL.

    """
    index_file = custom_index_path(key)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text(url.strip() + "\n", encoding="utf-8")
    logger.debug("Set custom index URL for %s to %s", key, url)


def profile_is_installed(key: str, profile: TorchProfile) -> bool:
    """Check whether a profile has torch dependencies installed.

    Parameters
    ----------
    key : str
        Runtime dependency key.
    profile : TorchProfile
        Profile to check.

    Returns
    -------
    bool
        ``True`` if the profile target directory exists and contains the
        ``torch`` and ``torchvision`` dist-info directories (proof of a
        completed torch install, not a partial failure).

    """
    fully_installed, _ = profile_install_status(key, profile)
    return fully_installed


def profile_install_status(
    key: str,
    profile: TorchProfile,
) -> tuple[bool, list[str]]:
    """Return install completeness and the missing module list for a profile.

    Plugin-managed packages (torch, torchvision, ultralytics, geosam) are
    checked by scanning the profile's target directory for dist-info
    directories on disk, so the result is authoritative for a profile that
    is *not* the active one. QGIS runtime-provided packages (rasterio,
    geopandas, shapely, pyproj) are checked via
    :func:`importlib.metadata.distribution` because they live in the QGIS
    Python environment.

    Parameters
    ----------
    key : str
        Runtime dependency key.
    profile : TorchProfile
        Profile to check.

    Returns
    -------
    tuple[bool, list[str]]
        ``(fully_installed, missing_module_names)``. ``fully_installed`` is
        ``True`` only when torch and torchvision are both present (the same
        gate as :func:`profile_is_installed`); ``missing_module_names`` lists
        every plugin runtime dependency that is absent, in canonical order.

    """
    import importlib.metadata

    from .plugin_settings import (
        PLUGIN_RUNTIME_DEPENDENCY_NAMES,
        RUNTIME_PROVIDED_DEPENDENCY_NAMES,
    )

    target = profile_target_path(key, profile)
    if not target.exists():
        return False, list(PLUGIN_RUNTIME_DEPENDENCY_NAMES)
    _SUFFIX_LEN = len(".dist-info")
    installed_distributions = {
        child.name[:-_SUFFIX_LEN].rsplit("-", 1)[0]
        for child in target.iterdir()
        if child.is_dir() and child.name.endswith(".dist-info")
    }
    missing: list[str] = []
    for module_name in PLUGIN_RUNTIME_DEPENDENCY_NAMES:
        if module_name in RUNTIME_PROVIDED_DEPENDENCY_NAMES:
            try:
                importlib.metadata.distribution(module_name)
            except importlib.metadata.PackageNotFoundError:
                missing.append(module_name)
        elif module_name not in installed_distributions:
            missing.append(module_name)
    required = {"torch", "torchvision"}
    fully_installed = required.issubset(installed_distributions)
    return fully_installed, missing


@dataclass(frozen=True)
class SwitchResult:
    """Result of a profile switch operation.

    Attributes
    ----------
    was_installed : bool
        Whether the target profile was already installed before the switch.
    profile : TorchProfile
        The newly active profile.
    target_path : Path
        The target directory path for the new profile.

    """

    was_installed: bool
    profile: TorchProfile
    target_path: Path


@dataclass(frozen=True)
class SwitchPlan:
    """Prepared switch operation ready for async execution.

    Attributes
    ----------
    profile : TorchProfile
        Target profile to switch to.
    target_path : Path
        Target directory for the new profile.
    old_target_path : Path
        Currently active target directory (to remove from ``sys.path``).
    was_installed : bool
        Whether the target profile was already installed before the switch.
    install_commands : list[list[str]]
        uv commands to run when ``was_installed`` is ``False``; empty when
        no installation is needed.

    """

    profile: TorchProfile
    target_path: Path
    old_target_path: Path
    was_installed: bool
    install_commands: list[list[str]]


def prepare_switch(
    profile: TorchProfile,
    *,
    install_if_missing: bool = True,
    custom_index_url: str | None = None,
) -> SwitchPlan:
    """Validate and prepare a profile switch without executing installation.

    This splits the non-blocking preparation from the side-effecting
    :func:`commit_switch`. By default a switch is a pure ``sys.path`` swap:
    the target profile must already be installed (dependencies are managed by
    the "Install Missing" flow). Pass ``install_if_missing=True`` to have the
    caller build install commands itself.

    Parameters
    ----------
    profile : TorchProfile
        Target profile to switch to.
    install_if_missing : bool, optional
        When ``True``, return install commands for the target profile if it is
        not yet installed (used by the deprecated blocking
        :func:`switch_profile` helper). When ``False`` (default), raise if the
        profile is not installed — the UI blocks and hints the user to use
        "Install Missing".
    custom_index_url : str | None, optional
        Custom index URL for the ``custom`` profile.

    Returns
    -------
    SwitchPlan
        Prepared switch plan. ``install_commands`` is non-empty only when
        ``install_if_missing=True`` and the profile was missing.

    Raises
    ------
    ValueError
        Raised when the profile is not allowed on the current platform.
    RuntimeError
        Raised when the profile is not installed and
        ``install_if_missing`` is ``False``.

    """
    from .torch_profile import profile_allowed, profile_unavailable_reason

    if not profile_allowed(profile):
        reason = profile_unavailable_reason(profile) or "Profile not available"
        raise ValueError(f"Cannot switch to profile {profile!r}: {reason}")

    key = _runtime_dependency_key()
    was_installed = profile_is_installed(key, profile)
    target = profile_target_path(key, profile)
    old_profile = active_profile(key)
    old_target = profile_target_path(key, old_profile)

    install_commands: list[list[str]] = []
    if not was_installed:
        if not install_if_missing:
            raise RuntimeError(
                f"Profile {profile!r} is not installed at {target}. "
                "Install dependencies for it first (e.g. via Install Missing)."
            )
        # Auto-install path (only the deprecated blocking helper opts in).
        # Only install backend-specific packages (torch + torchvision).
        # Other dependencies (rasterio, geopandas, …) are shared across all
        # profiles and are installed once via "Install Missing".
        from .dependency_installer import build_uv_commands, make_plan

        plan = make_plan(
            ("torch", "torchvision"),
            profile,
            custom_index_url=custom_index_url,
        )
        install_commands = build_uv_commands(plan)

    return SwitchPlan(
        profile=profile,
        target_path=target,
        old_target_path=old_target,
        was_installed=was_installed,
        install_commands=install_commands,
    )


def commit_switch(plan: SwitchPlan) -> SwitchResult:
    """Commit a prepared profile switch after installation succeeded.

    Swaps ``sys.path``, persists the active profile, and releases loaded
    models. Call this only after ``plan.install_commands`` (if any) have run
    successfully.

    Parameters
    ----------
    plan : SwitchPlan
        Prepared switch plan from :func:`prepare_switch`.

    Returns
    -------
    SwitchResult
        Result of the switch operation.

    """
    key = _runtime_dependency_key()

    if plan.old_target_path != plan.target_path:
        swap_dependency_path(plan.old_target_path, plan.target_path)

    set_active_profile(key, plan.profile)
    _release_runtime_and_flush_torch()

    logger.info(
        "Switched to profile %s (target: %s, was_installed=%s)",
        plan.profile,
        plan.target_path,
        plan.was_installed,
    )

    return SwitchResult(
        was_installed=plan.was_installed,
        profile=plan.profile,
        target_path=plan.target_path,
    )



def switch_profile(
    profile: TorchProfile,
    *,
    install_if_missing: bool = True,
    log_callback: Callable[[str], None] | None = None,
) -> SwitchResult:
    """Switch the active torch profile (blocking convenience wrapper).

    .. deprecated::
        Prefer :func:`prepare_switch` + :func:`commit_switch` for UI paths
        so installation can run asynchronously via ``QProcess``. This helper
        blocks the caller for the full install duration.

    Parameters
    ----------
    profile : TorchProfile
        Target profile to switch to.
    install_if_missing : bool, optional
        Whether to automatically install dependencies when the target
        profile is not yet installed.
    log_callback : Callable[[str], None] | None, optional
        Callback for install log output.

    Returns
    -------
    SwitchResult
        Result of the switch operation.

    Raises
    ------
    ValueError
        Raised when the profile is not allowed on the current platform.
    RuntimeError
        Raised when the profile is not installed and ``install_if_missing``
        is ``False``, or when automatic installation fails.

    """
    plan = prepare_switch(profile, install_if_missing=install_if_missing)

    if plan.install_commands:
        from .dependency_installer import install_dependencies

        logger.info("Profile %s not installed, auto-installing...", profile)
        success, output = install_dependencies(
            profile=profile,
            log_callback=log_callback,
        )
        if not success:
            raise RuntimeError(
                f"Failed to install dependencies for profile {profile!r}:\n{output}"
            )

    return commit_switch(plan)


def _release_runtime_and_flush_torch() -> None:
    """Release loaded model engines and flush torch accelerator memory.

    This is called after switching torch profiles to avoid ABI conflicts
    between different torch builds. Errors are logged as warnings rather
    than raised, because a failure to release memory must not block the
    profile switch (the new torch will be imported fresh).

    """
    try:
        from .geosam_runtime import _flush_torch_memory, release_runtime_models

        release_runtime_models()
        _flush_torch_memory()
    except Exception as exc:
        logger.warning(
            "Could not release runtime models or flush torch memory "
            "during profile switch: %s",
            exc,
        )
