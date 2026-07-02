"""Dependency installation engine for Geo-SAM using uv.

This module builds and executes uv pip install commands for plugin
dependencies, with torch backend profile support.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .dependency_path import (
    _runtime_dependency_key,
    register_plugin_managed_dependency_path,
)
from .runtime_profile import (
    TorchProfile,
    active_profile,
    get_custom_index_url,
    profile_target_path,
)
from .torch_profile import uv_torch_flag
from .uv_manager import UV_CACHE_DIR

logger = logging.getLogger(__name__)


def resolve_uv_python_flags() -> list[str]:
    """Return uv ``--python-version`` flag.

    uv resolves wheels matching the **running** process architecture when
    ``--python-platform`` is omitted — which is the correct behavior because
    QGIS runs as the native host architecture (arm64 on Apple Silicon, x86_64
    on Intel). The previous ``--python-platform`` derived from
    :func:`sysconfig.get_platform` which reports the **compile-time** arch
    (always ``x86_64`` for the universal2 QGIS build), causing uv to install
    x86_64 wheels that fail to load in the native arm64 process.

    Notes
    -----
    uv's ``--python <path>`` inspects the interpreter by running it with
    ``-I`` (isolated mode). On macOS QGIS, the interpreter's compile-time
    ``prefix`` points at a CI build path
    (``/Users/runner/work/QGIS/QGIS/build/...``); the runtime relies on
    ``PYTHONHOME`` to locate the stdlib. The ``-I`` flag strips
    ``PYTHONHOME``, causing ``ModuleNotFoundError: No module named 'encodings'``.
    We therefore pass ``--python-version`` only (no ``--python``) and let uv
    detect the host platform automatically.

    Returns
    -------
    list[str]
        uv flags such as ``["--python-version", "3.12"]``.

    """
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    return ["--python-version", pyver]


@dataclass(frozen=True)
class DependencyPlan:
    """Complete plan for a dependency installation operation.

    Attributes
    ----------
    packages : tuple[str, ...]
        Package names to install.
    python_path : Path
        Path to the QGIS Python interpreter.
    target_path : Path
        Target directory for pip/uv ``--target``.
    constraints_path : Path | None
        Path to the constraints file, or ``None`` if no constraints.
    profile : TorchProfile
        Active torch profile.
    uv_binary : Path
        Path to the uv binary.
    custom_index_url : str | None
        Custom PyPI index URL for the ``custom`` profile.

    """

    packages: tuple[str, ...]
    python_path: Path
    target_path: Path
    constraints_path: Path | None
    profile: TorchProfile
    uv_binary: Path
    custom_index_url: str | None = None


def requirement_for(module_name: str) -> str:
    """Return the pip requirement string for a module.

    Parameters
    ----------
    module_name : str
        Module name from ``DEPENDENCY_INSTALL_REQUIREMENTS``.

    Returns
    -------
    str
        Pip requirement string (e.g., ``"torch"`` or ``"ultralytics>=8.4.33"``).
        For ``geosam``, returns the local repository path when it exists so
        local development installs take precedence over PyPI.

    """
    from .plugin_settings import _dependency_install_requirement

    return _dependency_install_requirement(module_name)


def _build_uv_command_prefix(plan: DependencyPlan) -> list[str]:
    """Build the shared uv pip install command prefix.

    Parameters
    ----------
    plan : DependencyPlan
        Installation plan.

    Returns
    -------
    list[str]
        Command prefix with common flags.

    Notes
    -----
    Uses ``--python-version`` and ``--python-platform`` instead of
    ``--python <path>`` to avoid uv's interpreter inspection, which fails on
    macOS QGIS (the ``-I`` isolated probe strips ``PYTHONHOME`` and breaks the
    wrapper interpreter). See :func:`resolve_uv_python_flags`.

    """
    cmd = [
        str(plan.uv_binary),
        "pip",
        "install",
        "--target",
        str(plan.target_path),
        "--no-config",
        "--cache-dir",
        str(UV_CACHE_DIR),
        "--no-python-downloads",
        "--strict",
    ]
    cmd.extend(resolve_uv_python_flags())
    cmd.extend(uv_torch_flag(plan.profile))

    if plan.profile == "custom" and plan.custom_index_url:
        cmd.extend(["--index-url", plan.custom_index_url])

    if plan.constraints_path is not None:
        cmd.extend(["--constraint", str(plan.constraints_path)])

    return cmd


def build_uv_commands(plan: DependencyPlan) -> list[list[str]]:
    """Build uv pip install commands from a dependency plan.

    The ``geosam`` local package is always installed separately with
    ``--no-deps`` to avoid pip resolving its geospatial dependencies
    (QGIS already owns those).

    Parameters
    ----------
    plan : DependencyPlan
        Installation plan.

    Returns
    -------
    list[list[str]]
        List of command argument lists.

    """
    commands: list[list[str]] = []

    resolver_packages = [p for p in plan.packages if p != "geosam"]
    if resolver_packages:
        cmd = _build_uv_command_prefix(plan)
        cmd.extend(requirement_for(p) for p in resolver_packages)
        commands.append(cmd)

    if "geosam" in plan.packages:
        geosam_cmd = [
            str(plan.uv_binary),
            "pip",
            "install",
            "--target",
            str(plan.target_path),
            "--no-config",
            "--cache-dir",
            str(UV_CACHE_DIR),
            "--no-python-downloads",
            "--strict",
            "--upgrade",
            "--no-deps",
        ]
        geosam_cmd.extend(resolve_uv_python_flags())
        geosam_cmd.append(requirement_for("geosam"))
        commands.append(geosam_cmd)

    return commands


def build_full_commands(plan: DependencyPlan) -> list[list[str]]:
    """Build complete commands including the one-time uv installation.

    This produces a two-step command sequence:
    1. ``pip install uv`` (one-time setup)
    2. ``uv pip install <deps>`` (dependency installation)

    Used by the "Copy Full Command" button so users can reproduce the
    installation from a terminal.

    Parameters
    ----------
    plan : DependencyPlan
        Installation plan. ``plan.python_path`` is used for the ``pip install
        uv`` step; the uv dependency step relies on
        :func:`resolve_uv_python_flags` instead of ``--python``.

    Returns
    -------
    list[list[str]]
        List of command argument lists.

    """
    from .uv_manager import MIN_UV, UV_PKG_DIR

    # Step 1: pip install uv (uses the QGIS interpreter directly)
    pip_install_uv = [
        str(plan.python_path),
        "-m",
        "pip",
        "install",
        "--target",
        str(UV_PKG_DIR),
        "--upgrade",
        "--only-binary",
        ":all:",
        f"uv>={MIN_UV}",
    ]

    # Step 2: uv pip install deps
    uv_commands = build_uv_commands(plan)

    return [pip_install_uv, *uv_commands]


def make_plan(
    packages: tuple[str, ...],
    profile: TorchProfile | None = None,
    *,
    custom_index_url: str | None = None,
) -> DependencyPlan:
    """Create a dependency installation plan.

    Parameters
    ----------
    packages : tuple[str, ...]
        Package names to install.
    profile : TorchProfile | None, optional
        Torch profile. When ``None``, the active profile is used.
    custom_index_url : str | None, optional
        Custom index URL for the ``custom`` profile. When ``None``,
        the persisted URL is loaded.

    Returns
    -------
    DependencyPlan
        Complete installation plan.

    Raises
    ------
    RuntimeError
        Raised when uv cannot be installed or located.

    """
    from .plugin_settings import (
        _create_qgis_runtime_constraints_file,
        resolve_python_interpreter,
    )
    from .uv_manager import ensure_uv

    key = _runtime_dependency_key()

    if profile is None:
        profile = active_profile(key)

    if custom_index_url is None and profile == "custom":
        custom_index_url = get_custom_index_url(key)

    uv_binary = ensure_uv()
    python_path = resolve_python_interpreter()
    target_path = profile_target_path(key, profile)
    constraints_path = _create_qgis_runtime_constraints_file(profile=profile)

    return DependencyPlan(
        packages=packages,
        python_path=python_path,
        target_path=target_path,
        constraints_path=constraints_path,
        profile=profile,
        uv_binary=uv_binary,
        custom_index_url=custom_index_url,
    )


def make_plan_for_copy(
    packages: tuple[str, ...],
    profile: TorchProfile | None = None,
    *,
    custom_index_url: str | None = None,
) -> DependencyPlan:
    """Create a dependency plan for display or clipboard copy only.

    Unlike :func:`make_plan`, this does **not** call :func:`ensure_uv` and
    will not download or install uv. The ``uv_binary`` field is set to a
    placeholder ``Path("uv")`` so the generated command is runnable in a
    terminal where the user's own uv (or a system uv) is on PATH.

    Parameters
    ----------
    packages : tuple[str, ...]
        Package names to install.
    profile : TorchProfile | None, optional
        Torch profile. When ``None``, the active profile is used.
    custom_index_url : str | None, optional
        Custom index URL for the ``custom`` profile. When ``None``,
        the persisted URL is loaded.

    Returns
    -------
    DependencyPlan
        Installation plan with a placeholder ``uv_binary``.

    """
    from .plugin_settings import (
        _create_qgis_runtime_constraints_file,
        resolve_python_interpreter,
    )

    key = _runtime_dependency_key()

    if profile is None:
        profile = active_profile(key)

    if custom_index_url is None and profile == "custom":
        custom_index_url = get_custom_index_url(key)

    python_path = resolve_python_interpreter()
    target_path = profile_target_path(key, profile)
    constraints_path = _create_qgis_runtime_constraints_file(profile=profile)

    return DependencyPlan(
        packages=packages,
        python_path=python_path,
        target_path=target_path,
        constraints_path=constraints_path,
        profile=profile,
        uv_binary=Path("uv"),
        custom_index_url=custom_index_url,
    )


def install_dependencies(
    packages: tuple[str, ...] | None = None,
    profile: TorchProfile | None = None,
    *,
    custom_index_url: str | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Install Geo-SAM dependencies using uv.

    Parameters
    ----------
    packages : tuple[str, ...] | None, optional
        Package names to install. When ``None``, all known plugin
        dependencies are installed.
    profile : TorchProfile | None, optional
        Torch profile. When ``None``, the active profile is used.
    custom_index_url : str | None, optional
        Custom index URL for the ``custom`` profile.
    log_callback : Callable[[str], None] | None, optional
        Callback that receives incremental install log lines.

    Returns
    -------
    tuple[bool, str]
        Installation success flag and combined command output.

    """
    from .plugin_settings import DEPENDENCY_INSTALL_REQUIREMENTS, format_dependency_install_command

    if packages is None:
        packages = tuple(DEPENDENCY_INSTALL_REQUIREMENTS.keys())

    try:
        plan = make_plan(packages, profile, custom_index_url=custom_index_url)
    except RuntimeError as exc:
        logger.error("Dependency installation aborted: %s", exc)
        return False, str(exc)

    commands = build_uv_commands(plan)
    output_lines: list[str] = []

    for command in commands:
        if log_callback is not None:
            log_callback(f"Command: {format_dependency_install_command(command)}")

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            msg = f"Failed to start uv: {exc}"
            logger.error(msg)
            output_lines.append(msg)
            return False, "\n".join(output_lines)

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
