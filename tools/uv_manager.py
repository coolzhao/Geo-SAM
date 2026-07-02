"""Private uv binary management for Geo-SAM dependency installation.

This module handles locating, version-checking, and installing a private
uv binary used as the dependency installation engine. The binary is
installed into ``.deps/uv-pkg/`` via pip and is never added to the
system PATH.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from .dependency_path import PLUGIN_DEPENDENCY_ROOT

logger = logging.getLogger(__name__)

MIN_UV_VERSION = (0, 6, 0)
MIN_UV = "0.6.0"
UV_PKG_DIR = PLUGIN_DEPENDENCY_ROOT / "uv-pkg"
UV_CACHE_DIR = PLUGIN_DEPENDENCY_ROOT / "uv-cache"


def ensure_uv() -> Path:
    """Ensure a private uv binary is available and return its path.

    This function checks for an existing uv installation in
    :data:`UV_PKG_DIR`, verifies the version meets :data:`MIN_UV_VERSION`,
    and installs uv via pip if needed.

    Returns
    -------
    Path
        Path to the uv binary executable.

    Raises
    ------
    RuntimeError
        Raised when uv cannot be located or installed.

    """
    binary = _locate_uv_binary(UV_PKG_DIR)
    if binary is not None and version_ok(binary):
        return binary

    _install_private_uv()

    binary = _locate_uv_binary(UV_PKG_DIR)
    if binary is None:
        msg = (
            "uv binary not found after pip installation. "
            f"Expected location: {UV_PKG_DIR}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    if not version_ok(binary):
        version_text = _uv_version_text(binary)
        msg = (
            f"Installed uv version {version_text} does not meet "
            f"minimum requirement {MIN_UV}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    return binary


def _locate_uv_binary(pkg_dir: Path) -> Path | None:
    """Locate the uv binary within a pip-installed package directory.

    Parameters
    ----------
    pkg_dir : Path
        Directory where uv was installed via ``pip --target``.

    Returns
    -------
    Path | None
        Path to the uv binary, or ``None`` if not found.

    """
    if not pkg_dir.exists():
        return None

    # Try the uv Python package's _find_uv helper first
    pkg_str = str(pkg_dir)
    if pkg_str not in sys.path:
        sys.path.insert(0, pkg_str)
        inserted = True
    else:
        inserted = False
    try:
        from uv import _find_uv

        found = Path(_find_uv.find_uv())
        if found.exists():
            logger.debug("Located uv binary via _find_uv: %s", found)
            return found
    except (ImportError, AttributeError, OSError) as exc:
        logger.debug("Could not locate uv via _find_uv: %s", exc)
    finally:
        if inserted:
            try:
                sys.path.remove(pkg_str)
            except ValueError:
                pass

    # Fallback: scan common binary locations
    candidate_names = ["uv", "uv.exe"]
    search_dirs = [pkg_dir / "bin", pkg_dir / "Scripts", pkg_dir]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for candidate_name in candidate_names:
            candidate = search_dir / candidate_name
            if candidate.is_file() and _is_executable(candidate):
                logger.debug("Located uv binary via fallback scan: %s", candidate)
                return candidate

    return None


def _is_executable(path: Path) -> bool:
    """Check whether a file is executable.

    Parameters
    ----------
    path : Path
        File path to check.

    Returns
    -------
    bool
        ``True`` if the file is executable (on Unix) or exists (on Windows).

    """
    import stat

    if sys.platform == "win32":
        return path.is_file()
    return path.is_file() and bool(path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


def _uv_version_text(binary: Path) -> str | None:
    """Return the raw version text from a uv binary.

    Parameters
    ----------
    binary : Path
        Path to the uv binary.

    Returns
    -------
    str | None
        Version string (e.g., ``"0.7.4"``), or ``None`` on failure.

    """
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        # Output format: "uv 0.7.4 (abc1234 2024-01-01)"
        match = re.search(r"uv\s+(\d+\.\d+\.\d+)", result.stdout)
        if match:
            return match.group(1)
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("Could not get uv version from %s: %s", binary, exc)
    return None


def _parse_version(version_text: str) -> tuple[int, ...] | None:
    """Parse a version string into a comparable tuple.

    Parameters
    ----------
    version_text : str
        Version string (e.g., ``"0.7.4"``).

    Returns
    -------
    tuple[int, ...] | None
        Parsed version tuple, or ``None`` on parse failure.

    """
    try:
        return tuple(int(part) for part in version_text.split("."))
    except (ValueError, AttributeError):
        return None


def _uv_version(binary: Path) -> tuple[int, ...] | None:
    """Return the parsed version of a uv binary.

    Parameters
    ----------
    binary : Path
        Path to the uv binary.

    Returns
    -------
    tuple[int, ...] | None
        Parsed version tuple, or ``None`` if the version cannot be determined.

    """
    text = _uv_version_text(binary)
    if text is None:
        return None
    return _parse_version(text)


def version_ok(binary: Path) -> bool:
    """Check whether a uv binary meets the minimum version requirement.

    Parameters
    ----------
    binary : Path
        Path to the uv binary.

    Returns
    -------
    bool
        ``True`` if the binary version is >= :data:`MIN_UV_VERSION`.

    """
    version = _uv_version(binary)
    if version is None:
        logger.warning("Could not determine uv version for %s", binary)
        return False
    return version >= MIN_UV_VERSION


def _install_private_uv() -> None:
    """Install a private uv binary via pip into :data:`UV_PKG_DIR`.

    Raises
    ------
    RuntimeError
        Raised when the pip installation fails.

    """
    # Import here to avoid circular dependency at module level
    from .plugin_settings import resolve_python_interpreter

    python = resolve_python_interpreter()
    UV_PKG_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(python),
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

    logger.info("Installing private uv: %s", " ".join(cmd))

    try:
        env = os.environ.copy()
        env["PYTHONHOME"] = sys.prefix
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"uv installation timed out after {exc.timeout}s"
        logger.error(msg)
        raise RuntimeError(msg) from exc
    except OSError as exc:
        msg = f"Failed to run pip for uv installation: {exc}"
        logger.error(msg)
        raise RuntimeError(msg) from exc

    if result.returncode != 0:
        output = (result.stdout + "\n" + result.stderr).strip()
        msg = f"pip install uv failed (exit code {result.returncode}):\n{output}"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Successfully installed private uv to %s", UV_PKG_DIR)
