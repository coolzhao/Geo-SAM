"""User-level uv binary management for Geo-SAM dependency installation.

This module locates, version-checks, and installs the ``uv`` binary used as
the dependency installation engine. The lookup order is:

1. ``uv`` on ``PATH`` (system-installed: Homebrew, pipx, curl, winget, …)
2. User-level binary at ``~/.local/bin/uv`` (Unix) or
   ``%APPDATA%\\Python\\Scripts\\uv.exe`` (Windows) — installed by this
   plugin or shared with other tools
3. ``pip install`` into a temporary directory, then copy the binary to the
   user-level location above

The cache is intentionally *not* configured: ``uv`` uses its own platform
default (``~/Library/Caches/uv`` on macOS, ``~/.cache/uv`` on Linux,
``%LOCALAPPDATA%\\uv\\cache`` on Windows), so caches are shared across all
QGIS plugins and the user's terminal.

Path sources (not guesses):

- Cache locations: https://docs.astral.sh/uv/concepts/cache/
- Installer locations: https://docs.astral.sh/uv/reference/installer/
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_UV_VERSION = (0, 6, 0)
MIN_UV = "0.6.0"


def _user_bin_dir() -> Path:
    """Return the user-level executable directory.

    Returns
    -------
    Path
        Directory that holds user-installed executables. Follows the
        ``uv`` / ``pipx`` convention of ``~/.local/bin`` on Unix, and the
        ``pip --user`` Scripts directory on Windows.

    """
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(appdata) / "Python" / "Scripts"
    return Path.home() / ".local" / "bin"


def _uv_binary_name() -> str:
    """Return the uv binary filename for the current platform."""
    return "uv.exe" if sys.platform == "win32" else "uv"


def _user_uv_binary() -> Path:
    """Return the expected path of the user-level uv binary."""
    return _user_bin_dir() / _uv_binary_name()


def _pip_target_for_uv() -> Path:
    """Return a temporary pip ``--target`` directory used during install.

    The directory lives under the system temp dir so it never occupies
    plugin space; it is removed after the binary is copied to the user
    bin directory.

    """
    return Path(tempfile.gettempdir()) / "qgis-geo-sam-uv-install"


def ensure_uv() -> Path:
    """Ensure a usable ``uv`` binary is available and return its path.

    Lookup order (first usable hit wins):

    1. ``uv`` found on ``PATH`` via :func:`shutil.which` — covers
       Homebrew, pipx, curl, and winget installs.
    2. The user-level binary at :func:`_user_uv_binary` — installed
       previously by this plugin or another tool.
    3. Install on demand: ``pip install`` to a temp dir, then copy the
       binary to the user-level location.

    Returns
    -------
    Path
        Path to the uv binary executable.

    Raises
    ------
    RuntimeError
        Raised when uv cannot be located, installed, or does not meet
        :data:`MIN_UV_VERSION`.

    """
    _augment_path_for_system_uv()

    # 1. System uv on PATH (Homebrew / pipx / curl / winget)
    system_uv = _find_system_uv()
    if system_uv is not None and version_ok(system_uv):
        logger.info("Using system uv from PATH: %s", system_uv)
        return system_uv
    if system_uv is not None:
        logger.info(
            "System uv at %s is too old or version unknown; will install a fresh one",
            system_uv,
        )

    # 2. User-level binary installed by a previous run / another tool
    user_uv = _user_uv_binary()
    if user_uv.exists() and version_ok(user_uv):
        logger.info("Using user-level uv: %s", user_uv)
        return user_uv

    # 3. Install into the user-level bin directory
    logger.info("No usable uv found, installing to %s", user_uv)
    _install_uv_to_user_bin()

    if not user_uv.exists():
        msg = f"uv binary not found after installation; expected at {user_uv}"
        logger.error(msg)
        raise RuntimeError(msg)
    if not version_ok(user_uv):
        version_text = _uv_version_text(user_uv)
        msg = (
            f"Installed uv version {version_text} does not meet "
            f"minimum requirement {MIN_UV}"
        )
        logger.error(msg)
        raise RuntimeError(msg)
    return user_uv


def _augment_path_for_system_uv() -> None:
    """Prepend common uv install directories to ``PATH``.

    QGIS on macOS, when launched from the Dock, inherits a minimal
    ``PATH`` that omits Homebrew. Other platforms are mostly fine, but
    prepending the user bin directory is harmless and helps headless
    QGIS launches.

    """
    if sys.platform == "darwin":
        extra = ["/opt/homebrew/bin", "/usr/local/bin", str(Path.home() / ".local" / "bin")]
    elif sys.platform.startswith("linux"):
        extra = [str(Path.home() / ".local" / "bin"), "/usr/local/bin"]
    else:
        extra = [str(_user_bin_dir())]

    current = os.environ.get("PATH", "")
    seen = set(current.split(os.pathsep))
    for candidate in extra:
        if candidate and candidate not in seen:
            os.environ["PATH"] = candidate + os.pathsep + current
            seen.add(candidate)
            current = os.environ.get("PATH", "")


def _find_system_uv() -> Path | None:
    """Return the first ``uv`` executable found on ``PATH``.

    Uses :func:`shutil.which`, which honors ``PATHEXT`` on Windows so
    ``uv.exe`` is resolved transparently.

    Returns
    -------
    Path | None
        Path to the uv binary, or ``None`` if not on ``PATH``.

    """
    found = shutil.which("uv")
    return Path(found) if found else None


def _install_uv_to_user_bin() -> None:
    """Install ``uv`` via pip and copy the binary to the user bin dir.

    A private ``pip install --target`` is run against the QGIS Python
    interpreter into a temporary directory; the resulting binary is then
    copied to :func:`_user_uv_binary`. This avoids polluting user
    ``site-packages`` (no ``pip --user``) and avoids coupling the
    install location to a plugin-private directory.

    Raises
    ------
    RuntimeError
        Raised when pip fails or the binary cannot be located after
        installation.

    Notes
    -----
    On macOS, ``PYTHONHOME`` is set to ``sys.prefix`` to work around the
    isolated-interpreter probe issue described in
    :func:`dependency_installer.resolve_uv_python_flags`.

    """
    from .plugin_settings import resolve_python_interpreter

    python = resolve_python_interpreter()
    target = _pip_target_for_uv()
    target.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(python),
        "-m",
        "pip",
        "install",
        "--target",
        str(target),
        "--upgrade",
        "--only-binary",
        ":all:",
        f"uv>={MIN_UV}",
    ]

    logger.info("Installing uv: %s", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONHOME"] = sys.prefix

    try:
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

    bin_subdir = "Scripts" if sys.platform == "win32" else "bin"
    installed_binary = target / bin_subdir / _uv_binary_name()
    if not installed_binary.exists():
        msg = (
            f"uv binary not found after pip install. "
            f"Expected: {installed_binary}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    dest = _user_uv_binary()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(installed_binary, dest)
    if sys.platform != "win32":
        dest.chmod(0o755)

    shutil.rmtree(target, ignore_errors=True)
    logger.info("Successfully installed uv to %s", dest)


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
        Parsed version tuple, or ``None`` if the version cannot be
        determined.

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
