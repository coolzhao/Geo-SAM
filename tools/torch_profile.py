"""Torch profile configuration for Geo-SAM dependency installation.

This module maps torch backend profiles to uv flags, pip index URLs,
and platform-specific availability constraints.
"""

from __future__ import annotations

import platform
import shutil
import subprocess

from .runtime_profile import TorchProfile

# PyTorch download index URLs for CUDA profiles
PYTORCH_INDEX_URLS: dict[str, str] = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu130": "https://download.pytorch.org/whl/cu130",
}


def uv_torch_flag(profile: TorchProfile) -> list[str]:
    """Return uv ``--torch-backend`` flags for a profile.

    Parameters
    ----------
    profile : TorchProfile
        Torch profile identifier.

    Returns
    -------
    list[str]
        uv command-line flags for the torch backend. Empty for ``cpu``
        (uv defaults to CPU) and ``custom`` (uses ``--index-url``).

    """
    if profile == "cpu":
        return []
    if profile == "auto":
        return ["--torch-backend", "auto"]
    if profile in ("cu128", "cu130"):
        return ["--torch-backend", profile]
    # custom → uses --index-url instead
    return []


def pip_index_url_for_profile(
    profile: TorchProfile,
    custom_url: str | None = None,
) -> str | None:
    """Return the PyPI index URL for a torch profile.

    Parameters
    ----------
    profile : TorchProfile
        Torch profile identifier.
    custom_url : str | None, optional
        Custom index URL, used only when ``profile`` is ``"custom"``.

    Returns
    -------
    str | None
        Index URL for the profile, or ``None`` for default PyPI.

    """
    if profile == "custom":
        return custom_url or None
    return PYTORCH_INDEX_URLS.get(profile)


def profile_allowed(profile: TorchProfile) -> bool:
    """Check whether a profile is allowed on the current platform.

    Parameters
    ----------
    profile : TorchProfile
        Torch profile identifier.

    Returns
    -------
    bool
        ``True`` if the profile is available on this platform. On macOS,
        CUDA profiles (cu128, cu130) are not allowed.

    """
    if platform.system() == "Darwin" and profile in {"cu128", "cu130"}:
        return False
    return True


def profile_unavailable_reason(profile: TorchProfile) -> str | None:
    """Return a human-readable reason why a profile is unavailable.

    Parameters
    ----------
    profile : TorchProfile
        Torch profile identifier.

    Returns
    -------
    str | None
        Reason text, or ``None`` if the profile is available.

    """
    if platform.system() == "Darwin" and profile in {"cu128", "cu130"}:
        return "macOS does not support CUDA"
    return None


def detect_recommended_profile() -> TorchProfile:
    """Detect the recommended torch profile for the current system.

    Returns
    -------
    TorchProfile
        Recommended profile based on available GPU hardware.

    """
    if platform.system() == "Darwin":
        # macOS: no CUDA, auto will install CPU torch
        return "auto"

    # Try nvidia-smi to detect CUDA capability
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return "auto"

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return "auto"

        # Parse driver version to determine CUDA capability
        driver_version = result.stdout.strip().split("\n")[0].strip()
        if not driver_version:
            return "auto"

        # Driver version mapping (approximate):
        # Driver >= 560.x → CUDA 12.8 support
        # Driver >= 570.x → CUDA 13.0 support
        parts = driver_version.split(".")
        if len(parts) >= 1:
            major = int(parts[0])
            if major >= 570:
                return "cu130"
            if major >= 560:
                return "cu128"

    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass

    return "auto"
