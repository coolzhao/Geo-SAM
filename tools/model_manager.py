"""Model catalog and checkpoint management for Geo-SAM."""

from __future__ import annotations

import gc
import logging
import sys
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .geosam_backend import configure_geosam_qgis_runtime
from .plugin_settings import get_model_directory

if TYPE_CHECKING:
    from geosam.models import ModelSpec

logger = logging.getLogger(__name__)

_DATACLASS_SLOTS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}
_FROZEN_DATACLASS_KWARGS = {"frozen": True, **_DATACLASS_SLOTS_KWARGS}
SAM3_EFFECTIVE_IMGSZ = (1008, 1008)

# download URLs from modelscope
# This is used as a fallback when Ultralytics downloads are unavailable or fail.
DIRECT_MODEL_DOWNLOAD_SOURCES: dict[str, str] = {
    # SAM3
    "sam3": "https://modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt",
    "sam3.1_multiplex": (
        "https://modelscope.cn/models/facebook/sam3.1/resolve/master/"
        "sam3.1_multiplex.pt"
    ),
    # SAM2
    "sam2_t": "https://modelscope.cn/models/facebook/sam2-hiera-tiny/resolve/master/sam2_hiera_tiny.pt",
    "sam2_s": "https://modelscope.cn/models/facebook/sam2-hiera-small/resolve/master/sam2_hiera_small.pt",
    "sam2_b": "https://modelscope.cn/models/facebook/sam2-hiera-base-plus/resolve/master/sam2_hiera_base_plus.pt",
    "sam2_l": "https://modelscope.cn/models/facebook/sam2-hiera-large/resolve/master/sam2_hiera_large.pt",
    # SAM2.1
    "sam2.1_t": "https://modelscope.cn/models/facebook/sam2.1-hiera-tiny/resolve/master/sam2.1_hiera_tiny.pt",
    "sam2.1_s": "https://modelscope.cn/models/facebook/sam2.1-hiera-small/resolve/master/sam2.1_hiera_small.pt",
    "sam2.1_b": "https://modelscope.cn/models/facebook/sam2.1-hiera-base-plus/resolve/master/sam2.1_hiera_base_plus.pt",
    "sam2.1_l": "https://modelscope.cn/models/facebook/sam2.1-hiera-large/resolve/master/sam2.1_hiera_large.pt",
    # SAM (legacy)
    "sam_b": "https://modelscope.cn/models/yatengLG/ISAT_with_segment_anything_checkpoints/resolve/master/checkpoints/sam_vit_b_01ec64.pth",
    "sam_l": "https://modelscope.cn/models/yatengLG/ISAT_with_segment_anything_checkpoints/resolve/master/checkpoints/sam_vit_l_0b3195.pth",
}
ULTRALYTICS_DOWNLOAD_MODEL_IDS = {
    "sam_b",
    "sam_l",
    "sam2_t",
    "sam2_s",
    "sam2_b",
    "sam2_l",
    "sam2.1_t",
    "sam2.1_s",
    "sam2.1_b",
    "sam2.1_l",
}


@dataclass(**_FROZEN_DATACLASS_KWARGS)
class ModelDefinition:
    """Supported downloadable model definition.

    Parameters
    ----------
    model_id : str
        Stable GeoSAM model identifier.
    label : str
        Human-readable model name.
    model_type : {"sam", "sam2", "sam3"}
        Underlying model family.
    filename : str
        Expected checkpoint filename.
    supports_feature_reuse : bool, optional
        Whether encoded feature reuse is supported for the model.

    """

    model_id: str
    label: str
    model_type: Literal["sam", "sam2", "sam3"]
    filename: str
    supports_feature_reuse: bool = True


MODEL_DEFINITIONS: tuple[ModelDefinition, ...] = (
    ModelDefinition("sam_b", "SAM Base", "sam", "sam_b.pt"),
    ModelDefinition("sam_l", "SAM Large", "sam", "sam_l.pt"),
    ModelDefinition("sam2_t", "SAM2 Tiny", "sam2", "sam2_t.pt"),
    ModelDefinition("sam2_s", "SAM2 Small", "sam2", "sam2_s.pt"),
    ModelDefinition("sam2_b", "SAM2 Base", "sam2", "sam2_b.pt"),
    ModelDefinition("sam2_l", "SAM2 Large", "sam2", "sam2_l.pt"),
    ModelDefinition("sam2.1_t", "SAM2.1 Tiny", "sam2", "sam2.1_t.pt"),
    ModelDefinition("sam2.1_s", "SAM2.1 Small", "sam2", "sam2.1_s.pt"),
    ModelDefinition("sam2.1_b", "SAM2.1 Base", "sam2", "sam2.1_b.pt"),
    ModelDefinition("sam2.1_l", "SAM2.1 Large", "sam2", "sam2.1_l.pt"),
    ModelDefinition(
        "sam3",
        "SAM3",
        "sam3",
        "sam3.pt",
        # supports_feature_reuse=False,
    ),
    # The SAM3.1 Multiplex is currently not supported by Ultralytics and yeilds 
    # wrong segmentation results when used with the SAM3 inference code.
    
    # ModelDefinition(
    #     "sam3.1_multiplex",
    #     "SAM3.1 Multiplex",
    #     "sam3",
    #     "sam3.1_multiplex.pt",
    #     # supports_feature_reuse=False,
    # ),
)


def get_model_definition(model_id: str) -> ModelDefinition:
    """Return a model definition by id.

    Parameters
    ----------
    model_id : str
        Registered model identifier.

    Returns
    -------
    ModelDefinition
        Matching model metadata.

    Raises
    ------
    KeyError
        Raised when the model id is unknown.

    """
    for definition in MODEL_DEFINITIONS:
        if definition.model_id == model_id:
            return definition
    logger.error("Unknown GeoSAM model id requested: %s", model_id)
    raise KeyError(model_id)


def get_model_checkpoint_path(model_id: str) -> Path:
    """Return the local checkpoint path for a model.

    Parameters
    ----------
    model_id : str
        Registered model identifier.

    Returns
    -------
    Path
        Local checkpoint path for the model.

    """
    definition = get_model_definition(model_id)
    return get_model_directory() / definition.filename


def effective_model_imgsz(model_id: str) -> tuple[int, int]:
    """Return the plugin-normalized model image size.

    Parameters
    ----------
    model_id : str
        Registered model identifier.

    Returns
    -------
    tuple[int, int]
        Height and width used for model-ready chips.

    """
    definition = get_model_definition(model_id)
    if definition.model_type == "sam3":
        return SAM3_EFFECTIVE_IMGSZ
    return (1024, 1024)


def get_model_display_items() -> list[tuple[str, str]]:
    """Return selectable model entries for UI comboboxes.

    Returns
    -------
    list[tuple[str, str]]
        Model id and label pairs.

    """
    return [(definition.model_id, definition.label) for definition in MODEL_DEFINITIONS]


def get_model_status_rows() -> list[dict[str, Any]]:
    """Return model download status rows.

    Returns
    -------
    list[dict[str, Any]]
        Model status dictionaries for the settings UI.

    """
    rows: list[dict[str, Any]] = []
    for definition in MODEL_DEFINITIONS:
        checkpoint_path = get_model_checkpoint_path(definition.model_id)
        rows.append({
            "model_id": definition.model_id,
            "label": definition.label,
            "model_type": definition.model_type,
            "downloaded": checkpoint_path.exists(),
            "path": checkpoint_path,
        })
    return rows


def infer_model_id_from_checkpoint_path(
    checkpoint_path: str | Path,
    *,
    fallback_model_id: str | None = None,
    allow_unknown: bool = False,
) -> str | None:
    """Infer a registered model id from a checkpoint file path.

    Parameters
    ----------
    checkpoint_path : str | Path
        Checkpoint file path.
    fallback_model_id : str | None, optional
        Fallback model id to return when inference fails.
    allow_unknown : bool, optional
        Whether to return ``None`` instead of raising an error.

    Returns
    -------
    str | None
        Inferred model id.

    Raises
    ------
    ValueError
        Raised when inference fails and ``allow_unknown`` is ``False``.

    """
    checkpoint_name = Path(checkpoint_path).name.lower()
    for definition in MODEL_DEFINITIONS:
        if checkpoint_name == definition.filename.lower():
            return definition.model_id
    for definition in MODEL_DEFINITIONS:
        stem = Path(definition.filename).stem.lower()
        if stem in checkpoint_name:
            return definition.model_id
    if fallback_model_id is not None:
        return fallback_model_id
    if allow_unknown:
        logger.warning(
            "Could not infer a GeoSAM model id from checkpoint path: %s",
            checkpoint_path,
        )
        return None
    msg = (
        "Could not infer a GeoSAM model id from the checkpoint path. "
        "Please select a supported model explicitly."
    )
    logger.error(msg)
    raise ValueError(msg)


def create_model_spec(model_id: str) -> ModelSpec:
    """Create a GeoSAM model spec for a registered local checkpoint.

    Parameters
    ----------
    model_id : str
        Registered model identifier.

    Returns
    -------
    ModelSpec
        Runtime model specification.

    Raises
    ------
    FileNotFoundError
        Raised when the checkpoint is missing locally.

    """
    configure_geosam_qgis_runtime()
    from geosam.runtime import create_model_spec as create_runtime_model_spec

    checkpoint_path = get_model_checkpoint_path(model_id)
    if not checkpoint_path.exists():
        logger.error("Model checkpoint is missing: %s", checkpoint_path)
        raise FileNotFoundError(checkpoint_path)
    model_spec = create_runtime_model_spec(model_id, checkpoint_path)
    definition = get_model_definition(model_id)
    if definition.model_type == "sam3":
        return replace(model_spec, imgsz=SAM3_EFFECTIVE_IMGSZ)
    return model_spec


def create_model_spec_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_id: str | None = None,
    device: str | None = None,
) -> ModelSpec:
    """Create a model spec from an arbitrary checkpoint path.

    Parameters
    ----------
    checkpoint_path : str | Path
        Checkpoint file path.
    model_id : str | None, optional
        Optional fallback model id.
    device : str | None, optional
        Optional inference device string.

    Returns
    -------
    ModelSpec
        Runtime model specification.

    """
    configure_geosam_qgis_runtime()
    from geosam.runtime import create_model_spec_from_checkpoint as create_runtime_spec

    model_spec = create_runtime_spec(
        checkpoint_path,
        model_id=model_id,
        device=device,
    )
    if model_spec.model_type == "sam3":
        return replace(model_spec, imgsz=SAM3_EFFECTIVE_IMGSZ)
    return model_spec


def _download_model_with_ultralytics(model_id: str, target_path: Path) -> bool:
    """Download a supported model checkpoint through Ultralytics.

    Parameters
    ----------
    model_id : str
        Registered model identifier.
    target_path : Path
        Local checkpoint destination.

    Returns
    -------
    bool
        ``True`` when Ultralytics downloaded the checkpoint successfully.

    """
    if model_id not in ULTRALYTICS_DOWNLOAD_MODEL_IDS:
        return False

    try:
        from ultralytics import SAM
    except Exception as exc:
        logger.warning(
            "Ultralytics download is unavailable for %s: %s",
            model_id,
            exc,
        )
        return False

    model: Any | None = None
    try:
        model = SAM(str(target_path))
    except Exception as exc:
        logger.warning(
            "Ultralytics failed to download checkpoint for %s: %s",
            model_id,
            exc,
        )
        return False
    finally:
        del model
        gc.collect()

    if target_path.exists():
        return True

    logger.warning(
        "Ultralytics completed without creating the expected checkpoint: %s",
        target_path,
    )
    return False


def download_model(model_id: str) -> Path:
    """Download or copy a model checkpoint into the local store.

    Parameters
    ----------
    model_id : str
        Registered model identifier.

    Returns
    -------
    Path
        Downloaded checkpoint path.

    Raises
    ------
    ValueError
        Raised when no download source exists for the requested model.

    """
    definition = get_model_definition(model_id)
    target_path = get_model_checkpoint_path(model_id)
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if _download_model_with_ultralytics(model_id, target_path):
        return target_path

    source = DIRECT_MODEL_DOWNLOAD_SOURCES.get(definition.model_id)
    if source is None:
        msg = (
            f"GeoSAM does not have a configured download source for model '{model_id}'."
        )
        logger.error(msg)
        raise ValueError(msg)

    download_target = target_path.with_suffix(f"{target_path.suffix}.download")
    try:
        urllib.request.urlretrieve(source, download_target)
        download_target.replace(target_path)
    except Exception as exc:
        if download_target.exists():
            download_target.unlink()
        logger.error(
            "Failed to download model checkpoint for %s from %s: %s",
            model_id,
            source,
            exc,
        )
        raise
    return target_path


def delete_model(model_id: str) -> None:
    """Delete a locally downloaded model checkpoint.

    Parameters
    ----------
    model_id : str
        Registered model identifier.

    """
    from .geosam_runtime import release_runtime_models

    release_runtime_models(model_id=model_id)
    checkpoint_path = get_model_checkpoint_path(model_id)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
