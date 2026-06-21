"""GeoSAM runtime backend helpers for the QGIS plugin."""

from __future__ import annotations

import logging
from typing import Any

from qgis.core import QgsProject

logger = logging.getLogger(__name__)


def configure_geosam_qgis_runtime(
    *,
    feedback: Any | None = None,
    task: Any | None = None,
    transform_context: Any | None = None,
) -> bool:
    """Configure GeoSAM to use QGIS runtime services.

    Parameters
    ----------
    feedback : Any | None, optional
        QGIS processing feedback object used for progress and cancellation.
    task : Any | None, optional
        QGIS task object used for progress and cancellation.
    transform_context : Any | None, optional
        Coordinate transform context to pass to GeoSAM CRS operations. When
        omitted, the current project transform context is used.

    Returns
    -------
    bool
        ``True`` when GeoSAM was available and configured, otherwise ``False``.

    """
    try:
        from geosam.context import configure_runtime
    except ModuleNotFoundError:
        logger.warning("GeoSAM is not installed; QGIS runtime was not configured.")
        return False

    project = QgsProject.instance()
    resolved_transform_context = transform_context or project.transformContext()
    configure_runtime(
        "qgis",
        qgis_project=project,
        qgis_transform_context=resolved_transform_context,
        qgis_feedback=feedback,
        qgis_task=task,
    )
    return True
