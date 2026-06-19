from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import cast

from qgis.PyQt import uic
from qgis.PyQt.QtCore import PYQT_VERSION
from qgis.PyQt.QtWidgets import QWidget
from qgis.gui import QgsDockWidget

UI_DIRECTORY = Path(__file__).parent
SELECTOR_PATH = UI_DIRECTORY / "Selector.ui"
ENCODER_PATH = UI_DIRECTORY / "EncoderCopilot.ui"
PYQT6_UI_ENUM_REPLACEMENTS = {
    "QDockWidget::AllDockWidgetFeatures": (
        "QDockWidget::DockWidgetFeature::DockWidgetClosable|"
        "QDockWidget::DockWidgetFeature::DockWidgetMovable|"
        "QDockWidget::DockWidgetFeature::DockWidgetFloatable"
    ),
    "Qt::Horizontal": "Qt::Orientation::Horizontal",
    "Qt::AlignCenter": "Qt::AlignmentFlag::AlignCenter",
    "Qt::AlignHCenter": "Qt::AlignmentFlag::AlignHCenter",
    "Qt::AlignTop": "Qt::AlignmentFlag::AlignTop",
}


def _load_ui(path: Path) -> QgsDockWidget:
    """Load a Qt Designer dock widget in PyQt5 or PyQt6.

    Parameters
    ----------
    path : Path
        Qt Designer ``.ui`` file to load.

    Returns
    -------
    QgsDockWidget
        Loaded QGIS dock widget.

    Notes
    -----
    PyQt6 requires fully scoped enum names in UI XML, while PyQt5's dynamic
    loader expects the legacy names. The source UI stays compatible with
    PyQt5 and is translated in memory only when running under PyQt6.
    """
    if PYQT_VERSION < 0x060000:
        return cast(QgsDockWidget, uic.loadUi(str(path)))

    ui_text = path.read_text(encoding="utf-8")
    for legacy_name, scoped_name in PYQT6_UI_ENUM_REPLACEMENTS.items():
        ui_text = ui_text.replace(legacy_name, scoped_name)
    return cast(QgsDockWidget, uic.loadUi(StringIO(ui_text)))


def load_selector_ui(parent: QWidget | None = None) -> QgsDockWidget:
    """Create a fresh selector dock widget instance."""
    widget = _load_ui(SELECTOR_PATH)
    if parent is not None:
        widget.setParent(parent)
    return widget


def load_encoder_copilot_ui(parent: QWidget | None = None) -> QgsDockWidget:
    """Create a fresh encoder-copilot dock widget instance."""
    widget = _load_ui(ENCODER_PATH)
    if parent is not None:
        widget.setParent(parent)
    return widget
