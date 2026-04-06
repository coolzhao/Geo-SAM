import os
from typing import cast

from PyQt5 import uic
from qgis.gui import QgsDockWidget

try:
    from qgis.PyQt.QtGui import QDockWidget, QWidget
except Exception:
    from qgis.PyQt.QtWidgets import QDockWidget, QWidget

cwd = os.path.abspath(os.path.dirname(__file__))
selector_path = os.path.join(cwd, "Selector.ui")
encoder_path = os.path.join(cwd, "EncoderCopilot.ui")


def load_selector_ui(parent: QWidget | None = None) -> QgsDockWidget:
    """Create a fresh selector dock widget instance."""
    widget = cast(QgsDockWidget, uic.loadUi(selector_path))
    if parent is not None:
        widget.setParent(parent)
    return widget


def load_encoder_copilot_ui(parent: QWidget | None = None) -> QgsDockWidget:
    """Create a fresh encoder-copilot dock widget instance."""
    widget = cast(QgsDockWidget, uic.loadUi(encoder_path))
    if parent is not None:
        widget.setParent(parent)
    return widget
