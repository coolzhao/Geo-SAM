"""Load Geo-SAM user interfaces on both Qt 5 and Qt 6."""

import os

from qgis.PyQt import uic
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QDockWidget, QFrame


def _add_qt5_ui_enum_aliases() -> None:
    """Add aliases required to load Qt 5 Designer files with PyQt 6.

    Notes
    -----
    PyQt 5's UI loader does not understand Qt 6 scoped enum names, while
    PyQt 6 removed the unscoped names emitted by Qt 5 Designer. Adding the
    missing aliases keeps the same ``.ui`` files loadable in both runtimes.
    """
    dock_widget_features = QDockWidget.DockWidgetFeature
    if not hasattr(QDockWidget, "AllDockWidgetFeatures"):
        QDockWidget.AllDockWidgetFeatures = (
            dock_widget_features.DockWidgetClosable
            | dock_widget_features.DockWidgetMovable
            | dock_widget_features.DockWidgetFloatable
        )

    enum_aliases = {
        Qt: {
            "AlignCenter": Qt.AlignmentFlag.AlignCenter,
            "AlignHCenter": Qt.AlignmentFlag.AlignHCenter,
            "AlignRight": Qt.AlignmentFlag.AlignRight,
            "AlignTop": Qt.AlignmentFlag.AlignTop,
            "AlignTrailing": Qt.AlignmentFlag.AlignTrailing,
            "AlignVCenter": Qt.AlignmentFlag.AlignVCenter,
            "Horizontal": Qt.Orientation.Horizontal,
            "LeftToRight": Qt.LayoutDirection.LeftToRight,
        },
        QFrame: {
            "NoFrame": QFrame.Shape.NoFrame,
            "Raised": QFrame.Shadow.Raised,
        },
    }
    for owner, aliases in enum_aliases.items():
        for name, value in aliases.items():
            if not hasattr(owner, name):
                setattr(owner, name, value)


_add_qt5_ui_enum_aliases()

cwd = os.path.abspath(os.path.dirname(__file__))
selector_path = os.path.join(cwd, "Selector.ui")
encoder_path = os.path.join(cwd, "EncoderCopilot.ui")

UI_Selector: QDockWidget = uic.loadUi(selector_path)
UI_EncoderCopilot: QDockWidget = uic.loadUi(encoder_path)
