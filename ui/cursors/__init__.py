import os
from qgis.core import Qgis, QgsApplication
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtCore import QSize

__all__ = ['CursorPointBlue', 'CursorPointRed', 'CursorRect']

cwd = os.path.abspath(os.path.dirname(__file__))

CursorPointBlue_path = os.path.join(cwd, "CursorPointBlue.svg")
CursorPointRed_path = os.path.join(cwd, "CursorPointRed.svg")
CursorRect_path = os.path.join(cwd, "CursorRect.svg")

# scaling ref: https://github.com/qgis/QGIS/blob/11c77af3dd95fb1f5bb4ce3a4ef5dc97de951ec5/src/core/qgsapplication.cpp#L873
UI_SCALE = round(Qgis.UI_SCALE_FACTOR *
                 QgsApplication.fontMetrics().height())  # / 32.0
# QIcon("filepath.svg").pixmap(QSize()) https://stackoverflow.com/a/36936216


CursorPointBlueBitmap = QIcon(
    CursorPointBlue_path).pixmap(QSize(UI_SCALE, UI_SCALE))
CursorPointBlue = QCursor(CursorPointBlueBitmap)

CursorPointRedBitmap = QIcon(CursorPointRed_path).pixmap(
    QSize(UI_SCALE, UI_SCALE))
CursorPointRed = QCursor(CursorPointRedBitmap)

CursorRectBitmap = QIcon(CursorRect_path).pixmap(QSize(UI_SCALE, UI_SCALE))
CursorRect = QCursor(CursorRectBitmap)
