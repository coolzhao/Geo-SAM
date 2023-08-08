import os
from qgis.core import Qgis, QgsApplication
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtCore import QSize

__all__ = [
    'CursorPointFG',
    'CursorPointBG',
    'CursorRect',
    'customize_fg_point_cursor',
    'customize_bg_point_cursor',
    'customize_bbox_cursor'
]

cwd = os.path.abspath(os.path.dirname(__file__))

__point_cursor_str = '''<?xml version="1.0" encoding="utf-8"?>
<!-- Generator: Adobe Illustrator 25.4.1, SVG Export Plug-In . SVG Version: 6.00 Build 0)  -->
<svg version="1.1" id="图层_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
	 viewBox="0 0 37.9 38" style="enable-background:new 0 0 37.9 38;" xml:space="preserve">
<style type="text/css">
	.st0{{{}}}
	.st1{{{}}}
	.st2{{{}}}
</style>
<g>
	<rect y="16.6" class="st0" width="37.9" height="4.8"/>
	<g>
		<line class="st1" x1="0" y1="19" x2="37.9" y2="19"/>
	</g>
</g>
<rect x="-0.1" y="16.6" transform="matrix(6.123234e-17 -1 1 6.123234e-17 -0.1 37.9)" class="st0" width="38" height="4.8"/>
<line class="st1" x1="18.9" y1="38" x2="18.9" y2="0"/>
<circle class="st2" cx="18.9" cy="19" r="7.5"/>
</svg>
'''

__rect_cursor_str = '''<?xml version="1.0" encoding="utf-8"?>
<!-- Generator: Adobe Illustrator 25.4.1, SVG Export Plug-In . SVG Version: 6.00 Build 0)  -->
<svg version="1.1" id="图层_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
	 viewBox="0 0 37.9 37.8" style="enable-background:new 0 0 37.9 37.8;" xml:space="preserve">
<style type="text/css">
	.st0{{{}}}
	.st1{{{}}}
	.st2{{{}}}
</style>
<rect y="16.5" transform="matrix(6.123234e-17 -1 1 6.123234e-17 0 37.8)" class="st0" width="37.8" height="4.8"/>
<g>
	<rect y="16.5" class="st0" width="37.9" height="4.8"/>
	<g>
		<line class="st1" x1="0" y1="18.9" x2="37.9" y2="18.9"/>
	</g>
</g>
<line class="st1" x1="18.9" y1="37.8" x2="18.9" y2="0"/>
<rect x="21.3" y="21.4" class="st2" width="15.1" height="14.9"/>
</svg>
'''

st0_point = 'fill:#FFFFFF;'
st1_point = 'fill:none;stroke:#000000;stroke-width:2;stroke-miterlimit:10;'
st2_point = 'fill:{};'

st0_rect = 'fill:#FFFFFF;'
st1_rect = 'fill:none;stroke:#000000;stroke-width:2;stroke-miterlimit:10;'
st2_rect = 'fill:none;stroke:{};stroke-width:3;stroke-miterlimit:10;'

CursorPointBlue_path = os.path.join(cwd, "CursorPointBlue.svg")
CursorPointRed_path = os.path.join(cwd, "CursorPointRed.svg")
CursorRect_path = os.path.join(cwd, "CursorRect.svg")

CursorFG_user_path = os.path.join(cwd, "CursorFG_user.svg")
CursorBG_user_path = os.path.join(cwd, "CursorBG_user.svg")
CursorBBox_user_path = os.path.join(cwd, "CursorBBox_user.svg")

if os.path.exists(CursorFG_user_path):
    CursorFG_path = CursorFG_user_path
else:
    CursorFG_path = CursorPointBlue_path

if os.path.exists(CursorBG_user_path):
    CursorBG_path = CursorBG_user_path
else:
    CursorBG_path = CursorPointRed_path

if os.path.exists(CursorBBox_user_path):
    CursorBBox_path = CursorBBox_user_path
else:
    CursorBBox_path = CursorRect_path

# scaling ref: https://github.com/qgis/QGIS/blob/11c77af3dd95fb1f5bb4ce3a4ef5dc97de951ec5/src/core/qgsapplication.cpp#L873
UI_SCALE = round(Qgis.UI_SCALE_FACTOR *
                 QgsApplication.fontMetrics().height())  # / 32.0
# QIcon("filepath.svg").pixmap(QSize()) https://stackoverflow.com/a/36936216


CursorPointFG = QCursor(
    QIcon(CursorFG_path)
    .pixmap(QSize(UI_SCALE, UI_SCALE))
)

CursorPointBG = QCursor(
    QIcon(CursorPointRed_path)
    .pixmap(QSize(UI_SCALE, UI_SCALE))
)

CursorRect = QCursor(
    QIcon(CursorRect_path)
    .pixmap(QSize(UI_SCALE, UI_SCALE))
)


def customize_fg_point_cursor(color2):
    svg = __point_cursor_str.format(
        st0_point,
        st1_point,
        st2_point.format(color2)
    )
    with open(CursorFG_user_path, 'w') as f:
        f.write(svg)

    CursorPointFG = QCursor(
        QIcon(CursorFG_user_path)
        .pixmap(QSize(UI_SCALE, UI_SCALE))
    )
    return CursorPointFG


def customize_bg_point_cursor(color2):
    svg = __point_cursor_str.format(
        st0_point,
        st1_point,
        st2_point.format(color2)
    )
    with open(CursorBG_user_path, 'w') as f:
        f.write(svg)

    CursorPointBG = QCursor(
        QIcon(CursorBG_user_path)
        .pixmap(QSize(UI_SCALE, UI_SCALE))
    )
    return CursorPointBG


def customize_bbox_cursor(color2):
    svg = __rect_cursor_str.format(
        st0_rect,
        st1_rect,
        st2_rect.format(color2)
    )

    with open(CursorBBox_user_path, 'w') as f:
        f.write(svg)

    CursorRect = QCursor(
        QIcon(CursorBBox_user_path)
        .pixmap(QSize(UI_SCALE, UI_SCALE))
    )
    return CursorRect
