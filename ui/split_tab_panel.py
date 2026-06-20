"""Split-tab panel with left-aligned and right-aligned tab bars."""

from __future__ import annotations

from typing import Literal

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QStackedWidget,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

_ActiveSide = Literal["left", "right"]

# Stylesheet fragments for active / inactive side visual distinction.
_ACTIVE_TAB_STYLE = """
QTabBar#leftTabBar::tab:selected,
QTabBar#rightTabBar::tab:selected {
    background: white;
    border-bottom-color: white;
    margin-bottom: -1px;
}
"""
_INACTIVE_TAB_STYLE = """
QTabBar#leftTabBar::tab:selected,
QTabBar#rightTabBar::tab:selected {
    background: #f0f0f0;
    border-bottom-color: #b8b8b8;
    margin-top: 3px;
}
"""
_LEFT_ACTIVE_STYLE = """
QTabBar#leftTabBar::tab:selected {
    background: white;
    border-bottom-color: white;
    margin-bottom: -1px;
}
QTabBar#rightTabBar::tab:selected {
    background: #f0f0f0;
    border-bottom-color: #b8b8b8;
    margin-top: 3px;
}
"""
_RIGHT_ACTIVE_STYLE = """
QTabBar#leftTabBar::tab:selected {
    background: #f0f0f0;
    border-bottom-color: #b8b8b8;
    margin-top: 3px;
}
QTabBar#rightTabBar::tab:selected {
    background: white;
    border-bottom-color: white;
    margin-bottom: -1px;
}
"""


class SplitTabPanel(QWidget):
    """Two tab bars sharing one stacked content area.

    Parameters
    ----------
    pages : list[QWidget]
        Content pages.  The first ``left_count`` pages belong to the left
        tab bar; the remaining pages belong to the right tab bar.
    left_count : int
        Number of pages owned by the left tab bar.
    left_labels : list[str]
        Tab labels for the left bar.
    right_labels : list[str]
        Tab labels for the right bar.
    parent : QWidget | None, optional
        Parent widget.

    """

    _BASE_STYLE = """
    QWidget#splitTabHeader {
        background: #fafafa;
        border-bottom: 1px solid #b8b8b8;
    }
    QTabBar {
        background: transparent;
    }
    QTabBar::tab {
        min-width: 64px;
        height: 24px;
        padding: 0 8px;
        border: 1px solid #b8b8b8;
        border-bottom: 1px solid #b8b8b8;
        background: #f0f0f0;
        color: #222;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar#leftTabBar::tab {
        margin-right: 2px;
    }
    QTabBar#rightTabBar::tab {
        margin-left: 2px;
        margin-right: 0px;
    }
    QTabBar::tab:!selected {
        margin-top: 3px;
    }
    QStackedWidget {
        border: 0px;
        background: white;
    }
    """

    def __init__(
        self,
        pages: list[QWidget],
        *,
        left_count: int,
        left_labels: list[str],
        right_labels: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._left_count = left_count
        self._active_side: _ActiveSide = "left"

        self.left_bar = QTabBar()
        self.right_bar = QTabBar()
        self.stack = QStackedWidget()

        self.left_bar.setObjectName("leftTabBar")
        self.right_bar.setObjectName("rightTabBar")

        for bar in (self.left_bar, self.right_bar):
            bar.setDrawBase(False)
            bar.setDocumentMode(True)
            bar.setExpanding(False)
            bar.setUsesScrollButtons(False)
            bar.setElideMode(Qt.TextElideMode.ElideNone)
            bar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        for label in left_labels:
            self.left_bar.addTab(label)
        for label in right_labels:
            self.right_bar.addTab(label)

        for page in pages:
            self.stack.addWidget(page)

        # -- Header: left tabs ... stretch ... right tabs ----------------
        header = QWidget()
        header.setObjectName("splitTabHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        header_layout.addWidget(
            self.left_bar, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom
        )
        header_layout.addStretch(1)
        header_layout.addWidget(
            self.right_bar,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
        )

        # -- Root layout -------------------------------------------------
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(header)
        root.addWidget(self.stack, 1)

        # -- Connections: use tabBarClicked for reliable switching --------
        self.left_bar.tabBarClicked.connect(self._on_left_clicked)
        self.right_bar.tabBarClicked.connect(self._on_right_clicked)

        # Initial state: first left tab selected.
        self.left_bar.setCurrentIndex(0)
        self.stack.setCurrentIndex(0)
        self._apply_active_style()

    # -- Public API ------------------------------------------------------

    def set_current_page(self, global_index: int) -> None:
        """Switch to a page by global index (0..left+right-1).

        Parameters
        ----------
        global_index : int
            Global page index.

        """
        if global_index < self._left_count:
            self._active_side = "left"
            self.left_bar.setCurrentIndex(global_index)
        else:
            self._active_side = "right"
            self.right_bar.setCurrentIndex(global_index - self._left_count)
        self.stack.setCurrentIndex(global_index)
        self._apply_active_style()

    # -- Slots -----------------------------------------------------------

    def _on_left_clicked(self, index: int) -> None:
        self._active_side = "left"
        self.stack.setCurrentIndex(index)
        self._apply_active_style()

    def _on_right_clicked(self, index: int) -> None:
        self._active_side = "right"
        self.stack.setCurrentIndex(self._left_count + index)
        self._apply_active_style()

    # -- Style -----------------------------------------------------------

    def _apply_active_style(self) -> None:
        """Update stylesheet so only the active side shows selected effect."""
        active_fragment = (
            _LEFT_ACTIVE_STYLE
            if self._active_side == "left"
            else _RIGHT_ACTIVE_STYLE
        )
        self.setStyleSheet(self._BASE_STYLE + active_fragment)
