"""User-facing QGIS message helpers."""

from __future__ import annotations

import logging

from qgis.PyQt.QtWidgets import QMessageBox
from qgis.core import Qgis, QgsMessageLog
from qgis.utils import iface

from .i18n import translate

logger = logging.getLogger(__name__)

LEVELS = {
    "info": Qgis.MessageLevel.Info,
    "warning": Qgis.MessageLevel.Warning,
    "critical": Qgis.MessageLevel.Critical,
    "success": Qgis.MessageLevel.Success,
}
LEVELS_ALL = list(LEVELS.keys())


class MessageTool:
    """A class for showing messages to the user."""

    @staticmethod
    def MessageBoxOK(text: str, title: str = "Warning") -> None:
        """Show a message box with an OK button.

        Parameters:
        ----------
        text : str
            The text of the message.
        title : str, optional
            The title of the message box, by default 'Warning'.

        Returns:
        -------
        int if the button clicked, or None if the message box was closed.
        """
        mb = QMessageBox()
        mb.setText(translate(text))
        mb.setStandardButtons(QMessageBox.StandardButton.Ok)
        mb.setWindowTitle(translate(title))
        mb.exec()

    @staticmethod
    def MessageBoxOKCancel(text: str, title: str = "Warning") -> int:
        """Show a message box with OK and Cancel buttons.

        Parameters:
        ----------
        text : str
            The text of the message.
        title : str, optional
            The title of the message box, by default 'Warning'.

        Returns:
        -------
        int if the button clicked, or None if the message box was closed.
        """
        mb = QMessageBox()
        mb.setText(translate(text))
        mb.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        mb.setDefaultButton(QMessageBox.StandardButton.Cancel)
        mb.setWindowTitle(translate(title))
        return mb.exec()

    @staticmethod
    def MessageBoxYesNo(text: str, title: str = "Warning") -> int:
        """Show a message box with Yes and No buttons.

        Parameters:
        ----------
        text : str
            The text of the message.
        title : str, optional
            The title of the message box, by default 'Warning'.

        Returns:
        -------
        int if the button clicked, or None if the message box was closed.
        """
        mb = QMessageBox()
        mb.setText(translate(text))
        mb.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        mb.setDefaultButton(QMessageBox.StandardButton.No)
        mb.setWindowTitle(translate(title))
        return mb.exec()

    @staticmethod
    def MessageBar(
        title: str, text: str, level: str = "info", duration: int = 10
    ) -> None:
        """Show a message in the QGIS message bar.

        Parameters:
        ----------
        title : str
            The title of the message.
        text : str
            The text of the message.
        level : str, optional
            The level of the message, by default 'info'
        duration : int, optional
            The duration of the message in seconds, by default 10
        """
        level = level.lower()
        if level not in LEVELS_ALL:
            msg = f"level must be one of {LEVELS_ALL}, got {level}"
            logger.error(msg)
            raise ValueError(msg)

        iface.messageBar().pushMessage(
            translate(title), translate(text), level=LEVELS[level], duration=duration
        )

    @staticmethod
    def MessageLog(
        text: str, level: str = "info", tag="Geo SAM", notify_user="auto"
    ) -> None:
        """Show a message in the QGIS log.

        Parameters:
        ----------
        text : str
            The text of the message.
        level : str, optional
            The level of the message, by default 'info'
        tag : str, optional
            The tag of the message, by default 'Geo SAM'
        notify_user : bool or 'auto', optional
            Whether to notify the user, by default 'auto'. If 'auto',
            the user will be notified for warning and critical messages.
        """
        level = level.lower()
        if level not in LEVELS_ALL:
            msg = f"level must be one of {LEVELS_ALL}, got {level}"
            logger.error(msg)
            raise ValueError(msg)
        if notify_user == "auto":
            notify_user = level in ["warning", "critical"]
        QgsMessageLog.logMessage(
            text, level=LEVELS[level], tag=tag, notifyUser=notify_user
        )
