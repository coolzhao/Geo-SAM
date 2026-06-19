"""QGIS message dialog, message bar, and log helpers."""

import logging
from typing import Literal, TypeAlias, cast

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtWidgets import QMessageBox
from qgis.utils import iface

logger = logging.getLogger(__name__)

MessageLevelName: TypeAlias = Literal["info", "warning", "critical", "success"]
NotifyUser: TypeAlias = bool | Literal["auto"]

LEVELS: dict[MessageLevelName, Qgis.MessageLevel] = {
    "info": Qgis.MessageLevel.Info,
    "warning": Qgis.MessageLevel.Warning,
    "critical": Qgis.MessageLevel.Critical,
    "success": Qgis.MessageLevel.Success,
}
LEVELS_ALL = list(LEVELS)


class MessageTool:
    """A class for showing messages to the user."""

    @staticmethod
    def MessageBoxOK(text: str, title: str = "Warning") -> None:
        """Show a message box with an OK button.

        Parameters
        ----------
        text : str
            The text of the message.
        title : str, optional
            The title of the message box, by default 'Warning'.

        """
        mb = QMessageBox()
        mb.setText(text)
        mb.setStandardButtons(QMessageBox.StandardButton.Ok)
        mb.setWindowTitle(title)
        mb.exec()

    @staticmethod
    def MessageBoxOKCancel(text: str, title: str = "Warning") -> int:
        """Show a message box with OK and Cancel buttons.

        Parameters
        ----------
        text : str
            The text of the message.
        title : str, optional
            The title of the message box, by default 'Warning'.

        Returns
        -------
        int
            Identifier of the clicked button.
        """
        mb = QMessageBox()
        mb.setText(text)
        mb.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        mb.setDefaultButton(QMessageBox.StandardButton.Cancel)
        mb.setWindowTitle(title)
        return mb.exec()

    @staticmethod
    def MessageBoxYesNo(text: str, title: str = "Warning") -> int:
        """Show a message box with Yes and No buttons.

        Parameters
        ----------
        text : str
            The text of the message.
        title : str, optional
            The title of the message box, by default 'Warning'.

        Returns
        -------
        int
            Identifier of the clicked button.
        """
        mb = QMessageBox()
        mb.setText(text)
        mb.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        mb.setDefaultButton(QMessageBox.StandardButton.No)
        mb.setWindowTitle(title)
        return mb.exec()

    @staticmethod
    def MessageBar(
        title: str,
        text: str,
        level: MessageLevelName = "info",
        duration: int = 10,
    ) -> None:
        """Show a message in the QGIS message bar.

        Parameters
        ----------
        title : str
            The title of the message.
        text : str
            The text of the message.
        level : {"info", "warning", "critical", "success"}, optional
            The level of the message, by default 'info'
        duration : int, optional
            The duration of the message in seconds, by default 10
        """
        normalized_level = level.lower()
        if normalized_level not in LEVELS:
            error_message = f"level must be one of {LEVELS_ALL}, got {normalized_level}"
            logger.error(error_message)
            raise ValueError(error_message)

        iface.messageBar().pushMessage(
            title,
            text,
            level=LEVELS[cast(MessageLevelName, normalized_level)],
            duration=duration,
        )

    @staticmethod
    def MessageLog(
        text: str,
        level: MessageLevelName = "info",
        tag: str = "Geo SAM",
        notify_user: NotifyUser = "auto",
    ) -> None:
        """Show a message in the QGIS log.

        Parameters
        ----------
        text : str
            The text of the message.
        level : {"info", "warning", "critical", "success"}, optional
            The level of the message, by default 'info'
        tag : str, optional
            The tag of the message, by default 'Geo SAM'
        notify_user : bool or {"auto"}, optional
            Whether to notify the user, by default 'auto'. If 'auto',
            the user will be notified for warning and critical messages.
        """
        normalized_level = level.lower()
        if normalized_level not in LEVELS:
            error_message = f"level must be one of {LEVELS_ALL}, got {normalized_level}"
            logger.error(error_message)
            raise ValueError(error_message)
        if notify_user == "auto":
            notify_user = normalized_level in {"warning", "critical"}
        QgsMessageLog.logMessage(
            text,
            level=LEVELS[cast(MessageLevelName, normalized_level)],
            tag=tag,
            notifyUser=notify_user,
        )
