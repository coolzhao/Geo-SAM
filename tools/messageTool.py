from typing import Optional
from PyQt5.QtWidgets import QMessageBox
from qgis.core import Qgis, QgsMessageLog
from qgis.utils import iface

LEVELS = {'info': Qgis.Info,
          'warning': Qgis.Warning,
          'critical': Qgis.Critical,
          'success': Qgis.Success}
LEVELS_ALL = list(LEVELS.keys())


class MessageTool:
    '''A class for showing messages to the user.'''

    @staticmethod
    def MessageBoxOK(text: str, title: str = 'Warning') -> None:
        '''Show a message box with an OK button. 

        Parameters:
        ----------
        text : str
            The text of the message.
        title : str, optional
            The title of the message box, by default 'Warning'.
        
        Returns:
        -------
        int if the button clicked, or None if the message box was closed.
        '''
        mb = QMessageBox()
        mb.setText(
            text
        )
        mb.setStandardButtons(QMessageBox.Ok)
        mb.setWindowTitle(title)
        mb.exec()

    @staticmethod
    def MessageBoxOKCancel(text: str, title: str = 'Warning') -> Optional[int]:
        '''Show a message box with OK and Cancel buttons.

        Parameters:
        ----------
        text : str
            The text of the message.
        title : str, optional
            The title of the message box, by default 'Warning'.

        Returns:
        -------
        int if the button clicked, or None if the message box was closed.
        '''
        mb = QMessageBox()
        mb.setText(text)
        mb.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        mb.setDefaultButton(QMessageBox.Cancel)
        mb.setWindowTitle(title)
        return mb.exec()

    @staticmethod
    def MessageBoxYesNo(text: str, title: str = 'Warning') -> Optional[int]:
        '''Show a message box with Yes and No buttons.

        Parameters:
        ----------
        text : str
            The text of the message.
        title : str, optional
            The title of the message box, by default 'Warning'.

        Returns:
        -------
        int if the button clicked, or None if the message box was closed.
        '''
        mb = QMessageBox()
        mb.setText(
            text
        )
        mb.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        mb.setDefaultButton(QMessageBox.No)
        mb.setWindowTitle(title)
        return mb.exec()

    @staticmethod
    def MessageBar(
            title: str,
            text: str,
            level: str = 'info',
            duration: int = 10) -> None:
        '''Show a message in the QGIS message bar.

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
        '''
        level = level.lower()
        if level not in LEVELS_ALL:
            raise ValueError(
                f'level must be one of {LEVELS_ALL}, got {level}'
            )

        iface.messageBar().pushMessage(
            title,
            text,
            level=LEVELS[level],
            duration=duration
        )

    @staticmethod
    def MessageLog(text: str, level: str = 'info', tag="Geo SAM", notify_user='auto') -> None:
        '''Show a message in the QGIS log.

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
        '''
        level = level.lower()
        if level not in LEVELS_ALL:
            raise ValueError(
                f'level must be one of {LEVELS_ALL}, got {level}'
            )
        if notify_user == 'auto':
            notify_user = level in ['warning', 'critical']
        QgsMessageLog.logMessage(
            text,
            level=LEVELS[level],
            tag=tag,
            notifyUser=notify_user
        )
