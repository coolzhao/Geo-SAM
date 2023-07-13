from typing import Optional
from PyQt5.QtWidgets import QMessageBox
from qgis.core import Qgis
from qgis.utils import iface

LEVELS = {'info': Qgis.Info,
          'warning': Qgis.Warning,
          'critical': Qgis.Critical,
          'success': Qgis.Success}
LEVELS_ALL = list(LEVELS.keys())


class MessageTool:
    '''A class for showing messages to the user.'''

    def __init__(self):
        pass

    def MessageBoxOK(self, text) -> None:
        '''Show a message box with an OK button. 

        Parameters:
        ----------
        text : str
            The text of the message.
        '''
        mb = QMessageBox()
        mb.setText(
            text
        )
        mb.setStandardButtons(QMessageBox.Ok)
        mb.exec()

    def MessageBoxOKCancel(self, text) -> Optional[int]:
        '''Show a message box with OK and Cancel buttons.

        Parameters:
        ----------
        text : str
            The text of the message.

        Returns:
        -------
        int if the button clicked, or None if the message box was closed.
        '''
        mb = QMessageBox()
        mb.setText(text)
        mb.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        mb.setDefaultButton(QMessageBox.Cancel)
        return mb.exec()

    def MessageBoxYesNo(self, text) -> Optional[int]:
        '''Show a message box with Yes and No buttons.

        Parameters:
        ----------
        text : str
            The text of the message.

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
        return mb.exec()

    def MessageBar(
            self,
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
