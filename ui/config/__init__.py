import json
from pathlib import Path

from PyQt5.QtGui import QColor

__all__ = ['Settings', "DefaultSettings", "save_user_settings"]

cwd = Path(__file__).parent.absolute()

setting_default_file = cwd / 'default.json'
setting_user_file = cwd / 'user.json'

with open(setting_default_file) as f:
    DefaultSettings = json.load(f)
with open(setting_user_file) as f:
    UserSettings = json.load(f)

color_list = [
    "fg_color",
    "bg_color",
    "bbox_color",
    "extent_color",
    "prompt_color",
    "preview_color"
]

Settings = DefaultSettings.copy()
Settings.update(UserSettings)

for color in color_list:
    Settings[color] = QColor(Settings[color])
    DefaultSettings[color] = QColor(DefaultSettings[color])


def save_user_settings(settings, mode='update'):
    '''Save user settings to file

    Parameters
    ----------
    settings : dict
        Settings to save
    mode : str, one of {'update', 'overwrite'}, optional
        'update' to update existing settings, 'overwrite' to overwrite all
        settings, by default 'update'
    '''
    if mode == 'update':
        with open(setting_user_file) as f:
            _new_settings = json.load(f)
    elif mode == 'overwrite':
        _new_settings = {}
    else:
        raise ValueError("mode must be either 'update' or 'overwrite'")
    with open(setting_user_file, 'w') as f:
        user_st = {}
        for st in settings:
            if settings[st] != Settings[st]:
                # skip for same color case
                if st in color_list:
                    if settings[st] == Settings[st].name():
                        continue
                user_st[st] = settings[st]
        _new_settings.update(user_st)
        json.dump(_new_settings, f, indent=4)
