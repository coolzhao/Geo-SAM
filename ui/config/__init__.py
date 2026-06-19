"""Configuration loading helpers for Geo-SAM UI settings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from qgis.PyQt.QtGui import QColor

__all__ = ['Settings', "DefaultSettings", "save_user_settings"]

logger = logging.getLogger(__name__)

cwd = Path(__file__).parent.absolute()

setting_default_file = cwd / 'default.json'
setting_user_file = cwd / 'user.json'


def _load_json_file(path: Path, *, create_missing: bool = False) -> dict[str, Any]:
    """Load a JSON settings file.

    Parameters
    ----------
    path : Path
        Path to the JSON settings file.
    create_missing : bool, optional
        Create an empty JSON object when the file does not exist.

    Returns
    -------
    dict[str, Any]
        Parsed settings dictionary.

    """
    if not path.exists() and create_missing:
        path.write_text("{}\n", encoding="utf-8")
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


DefaultSettings = _load_json_file(setting_default_file)
UserSettings = _load_json_file(setting_user_file, create_missing=True)

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


def _values_match_default(setting_name: str, value: Any) -> bool:
    """Return whether a user setting value matches the default value.

    Parameters
    ----------
    setting_name : str
        Name of the setting to compare.
    value : Any
        Candidate user setting value.

    Returns
    -------
    bool
        True when ``value`` matches the default setting value.

    """
    if setting_name not in DefaultSettings:
        return False
    default_value = DefaultSettings[setting_name]
    if setting_name in color_list:
        value_name = value.name() if hasattr(value, "name") else str(value)
        return value_name == default_value.name()
    return value == default_value


def _settings_runtime_value(setting_name: str, value: Any) -> Any:
    """Return the in-memory representation for a setting value.

    Parameters
    ----------
    setting_name : str
        Name of the setting to convert.
    value : Any
        Persisted setting value.

    Returns
    -------
    Any
        Runtime value, using :class:`QColor` for color settings.

    """
    if setting_name in color_list:
        return QColor(value)
    return value


def save_user_settings(
    settings: dict[str, Any],
    mode: Literal["update", "overwrite"] = "update",
) -> None:
    """Save user settings to file.

    Parameters
    ----------
    settings : dict
        Settings to save
    mode : {"update", "overwrite"}, optional
        'update' to update existing settings, 'overwrite' to overwrite all
        settings, by default 'update'
    Raises
    ------
    ValueError
        If ``mode`` is not ``"update"`` or ``"overwrite"``.

    """
    if mode == 'update':
        _new_settings = _load_json_file(setting_user_file, create_missing=True)
    elif mode == 'overwrite':
        _new_settings = {}
    else:
        msg = "mode must be either 'update' or 'overwrite'"
        logger.error("%s: %s", msg, mode)
        raise ValueError(msg)

    for st, value in settings.items():
        if _values_match_default(st, value):
            _new_settings.pop(st, None)
        else:
            _new_settings[st] = value
        Settings[st] = _settings_runtime_value(st, value)
    setting_user_file.write_text(
        json.dumps(_new_settings, indent=4),
        encoding="utf-8",
    )
