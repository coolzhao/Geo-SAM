"""Internationalization helpers for the Geo-SAM plugin."""

from __future__ import annotations

import logging
from pathlib import Path

from qgis.PyQt.QtCore import (
    QCoreApplication,
    QLocale,
    QSettings,
    QTranslator,
)

logger = logging.getLogger(__name__)

TRANSLATION_CONTEXT = "GeoSAM"
TRANSLATION_DIRECTORY = Path(__file__).parents[1] / "i18n"
_translator: QTranslator | None = None


def current_locale_name() -> str:
    """Return the locale selected in QGIS.

    Returns
    -------
    str
        Locale name such as ``zh_CN`` or ``fr``.

    """
    locale_value = QSettings().value("locale/userLocale", "", type=str).strip()
    return locale_value or QLocale.system().name()


def install_translator() -> bool:
    """Install the best available Geo-SAM translator for the QGIS locale.

    Returns
    -------
    bool
        ``True`` when a translation catalog was loaded and installed.

    """
    global _translator

    remove_translator()
    locale_name = current_locale_name().replace("-", "_")
    locale_candidates = tuple(dict.fromkeys((locale_name, locale_name.split("_")[0])))
    translator = QTranslator()
    for locale_candidate in locale_candidates:
        translation_path = TRANSLATION_DIRECTORY / f"GeoSAM_{locale_candidate}.qm"
        if not translation_path.is_file():
            continue
        if translator.load(str(translation_path)):
            QCoreApplication.installTranslator(translator)
            _translator = translator
            logger.info("Loaded Geo-SAM translation: %s", translation_path)
            return True

    logger.debug("No Geo-SAM translation found for locale %s", locale_name)
    return False


def remove_translator() -> None:
    """Remove the active Geo-SAM translator, if one is installed."""
    global _translator

    if _translator is None:
        return
    QCoreApplication.removeTranslator(_translator)
    _translator = None


def translate(text: str) -> str:
    """Translate a user-facing Geo-SAM string.

    Parameters
    ----------
    text : str
        Source-language text.

    Returns
    -------
    str
        Translated text, or the source text when no translation is available.

    """
    return QCoreApplication.translate(TRANSLATION_CONTEXT, text)
