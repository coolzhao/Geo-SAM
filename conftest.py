"""Pytest configuration: keep test collection out of the QGIS plugin package.

The plugin root ships an ``__init__.py`` (QGIS loads the folder as a package),
which makes pytest walk up and try to import the whole plugin — including
``from .tools.dependency_path import ...`` — before collecting any test. That
import only works inside QGIS. This conftest anchors the test rootdir at the
``tests/`` directory so the plugin package is never imported during collection.
"""

from __future__ import annotations

collect_ignore_glob = ["tools/*", "ui/*", "*.py"]
