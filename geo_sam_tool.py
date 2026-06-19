from __future__ import annotations

import logging
from typing import Literal

import processing
from qgis.PyQt.QtCore import QObject, pyqtSignal
from qgis.PyQt.QtGui import QAction
from qgis.PyQt.QtWidgets import (
    QMessageBox,
    QToolBar,
)
from qgis.core import QgsApplication
from qgis.gui import QgisInterface

from .tools.geosam_backend import configure_geosam_qgis_runtime
from .tools.i18n import install_translator, remove_translator
from .tools.plugin_settings import missing_plugin_runtime_dependencies
from .ui.icons import (
    QIcon_EncoderCopilot,
    QIcon_EncoderTool,
    QIcon_GeoSAMSettings,
    QIcon_GeoSAMTool,
)

logger = logging.getLogger(__name__)


class Geo_SAM(QObject):
    execute_SAM = pyqtSignal()

    def __init__(self, iface: QgisInterface, cwd: str):
        super().__init__()
        self.iface = iface
        self.cwd = cwd
        self.canvas = iface.mapCanvas()
        self.demo_img_name = "beiluhe_google_img_201211_clip"
        feature_dir = cwd + "/features/" + self.demo_img_name
        self.feature_dir = feature_dir

    def initProcessing(self):
        """Register the Geo-SAM processing provider lazily."""
        from .geo_sam_provider import GeoSamProvider

        self.provider = GeoSamProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        """Initialize plugin actions and configure the GeoSAM QGIS backend."""
        install_translator()
        configure_geosam_qgis_runtime()
        self.initProcessing()

        self.toolbar: QToolBar = self.iface.addToolBar(self.tr("Geo SAM Toolbar"))
        self.toolbar.setObjectName("mGeoSamToolbar")
        self.toolbar.setToolTip(self.tr("Geo SAM Toolbar"))

        self.actionSamTool = QAction(
            QIcon_GeoSAMTool, self.tr("Geo-SAM Segmentation"), self.iface.mainWindow()
        )

        self.actionSamEncoder = QAction(
            QIcon_EncoderTool, self.tr("Geo-SAM Image Encoder"), self.iface.mainWindow()
        )

        self.actionSamEncoderCopilot = QAction(
            QIcon_EncoderCopilot,
            self.tr("Geo-SAM Encoder Copilot"),
            self.iface.mainWindow(),
        )
        self.actionSamSettings = QAction(
            QIcon_GeoSAMSettings, self.tr("Geo-SAM Settings"), self.iface.mainWindow()
        )

        self.actionSamTool.setObjectName("mActionGeoSamTool")
        self.actionSamTool.setToolTip(
            self.tr("Geo-SAM Segmentation: Use it to label landforms")
        )
        self.actionSamTool.triggered.connect(self.create_widget_selector)

        self.actionSamEncoder.setObjectName("mActionGeoSamEncoder")
        self.actionSamEncoder.setToolTip(
            self.tr(
                "Geo-SAM Image Encoder: Use it to encode/preprocess image before labeling"
            )
        )
        self.actionSamEncoder.triggered.connect(self.encodeImage)

        self.actionSamEncoderCopilot.setObjectName("mActionGeoSamEncoderCopilot")
        self.actionSamEncoderCopilot.setToolTip(
            self.tr("Encoder Copilot: Assist you in optimizing your Encoder Settings")
        )
        self.actionSamEncoderCopilot.triggered.connect(
            self.create_widget_encoder_copilot
        )
        self.actionSamSettings.setObjectName("mActionGeoSamSettings")
        self.actionSamSettings.setToolTip(
            self.tr("Geo-SAM Settings: Manage dependencies, models, cache, and help")
        )
        self.actionSamSettings.triggered.connect(self.open_settings_dialog)

        menu_title = self.tr("Geo-SAM Tools")
        self.iface.addPluginToMenu(menu_title, self.actionSamTool)
        self.iface.addPluginToMenu(menu_title, self.actionSamEncoder)
        self.iface.addPluginToMenu(menu_title, self.actionSamEncoderCopilot)
        self.iface.addPluginToMenu(menu_title, self.actionSamSettings)

        # self.iface.addToolBarIcon(self.action)
        self.toolbar.addAction(self.actionSamTool)
        self.toolbar.addAction(self.actionSamEncoder)
        self.toolbar.addAction(self.actionSamEncoderCopilot)
        self.toolbar.addAction(self.actionSamSettings)
        self.toolbar.setVisible(True)

    def create_widget_selector(self):
        """Create widget for selecting landform by prompts"""
        action_label = self.tr("Segmentation")
        if not self._ensure_runtime_dependencies(action_label):
            return

        try:
            from .tools.widgetTool import Selector
        except ModuleNotFoundError as exc:
            self._show_missing_runtime_dependency(exc, action_label)
            return

        if not hasattr(self, "wdg_select"):
            self.wdg_select = Selector(self, self.iface, self.cwd)
        self.wdg_select.open_widget()

    def create_widget_encoder_copilot(self):
        """Create widget for co-piloting encoder settings"""
        action_label = self.tr("Encoder Copilot")
        if not self._ensure_runtime_dependencies(action_label):
            return

        try:
            from .tools.widgetTool import EncoderCopilot
        except ModuleNotFoundError as exc:
            self._show_missing_runtime_dependency(exc, action_label)
            return

        if not hasattr(self, "wdg_copilot"):
            self.wdg_copilot = EncoderCopilot(self, self.iface, self.cwd)
        self.wdg_copilot.open_widget()

    def open_settings_dialog(self) -> None:
        """Open the standalone settings dialog."""
        self._open_settings_dialog()

    def open_model_management_dialog(self, model_id: str | None = None) -> None:
        """Open model management and optionally select a model.

        Parameters
        ----------
        model_id : str | None, optional
            Model identifier to select after opening Model Management.

        """
        self._open_settings_dialog(initial_page="models", model_id=model_id)

    def _open_settings_dialog(
        self,
        *,
        initial_page: Literal["dependencies", "models"] | None = None,
        model_id: str | None = None,
    ) -> None:
        """Open settings and optionally select a management page.

        Parameters
        ----------
        initial_page : {"dependencies", "models"} | None, optional
            Settings page to select before showing the dialog.
        model_id : str | None, optional
            Model identifier to select on the Model Management page.

        """
        from .tools.settingsTool import GeoSamSettingsDialog

        dialog = GeoSamSettingsDialog(self.iface.mainWindow())
        if initial_page == "dependencies":
            dialog.show_dependencies()
        elif initial_page == "models":
            dialog.show_model_management(model_id)
        dialog.exec()
        if hasattr(self, "wdg_select"):
            self.wdg_select.refresh_runtime_controls()

    def _ensure_runtime_dependencies(self, action_label: str) -> bool:
        """Check plugin dependencies before opening a runtime feature.

        Parameters
        ----------
        action_label : str
            Translated feature name shown in the dependency prompt.

        Returns
        -------
        bool
            ``True`` when all dependencies are installed; otherwise ``False``.

        """
        missing_dependency_names = missing_plugin_runtime_dependencies()
        if not missing_dependency_names:
            return True

        logger.warning(
            "Geo-SAM dependencies are missing for %s: %s",
            action_label,
            ", ".join(missing_dependency_names),
        )
        self._prompt_dependency_installation(
            missing_dependency_names,
            action_label=action_label,
        )
        return False

    def _prompt_dependency_installation(
        self,
        missing_dependency_names: list[str],
        *,
        action_label: str,
    ) -> None:
        """Ask whether to open the Dependencies page for missing packages.

        Parameters
        ----------
        missing_dependency_names : list[str]
            Missing dependency module names to show to the user.
        action_label : str
            Translated feature name that requires the dependencies.

        """
        dependency_text = ", ".join(dict.fromkeys(missing_dependency_names))
        answer = QMessageBox.question(
            self.iface.mainWindow(),
            self.tr("Geo-SAM Dependencies Missing"),
            self.tr(
                "Geo-SAM requires the following dependencies for {action}:\n\n"
                "{dependencies}\n\n"
                "Would you like to open the Dependencies page and install them now?"
            ).format(
                action=action_label,
                dependencies=dependency_text,
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._open_settings_dialog(initial_page="dependencies")

    def _show_missing_runtime_dependency(
        self,
        error: ModuleNotFoundError,
        action_label: str,
    ) -> None:
        """Show dependency installation guidance for lazy runtime imports.

        Parameters
        ----------
        error : ModuleNotFoundError
            Import error raised while opening a runtime-only plugin feature.
        action_label : str
            Translated feature name that requires the missing dependency.
        """
        missing_name = getattr(error, "name", None) or "a required package"
        logger.warning("Geo-SAM runtime dependency is missing: %s", missing_name)
        self._prompt_dependency_installation(
            [missing_name],
            action_label=action_label,
        )

    def unload(self):
        """Unload actions when plugin is closed"""
        if hasattr(self, "wdg_select"):
            self.wdg_select.unload()
            self.wdg_select.setParent(None)
        if hasattr(self, "wdg_copilot"):
            self.wdg_copilot.unload()
            self.wdg_copilot.setParent(None)

        # self.wdg_select.setVisible(False)
        self.iface.removeToolBarIcon(self.actionSamTool)
        self.iface.removeToolBarIcon(self.actionSamEncoder)
        self.iface.removeToolBarIcon(self.actionSamEncoderCopilot)
        self.iface.removeToolBarIcon(self.actionSamSettings)
        self.iface.removePluginMenu("&Geo-SAM Tools", self.actionSamTool)
        self.iface.removePluginMenu("&Geo-SAM Tools", self.actionSamEncoder)
        self.iface.removePluginMenu("&Geo-SAM Tools", self.actionSamEncoderCopilot)
        self.iface.removePluginMenu("&Geo-SAM Tools", self.actionSamSettings)

        del self.actionSamTool
        del self.actionSamEncoder
        del self.actionSamEncoderCopilot
        del self.actionSamSettings
        del self.toolbar
        QgsApplication.processingRegistry().removeProvider(self.provider)
        from .tools.geosam_runtime import cleanup_on_plugin_unload

        cleanup_on_plugin_unload()
        remove_translator()

    def encodeImage(self):
        """Convert layer containing a point x & y coordinate to a new point layer"""
        if not self._ensure_runtime_dependencies(self.tr("Image Encoder")):
            return
        processing.execAlgorithmDialog("geo_sam:geo_sam_encoder", {})
