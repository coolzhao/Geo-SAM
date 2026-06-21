"""Settings dialog for the Geo-SAM plugin."""

from __future__ import annotations

from pathlib import Path

from qgis.PyQt.QtCore import QProcess, QThread, Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .messageTool import MessageTool
from .i18n import translate
from .dependency_path import (
    clear_all_plugin_managed_site_packages,
    clear_current_plugin_managed_site_packages,
    get_plugin_managed_dependency_stats,
    get_plugin_managed_site_packages,
    iter_all_plugin_managed_site_packages,
)
from .model_manager import delete_model, download_model, get_model_status_rows
from .plugin_settings import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_DIR,
    HELP_LINKS,
    PERFORMANCE_MODE_VALUES,
    PLUGIN_ROOT,
    clear_cache,
    dependency_status_text,
    dependency_status_rows,
    format_dependency_install_command,
    format_dependency_install_commands,
    get_cache_directory,
    get_cache_size_bytes,
    get_dependency_install_commands,
    get_model_directory,
    load_plugin_settings,
    open_path,
    open_url,
    save_plugin_settings,
)
from .geosam_runtime import get_loaded_model_ids, release_runtime_models


class ModelDownloadThread(QThread):
    """Background worker that downloads one model checkpoint."""

    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, model_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model_id = model_id

    def run(self) -> None:
        """Download the requested model and emit the outcome."""
        try:
            checkpoint_path = download_model(self.model_id)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(str(checkpoint_path))


class GeoSamSettingsDialog(QDialog):
    """Standalone settings dialog for the Geo-SAM plugin."""

    DEPENDENCIES_TAB_INDEX = 0
    MODEL_MANAGEMENT_TAB_INDEX = 1

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Geo-SAM Settings"))
        self.resize(760, 520)
        self.settings = load_plugin_settings()
        self._dependency_install_process: QProcess | None = None
        self._dependency_install_commands: list[list[str]] = []
        self._dependency_install_command_index = 0
        self._dependency_install_output_buffer = ""
        self._missing_dependency_names: list[str] = []
        self._model_download_thread: ModelDownloadThread | None = None

        self.tab_widget = QTabWidget(self)
        self.tab_widget.addTab(self._build_dependency_tab(), self.tr("Dependencies"))
        self.tab_widget.addTab(self._build_model_tab(), self.tr("Model Management"))
        self.tab_widget.addTab(self._build_cache_tab(), self.tr("Cache"))
        self.tab_widget.addTab(self._build_help_tab(), self.tr("Help"))

        self.close_button = QPushButton(self.tr("Close"), self)
        self.close_button.clicked.connect(self.accept)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(self.tab_widget)
        root_layout.addWidget(self.close_button, alignment=Qt.AlignmentFlag.AlignRight)

        self.refresh_dependency_status()
        self.refresh_model_list()
        self.refresh_cache_status()

    def show_dependencies(self) -> None:
        """Select the dependency management page."""
        self.tab_widget.setCurrentIndex(self.DEPENDENCIES_TAB_INDEX)

    def show_model_management(self, model_id: str | None = None) -> None:
        """Select model management and optionally highlight a model.

        Parameters
        ----------
        model_id : str | None, optional
            Model identifier to select in the model table.

        """
        self.tab_widget.setCurrentIndex(self.MODEL_MANAGEMENT_TAB_INDEX)
        if model_id is None:
            return

        for row_index in range(self.model_table.rowCount()):
            item = self.model_table.item(row_index, self.COL_NAME)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == model_id:
                self.model_table.selectRow(row_index)
                return

    def _build_dependency_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        self.dependency_summary_label = QLabel(tab)
        self.dependency_summary_label.setWordWrap(True)
        self.dependency_storage_label = QLabel(tab)
        self.dependency_storage_label.setWordWrap(True)
        self.dependency_storage_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.dependency_table = QTableWidget(tab)
        self.dependency_table.setColumnCount(4)
        self.dependency_table.setHorizontalHeaderLabels(
            [
                self.tr("Package"),
                self.tr("Status"),
                self.tr("Version"),
                self.tr("Source"),
            ]
        )
        self.dependency_table.verticalHeader().setVisible(True)
        self.dependency_table.setAlternatingRowColors(True)
        self.dependency_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.dependency_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.dependency_table.horizontalHeader().setStretchLastSection(True)
        self.dependency_table.setMinimumHeight(220)

        button_row = QHBoxLayout()
        self.refresh_dependencies_button = QPushButton(self.tr("Refresh Status"), tab)
        self.refresh_dependencies_button.clicked.connect(self.refresh_dependency_status)
        self.install_dependencies_button = QPushButton(self.tr("Install Missing"), tab)
        self.install_dependencies_button.clicked.connect(
            self.install_dependencies_clicked
        )
        self.open_dependency_folder_button = QPushButton(self.tr("Open Folder"), tab)
        self.open_dependency_folder_button.clicked.connect(
            self.open_dependency_folder_clicked
        )
        self.clear_current_dependencies_button = QPushButton(
            self.tr("Clear Current Runtime"),
            tab,
        )
        self.clear_current_dependencies_button.clicked.connect(
            self.clear_current_dependencies_clicked
        )
        self.clear_all_dependencies_button = QPushButton(
            self.tr("Clear All Runtimes"), tab
        )
        self.clear_all_dependencies_button.clicked.connect(
            self.clear_all_dependencies_clicked
        )
        button_row.addWidget(self.install_dependencies_button)
        button_row.addWidget(self.refresh_dependencies_button)
        button_row.addWidget(self.open_dependency_folder_button)
        button_row.addWidget(self.clear_current_dependencies_button)
        button_row.addWidget(self.clear_all_dependencies_button)

        self.dependency_install_status_label = QLabel(
            self.tr("Installable dependency status will appear here."),
            tab,
        )
        self.dependency_install_status_label.setWordWrap(True)
        self.dependency_install_progress = QProgressBar(tab)
        self.dependency_install_progress.setRange(0, 0)
        self.dependency_install_progress.setVisible(False)

        self.dependency_install_log = QPlainTextEdit(tab)
        self.dependency_install_log.setReadOnly(True)
        self.dependency_install_log.setPlaceholderText(
            self.tr("Dependency installation output will appear here.")
        )
        self.dependency_install_log.setMinimumHeight(220)

        layout.addWidget(self.dependency_summary_label)
        layout.addWidget(self.dependency_storage_label)
        layout.addWidget(self.dependency_table)
        layout.addLayout(button_row)
        layout.addWidget(self.dependency_install_status_label)
        layout.addWidget(self.dependency_install_progress)
        layout.addWidget(self.dependency_install_log)
        return tab

    # Filter tab indices for the model filter bar.
    FILTER_ALL = 0
    FILTER_DOWNLOADED = 1
    FILTER_NOT_DOWNLOADED = 2
    FILTER_IN_MEMORY = 3

    # Table column indices.
    COL_NAME = 0
    COL_SERIES = 1
    COL_FILE_STATUS = 2
    COL_LOAD_STATUS = 3
    COL_ACTIONS = 4

    def _build_model_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        # -- Storage group ------------------------------------------------
        path_group = QGroupBox(self.tr("Storage"), tab)
        path_layout = QGridLayout(path_group)
        self.model_dir_edit = QLineEdit(str(self.settings["model_store_dir"]), tab)
        self.model_dir_edit.editingFinished.connect(self.save_model_directory)

        browse_button = QPushButton(self.tr("Browse"), tab)
        browse_button.clicked.connect(self.browse_model_directory)
        open_button = QPushButton(self.tr("Open Folder"), tab)
        open_button.clicked.connect(lambda: open_path(get_model_directory()))

        path_layout.addWidget(QLabel(self.tr("Model Folder"), tab), 0, 0)
        path_layout.addWidget(self.model_dir_edit, 0, 1)
        path_layout.addWidget(browse_button, 0, 2)
        path_layout.addWidget(open_button, 0, 3)

        # -- Models group --------------------------------------------------
        list_group = QGroupBox(self.tr("Models"), tab)
        list_layout = QVBoxLayout(list_group)

        # Filter bar
        self.model_filter_bar = QTabBar(tab)
        self.model_filter_bar.addTab(self.tr("All"))
        self.model_filter_bar.addTab(self.tr("Downloaded"))
        self.model_filter_bar.addTab(self.tr("Not Downloaded"))
        self.model_filter_bar.addTab(self.tr("In Memory"))
        self.model_filter_bar.currentChanged.connect(self._apply_model_filter)
        list_layout.addWidget(self.model_filter_bar)

        # Model table
        self.model_table = QTableWidget(tab)
        self.model_table.setColumnCount(5)
        self.model_table.setHorizontalHeaderLabels([
            self.tr("Model Name"),
            self.tr("Series"),
            self.tr("File Status"),
            self.tr("Load Status"),
            self.tr("Actions"),
        ])
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.model_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.model_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.model_table.horizontalHeader().setStretchLastSection(True)
        self.model_table.setMinimumHeight(260)
        list_layout.addWidget(self.model_table)

        # Download progress
        self.model_download_status_label = QLabel(self.tr("Ready"), tab)
        list_layout.addWidget(self.model_download_status_label)
        self.model_download_progress = QProgressBar(tab)
        self.model_download_progress.setRange(0, 0)
        self.model_download_progress.setVisible(False)
        list_layout.addWidget(self.model_download_progress)

        # -- Current model status -----------------------------------------
        self.current_model_label = QLabel(tab)
        self.current_model_label.setWordWrap(True)
        self.current_model_status_label = QLabel(tab)
        self.current_model_status_label.setWordWrap(True)

        # -- Bottom action buttons ----------------------------------------
        bottom_row = QHBoxLayout()
        self.unload_button = QPushButton(self.tr("Unload Current"), tab)
        self.unload_button.setEnabled(False)
        self.unload_button.clicked.connect(self.unload_current_model_clicked)
        refresh_button = QPushButton(self.tr("Refresh"), tab)
        refresh_button.clicked.connect(self.refresh_model_list)
        bottom_row.addWidget(self.unload_button)
        bottom_row.addWidget(refresh_button)
        bottom_row.addStretch(1)

        layout.addWidget(path_group)
        layout.addWidget(list_group)
        layout.addWidget(self.current_model_label)
        layout.addWidget(self.current_model_status_label)
        layout.addLayout(bottom_row)
        return tab

    def _build_cache_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        form_layout = QFormLayout()
        self.cache_enabled_checkbox = QCheckBox(self.tr("Enable cache"), tab)
        self.cache_enabled_checkbox.setChecked(bool(self.settings["cache_enabled"]))
        self.cache_enabled_checkbox.toggled.connect(self.save_cache_settings)

        cache_dir_row = QHBoxLayout()
        self.cache_dir_edit = QLineEdit(str(self.settings["cache_dir"]), tab)
        self.cache_dir_edit.editingFinished.connect(self.save_cache_settings)
        cache_browse_button = QPushButton(self.tr("Browse"), tab)
        cache_browse_button.clicked.connect(self.browse_cache_directory)
        cache_open_button = QPushButton(self.tr("Open Folder"), tab)
        cache_open_button.clicked.connect(lambda: open_path(get_cache_directory()))
        cache_dir_row.addWidget(self.cache_dir_edit)
        cache_dir_row.addWidget(cache_browse_button)
        cache_dir_row.addWidget(cache_open_button)

        cache_size_row = QHBoxLayout()
        self.cache_size_slider = QSlider(Qt.Orientation.Horizontal, tab)
        self.cache_size_slider.setRange(100, 20480)
        self.cache_size_slider.setValue(int(self.settings["cache_max_size_mb"]))
        self.cache_size_edit = QLineEdit(str(self.settings["cache_max_size_mb"]), tab)
        self.cache_size_edit.setFixedWidth(70)
        self.cache_size_suffix = QLabel(self.tr("MB"), tab)
        self.cache_size_slider.valueChanged.connect(self._on_cache_slider_changed)
        self.cache_size_edit.editingFinished.connect(self._on_cache_edit_finished)
        cache_size_row.addWidget(self.cache_size_slider)
        cache_size_row.addWidget(self.cache_size_edit)
        cache_size_row.addWidget(self.cache_size_suffix)

        self.performance_mode_combo = QComboBox(tab)
        self.performance_mode_combo.addItem(self.tr("Balanced"), "balanced")
        self.performance_mode_combo.addItem(self.tr("Fastest"), "fastest")
        self.performance_mode_combo.addItem(self.tr("Low Memory"), "low_memory")
        self.performance_mode_combo.setCurrentIndex(
            max(
                0,
                self.performance_mode_combo.findData(
                    str(self.settings.get("performance_mode", "balanced"))
                ),
            )
        )
        self.performance_mode_combo.currentIndexChanged.connect(
            self.save_cache_settings
        )

        self.clear_cache_on_close_checkbox = QCheckBox(
            self.tr("Clear cache when the plugin closes"),
            tab,
        )
        self.clear_cache_on_close_checkbox.setChecked(
            bool(self.settings.get("clear_cache_on_plugin_close", True))
        )
        self.clear_cache_on_close_checkbox.toggled.connect(self.save_cache_settings)

        self.cache_status_label = QLabel("", tab)
        clear_button = QPushButton(self.tr("Clear Cache Now"), tab)
        clear_button.clicked.connect(self.clear_cache_clicked)

        form_layout.addRow(self.tr("Cache"), self.cache_enabled_checkbox)
        form_layout.addRow(self.tr("Location"), cache_dir_row)
        form_layout.addRow(self.tr("Max Size"), cache_size_row)
        form_layout.addRow(self.tr("Model Performance"), self.performance_mode_combo)
        form_layout.addRow(
            self.tr("Close Behavior"), self.clear_cache_on_close_checkbox
        )
        form_layout.addRow(self.tr("Current Usage"), self.cache_status_label)
        form_layout.addRow("", clear_button)

        layout.addLayout(form_layout)
        layout.addStretch(1)
        return tab

    def _build_help_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        metadata_path = PLUGIN_ROOT / "metadata.txt"
        version_label = QLabel(self._plugin_version_text(metadata_path), tab)
        version_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(version_label)

        for label_text, url in HELP_LINKS.items():
            button = QPushButton(translate(label_text), tab)
            button.clicked.connect(lambda _checked=False, target=url: open_url(target))
            layout.addWidget(button)

        layout.addStretch(1)
        return tab

    def _plugin_version_text(self, metadata_path: Path) -> str:
        version = self.tr("unknown")
        if metadata_path.exists():
            for line in metadata_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("version="):
                    version = line.split("=", 1)[1].strip()
                    break
        return self.tr("Geo-SAM Version: {version}").format(version=version)

    def refresh_dependency_status(self) -> None:
        rows = dependency_status_rows()
        installed_count = sum(1 for row in rows if row["installed"])
        missing_count = len(rows) - installed_count
        installable_missing_rows = [
            row for row in rows if not row["installed"] and row["installable"]
        ]
        self._missing_dependency_names = [
            row["package"] for row in installable_missing_rows
        ]
        self.dependency_summary_label.setText(
            self.tr(
                "Dependencies: {installed} installed, {missing} missing. "
                "{installable} installable. Installation runs in the background."
            ).format(
                installed=installed_count,
                missing=missing_count,
                installable=len(installable_missing_rows),
            )
        )
        current_dependency_path = get_plugin_managed_site_packages()
        current_stats = get_plugin_managed_dependency_stats(current_dependency_path)
        all_dependency_paths = iter_all_plugin_managed_site_packages(
            include_legacy=True,
        )
        all_stats = [
            get_plugin_managed_dependency_stats(dependency_path)
            for dependency_path in all_dependency_paths
        ]
        total_size_bytes = sum(row["size_bytes"] for row in all_stats)
        self.dependency_storage_label.setText(
            self.tr(
                "Plugin-managed install: {current_size} in current runtime, "
                "{total_size} across {folder_count} runtime folder(s). "
                "Current path: {current_path}"
            ).format(
                current_size=self._format_bytes(current_stats["size_bytes"]),
                total_size=self._format_bytes(total_size_bytes),
                folder_count=len(all_dependency_paths),
                current_path=current_dependency_path,
            )
        )
        self.dependency_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            status_text = translate(dependency_status_text(row["state"]))
            version_text = row["version"] if row["installed"] else "-"
            text_color = QColor("#008c4a") if row["installed"] else QColor("#b00020")
            for column_index, value in enumerate(
                (row["package"], status_text, version_text, translate(row["source"]))
            ):
                item = QTableWidgetItem(value)
                item.setForeground(text_color)
                self.dependency_table.setItem(row_index, column_index, item)
        self.dependency_table.resizeColumnsToContents()
        self.dependency_table.horizontalHeader().setStretchLastSection(True)

        is_installing = self._dependency_install_process is not None
        self.install_dependencies_button.setEnabled(
            bool(self._missing_dependency_names) and not is_installing
        )
        self.refresh_dependencies_button.setEnabled(not is_installing)
        self.open_dependency_folder_button.setEnabled(not is_installing)
        self.clear_current_dependencies_button.setEnabled(
            current_stats["exists"] and not is_installing
        )
        self.clear_all_dependencies_button.setEnabled(
            bool(all_dependency_paths) and not is_installing
        )

    def install_dependencies_clicked(self) -> None:
        if self._dependency_install_process is not None:
            return
        if not self._missing_dependency_names:
            self.refresh_dependency_status()
            self.dependency_install_status_label.setText(
                self.tr("No installable dependencies are missing.")
            )
            return

        self.dependency_install_log.clear()
        missing_dependencies = ", ".join(self._missing_dependency_names)
        self.dependency_install_status_label.setText(
            self.tr("Installing missing dependencies: {dependencies}").format(
                dependencies=missing_dependencies
            )
        )
        self.dependency_install_progress.setVisible(True)
        self.install_dependencies_button.setEnabled(False)
        self.refresh_dependencies_button.setEnabled(False)
        self.open_dependency_folder_button.setEnabled(False)
        self.clear_current_dependencies_button.setEnabled(False)
        self.clear_all_dependencies_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self._dependency_install_output_buffer = ""
        self._append_dependency_install_log(
            self.tr("Starting dependency installation: {dependencies}").format(
                dependencies=missing_dependencies
            )
        )
        try:
            commands = get_dependency_install_commands(self._missing_dependency_names)
        except RuntimeError as exc:
            self._finish_dependency_install(False, str(exc))
            return
        self._append_dependency_install_log(
            self.tr("Commands:\n{commands}").format(
                commands=format_dependency_install_commands(commands)
            )
        )

        self._dependency_install_commands = commands
        self._dependency_install_command_index = 0
        self._start_next_dependency_install_command()

    def _start_next_dependency_install_command(self) -> None:
        """Start the next pending dependency install command."""
        if self._dependency_install_command_index >= len(
            self._dependency_install_commands
        ):
            output = self.dependency_install_log.toPlainText().strip()
            self._finish_dependency_install(True, output)
            return

        command = self._dependency_install_commands[
            self._dependency_install_command_index
        ]
        self._dependency_install_command_index += 1
        self._append_dependency_install_log(
            self.tr("Running command: {command}").format(
                command=format_dependency_install_command(command)
            )
        )
        process = QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_dependency_install_output)
        process.errorOccurred.connect(self._dependency_install_error)
        process.finished.connect(self._dependency_install_finished)
        self._dependency_install_process = process
        process.start()

    def _read_dependency_install_output(self) -> None:
        """Read and append incremental dependency install output."""
        process = self._dependency_install_process
        if process is None:
            return

        raw_output = bytes(process.readAllStandardOutput()).decode(
            "utf-8",
            errors="replace",
        )
        if not raw_output:
            return

        self._dependency_install_output_buffer += raw_output
        while True:
            newline_index = self._dependency_install_output_buffer.find("\n")
            carriage_index = self._dependency_install_output_buffer.find("\r")
            separator_indexes = [
                index for index in (newline_index, carriage_index) if index >= 0
            ]
            if not separator_indexes:
                break

            separator_index = min(separator_indexes)
            separator = self._dependency_install_output_buffer[separator_index]
            line = self._dependency_install_output_buffer[:separator_index]
            self._dependency_install_output_buffer = (
                self._dependency_install_output_buffer[separator_index + 1 :]
            )
            clean_line = line.strip()
            if separator == "\n":
                self._append_dependency_install_log(clean_line)

    def _dependency_install_finished(self, exit_code: int, _exit_status: int) -> None:
        """Handle completion of the asynchronous dependency install command."""
        if self._dependency_install_process is None:
            return

        if self._dependency_install_output_buffer:
            clean_line = self._dependency_install_output_buffer.strip()
            self._append_dependency_install_log(clean_line)
            self._dependency_install_output_buffer = ""

        self._dependency_install_process = None
        if exit_code == 0:
            self._start_next_dependency_install_command()
            return

        output = self.dependency_install_log.toPlainText().strip()
        self._finish_dependency_install(False, output)

    def _dependency_install_error(self, _error: QProcess.ProcessError) -> None:
        """Handle startup or runtime failures from the dependency installer."""
        process = self._dependency_install_process
        if process is None:
            return

        self._read_dependency_install_output()
        error_message = process.errorString().strip() or (
            self.tr("Dependency installer process failed to start.")
        )
        combined_output = self.dependency_install_log.toPlainText().strip()
        if combined_output:
            combined_output = f"{combined_output}\n{error_message}"
        else:
            combined_output = error_message
        self._finish_dependency_install(False, combined_output)

    def _finish_dependency_install(self, ok: bool, output: str) -> None:
        """Restore dependency install UI state and report the result."""
        self.dependency_install_progress.setVisible(False)
        self.install_dependencies_button.setEnabled(True)
        self.refresh_dependencies_button.setEnabled(True)
        self.close_button.setEnabled(True)
        self.dependency_install_status_label.setText(
            self.tr("Dependency installation completed.")
            if ok
            else self.tr("Dependency installation failed.")
        )
        self._dependency_install_process = None
        self._dependency_install_commands = []
        self._dependency_install_command_index = 0

        if ok and not output.strip():
            self._append_dependency_install_log(
                self.tr("Dependency installation completed.")
            )
        elif not ok and not output.strip():
            self._append_dependency_install_log(
                self.tr("Dependency installation failed.")
            )
        self.refresh_dependency_status()

    def _append_dependency_install_log(self, message: str) -> None:
        """Append a dependency installation log line to the settings dialog."""
        if not message:
            return
        self.dependency_install_log.appendPlainText(message)
        scrollbar = self.dependency_install_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def open_dependency_folder_clicked(self) -> None:
        """Open the active runtime dependency folder."""
        open_path(get_plugin_managed_site_packages(create=True))

    def clear_current_dependencies_clicked(self) -> None:
        """Delete plugin-managed dependencies for the active runtime."""
        dependency_path = get_plugin_managed_site_packages()
        answer = QMessageBox.question(
            self,
            self.tr("Clear Current Runtime Dependencies"),
            self.tr(
                "Delete plugin-managed dependencies for the current QGIS runtime?\n\n"
                "Path:\n{path}\n\n"
                "Restart QGIS after clearing if any dependency was already imported."
            ).format(path=dependency_path),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            removed_count = clear_current_plugin_managed_site_packages()
        except OSError as exc:
            MessageTool.MessageBoxOK(
                str(exc),
                title=self.tr("Dependency Cleanup Failed"),
            )
            return
        self._append_dependency_install_log(
            self.tr(
                "Cleared {count} file(s) from current runtime dependencies."
            ).format(count=removed_count)
        )
        self.refresh_dependency_status()

    def clear_all_dependencies_clicked(self) -> None:
        """Delete all plugin-managed dependency installs."""
        dependency_paths = iter_all_plugin_managed_site_packages(include_legacy=True)
        path_text = "\n".join(str(path) for path in dependency_paths)
        answer = QMessageBox.question(
            self,
            self.tr("Clear All Runtime Dependencies"),
            self.tr(
                "Delete all plugin-managed dependency installs for Geo-SAM?\n\n"
                "Paths:\n{paths}\n\n"
                "This does not remove QGIS, conda, or global Python packages. "
                "Restart QGIS after clearing if any dependency was already imported."
            ).format(paths=path_text),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            removed_count = clear_all_plugin_managed_site_packages()
        except OSError as exc:
            MessageTool.MessageBoxOK(
                str(exc),
                title=self.tr("Dependency Cleanup Failed"),
            )
            return
        self._append_dependency_install_log(
            self.tr("Cleared {count} file(s) from all runtime dependencies.").format(
                count=removed_count
            )
        )
        self.refresh_dependency_status()

    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        """Return a compact human-readable byte count.

        Parameters
        ----------
        size_bytes : int
            Size in bytes.

        Returns
        -------
        str
            Human-readable size text.

        """
        size_value = float(size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size_value < 1024 or unit == "GB":
                return f"{size_value:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
            size_value /= 1024
        return f"{size_value:.1f} GB"

    def save_model_directory(self) -> None:
        value = self.model_dir_edit.text().strip()
        if not value:
            value = str(DEFAULT_MODEL_DIR)
            self.model_dir_edit.setText(value)
        self.settings = save_plugin_settings({"model_store_dir": value})
        self.refresh_model_list()

    def browse_model_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Model Directory"),
            self.model_dir_edit.text() or str(DEFAULT_MODEL_DIR),
        )
        if directory:
            self.model_dir_edit.setText(directory)
            self.save_model_directory()

    def refresh_model_list(self) -> None:
        """Rebuild the model table from current status and apply the active filter."""
        loaded_ids = get_loaded_model_ids()
        rows = get_model_status_rows()
        self.model_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            model_id = row["model_id"]
            is_downloaded = row["downloaded"]
            is_loaded = model_id in loaded_ids

            # Column 0: Model Name
            name_item = QTableWidgetItem(row["label"])
            name_item.setData(Qt.ItemDataRole.UserRole, model_id)
            self.model_table.setItem(row_index, self.COL_NAME, name_item)

            # Column 1: Series
            series_item = QTableWidgetItem(row["model_type"].upper())
            self.model_table.setItem(row_index, self.COL_SERIES, series_item)

            # Column 2: File Status
            if is_downloaded:
                file_text = self.tr("Downloaded")
                file_color = QColor("#008c4a")
            else:
                file_text = self.tr("Not Downloaded")
                file_color = QColor("#b00020")
            file_item = QTableWidgetItem(file_text)
            file_item.setForeground(file_color)
            self.model_table.setItem(row_index, self.COL_FILE_STATUS, file_item)

            # Column 3: Load Status
            if is_loaded:
                load_text = self.tr("In Memory")
                load_color = QColor("#008c4a")
            else:
                load_text = "—"
                load_color = QColor("#888888")
            load_item = QTableWidgetItem(load_text)
            load_item.setForeground(load_color)
            self.model_table.setItem(row_index, self.COL_LOAD_STATUS, load_item)

            # Column 4: Actions
            action_widget = self._create_model_action_widget(
                model_id, is_downloaded, is_loaded
            )
            self.model_table.setCellWidget(row_index, self.COL_ACTIONS, action_widget)

        header = self.model_table.horizontalHeader()
        header.setStretchLastSection(False)
        for col in range(self.model_table.columnCount()):
            header.setSectionResizeMode(
                col, QHeaderView.ResizeMode.Stretch
            )
        self._apply_model_filter()
        self._update_current_model_status()

    def _create_model_action_widget(
        self,
        model_id: str,
        is_downloaded: bool,
        is_loaded: bool,
    ) -> QWidget:
        """Create the inline action buttons for a model table row.

        Parameters
        ----------
        model_id : str
            Model identifier.
        is_downloaded : bool
            Whether the checkpoint file exists locally.
        is_loaded : bool
            Whether the model has engines in memory.

        Returns
        -------
        QWidget
            Widget with horizontal layout of contextual action buttons.

        """
        widget = QWidget(self.model_table)
        hbox = QHBoxLayout(widget)
        hbox.setContentsMargins(2, 2, 2, 2)
        hbox.setSpacing(4)

        is_downloading = self._model_download_thread is not None

        if not is_downloaded:
            download_btn = QPushButton(self.tr("Download"), widget)
            download_btn.clicked.connect(
                lambda _checked=False, mid=model_id: self.download_model_by_id(mid)
            )
            download_btn.setEnabled(not is_downloading)
            hbox.addWidget(download_btn)
        else:
            delete_btn = QPushButton(self.tr("Delete"), widget)
            delete_btn.clicked.connect(
                lambda _checked=False, mid=model_id: self.delete_model_by_id(mid)
            )
            delete_btn.setEnabled(not is_downloading)
            hbox.addWidget(delete_btn)

        hbox.addStretch(1)
        return widget

    def _apply_model_filter(self) -> None:
        """Show or hide table rows based on the active filter tab."""
        filter_index = self.model_filter_bar.currentIndex()
        loaded_ids = get_loaded_model_ids()
        for row_index in range(self.model_table.rowCount()):
            name_item = self.model_table.item(row_index, self.COL_NAME)
            if name_item is None:
                continue
            model_id = str(name_item.data(Qt.ItemDataRole.UserRole))
            file_item = self.model_table.item(row_index, self.COL_FILE_STATUS)
            is_downloaded = (
                file_item is not None
                and file_item.text() == self.tr("Downloaded")
            )
            is_loaded = model_id in loaded_ids

            visible = True
            if filter_index == self.FILTER_DOWNLOADED:
                visible = is_downloaded
            elif filter_index == self.FILTER_NOT_DOWNLOADED:
                visible = not is_downloaded
            elif filter_index == self.FILTER_IN_MEMORY:
                visible = is_loaded

            self.model_table.setRowHidden(row_index, not visible)

    def _update_current_model_status(self) -> None:
        """Update the current-model and unload-button state."""
        loaded_ids = get_loaded_model_ids()
        if loaded_ids:
            labels: list[str] = []
            for model_id in loaded_ids:
                try:
                    from .model_manager import get_model_definition

                    definition = get_model_definition(model_id)
                    labels.append(definition.label)
                except KeyError:
                    labels.append(model_id)
            display = ", ".join(sorted(labels))
            self.current_model_label.setText(
                self.tr("Current Model: {model}").format(model=display)
            )
            self.current_model_status_label.setText(
                self.tr("Status: Loaded, ready for segmentation")
            )
            self.unload_button.setEnabled(True)
        else:
            self.current_model_label.setText(self.tr("Current Model: —"))
            self.current_model_status_label.setText(
                self.tr("Status: No model loaded")
            )
            self.unload_button.setEnabled(False)

    def _selected_model_id(self) -> str | None:
        """Return the model id of the currently selected table row."""
        selected = self.model_table.selectedItems()
        if not selected:
            return None
        row_index = selected[0].row()
        name_item = self.model_table.item(row_index, self.COL_NAME)
        if name_item is None:
            return None
        return str(name_item.data(Qt.ItemDataRole.UserRole))

    def download_model_by_id(self, model_id: str) -> None:
        """Start downloading a model by its identifier.

        Parameters
        ----------
        model_id : str
            Model identifier to download.

        """
        if self._model_download_thread is not None:
            return
        self.model_download_status_label.setText(
            self.tr("Downloading {model_id}...").format(model_id=model_id)
        )
        self.model_download_progress.setVisible(True)
        self.close_button.setEnabled(False)
        self.model_table.setEnabled(False)
        self.model_dir_edit.setEnabled(False)

        thread = ModelDownloadThread(model_id, self)
        thread.succeeded.connect(self._download_model_succeeded)
        thread.failed.connect(self._download_model_failed)
        thread.finished.connect(self._download_model_finished)
        self._model_download_thread = thread
        thread.start()

    def delete_model_by_id(self, model_id: str) -> None:
        """Confirm and delete a model checkpoint by its identifier.

        Parameters
        ----------
        model_id : str
            Model identifier to delete.

        """
        answer = QMessageBox.question(
            self,
            self.tr("Delete Model"),
            self.tr("Delete the selected checkpoint from the local model folder?"),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        delete_model(model_id)
        self.refresh_model_list()

    def _download_model_succeeded(self, checkpoint_path: str) -> None:
        """Handle a successful background model download."""
        self.model_download_status_label.setText(self.tr("Download completed."))
        MessageTool.MessageBar(
            "Geo-SAM",
            self.tr("Model downloaded to {path}").format(path=checkpoint_path),
            level="success",
        )
        self.refresh_model_list()

    def _download_model_failed(self, error_message: str) -> None:
        """Handle a failed background model download."""
        self.model_download_status_label.setText(self.tr("Download failed."))
        MessageTool.MessageBoxOK(
            error_message,
            title=self.tr("Model Download Failed"),
        )

    def _download_model_finished(self) -> None:
        """Restore UI state after the background model download ends."""
        self.model_download_progress.setVisible(False)
        self.close_button.setEnabled(True)
        self.model_table.setEnabled(True)
        self.model_dir_edit.setEnabled(True)
        thread = self._model_download_thread
        self._model_download_thread = None
        if thread is not None:
            thread.deleteLater()
        self.refresh_model_list()

    def unload_current_model_clicked(self) -> None:
        """Release all loaded GeoSAM model sessions from memory."""
        removed_count = release_runtime_models()
        MessageTool.MessageBar(
            "Geo-SAM",
            self.tr("Released {count} loaded model session(s).").format(
                count=removed_count
            ),
            level="info",
        )
        self.refresh_model_list()

    def _on_cache_slider_changed(self, value: int) -> None:
        """Sync the line edit when the slider moves."""
        self.cache_size_edit.setText(str(value))
        self.save_cache_settings()

    def _on_cache_edit_finished(self) -> None:
        """Sync the slider when the line edit is committed."""
        try:
            value = int(self.cache_size_edit.text())
        except ValueError:
            value = self.cache_size_slider.value()
        value = max(100, min(20480, value))
        self.cache_size_slider.setValue(value)
        # slider valueChanged will trigger save_cache_settings

    def save_cache_settings(self) -> None:
        cache_dir = self.cache_dir_edit.text().strip() or str(DEFAULT_CACHE_DIR)
        self.cache_dir_edit.setText(cache_dir)
        performance_mode = str(
            self.performance_mode_combo.currentData() or "balanced"
        ).strip()
        if performance_mode not in PERFORMANCE_MODE_VALUES:
            performance_mode = "balanced"
        self.settings = save_plugin_settings(
            {
                "cache_enabled": self.cache_enabled_checkbox.isChecked(),
                "cache_dir": cache_dir,
                "cache_max_size_mb": self.cache_size_slider.value(),
                "clear_cache_on_plugin_close": self.clear_cache_on_close_checkbox.isChecked(),
                "performance_mode": performance_mode,
            }
        )
        self.refresh_cache_status()

    def browse_cache_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Cache Directory"),
            self.cache_dir_edit.text() or str(DEFAULT_CACHE_DIR),
        )
        if directory:
            self.cache_dir_edit.setText(directory)
            self.save_cache_settings()

    def refresh_cache_status(self) -> None:
        current_size_mb = get_cache_size_bytes(get_cache_directory()) / (1024 * 1024)
        self.cache_status_label.setText(f"{current_size_mb:.1f} MB")

    def clear_cache_clicked(self) -> None:
        """Delete all plugin cache files immediately."""
        removed_count = clear_cache()
        self.refresh_cache_status()
        MessageTool.MessageBar(
            "Geo-SAM",
            self.tr("Deleted {count} cached file(s).").format(count=removed_count),
            level="info",
        )
