"""Settings dialog for the Geo-SAM plugin."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QProcess, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .messageTool import MessageTool
from .model_manager import delete_model, download_model, get_model_status_rows
from .plugin_settings import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_DIR,
    HELP_LINKS,
    PERFORMANCE_MODE_VALUES,
    PLUGIN_ROOT,
    PREVIEW_RENDER_MODE_VALUES,
    clear_cache,
    cleanup_cache,
    dependency_status,
    get_cache_directory,
    get_cache_size_bytes,
    get_dependency_install_command,
    get_model_directory,
    load_plugin_settings,
    open_path,
    open_url,
    save_plugin_settings,
)
from .geosam_runtime import release_runtime_models


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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Geo-SAM Settings")
        self.resize(760, 520)
        self.settings = load_plugin_settings()
        self._dependency_install_process: QProcess | None = None
        self._dependency_install_output_buffer = ""
        self._model_download_thread: ModelDownloadThread | None = None

        self.tab_widget = QTabWidget(self)
        self.tab_widget.addTab(self._build_dependency_tab(), "Dependencies")
        self.tab_widget.addTab(self._build_model_tab(), "Model Management")
        self.tab_widget.addTab(self._build_cache_tab(), "Cache")
        self.tab_widget.addTab(self._build_help_tab(), "Help")

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.accept)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(self.tab_widget)
        root_layout.addWidget(self.close_button, alignment=Qt.AlignRight)

        self.refresh_dependency_status()
        self.refresh_model_list()
        self.refresh_cache_status()

    def _build_dependency_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        self.dependency_form = QFormLayout()
        self.dependency_labels: dict[str, QLabel] = {}
        for module_name in dependency_status():
            label = QLabel("", tab)
            self.dependency_labels[module_name] = label
            self.dependency_form.addRow(module_name, label)

        self.dependency_install_status_label = QLabel("Ready", tab)
        self.dependency_form.addRow("Install Status", self.dependency_install_status_label)

        button_row = QHBoxLayout()
        self.refresh_dependencies_button = QPushButton("Refresh", tab)
        self.refresh_dependencies_button.clicked.connect(self.refresh_dependency_status)
        self.install_dependencies_button = QPushButton("Install", tab)
        self.install_dependencies_button.clicked.connect(self.install_dependencies_clicked)
        button_row.addWidget(self.refresh_dependencies_button)
        button_row.addWidget(self.install_dependencies_button)
        button_row.addStretch(1)

        self.dependency_install_progress = QProgressBar(tab)
        self.dependency_install_progress.setRange(0, 0)
        self.dependency_install_progress.setVisible(False)

        self.dependency_install_log = QPlainTextEdit(tab)
        self.dependency_install_log.setReadOnly(True)
        self.dependency_install_log.setPlaceholderText(
            "Dependency installation output will appear here."
        )
        self.dependency_install_log.setMinimumHeight(180)

        layout.addLayout(self.dependency_form)
        layout.addLayout(button_row)
        layout.addWidget(self.dependency_install_progress)
        layout.addWidget(self.dependency_install_log)
        return tab

    def _build_model_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        path_group = QGroupBox("Storage", tab)
        path_layout = QGridLayout(path_group)
        self.model_dir_edit = QLineEdit(str(self.settings["model_store_dir"]), tab)
        self.model_dir_edit.editingFinished.connect(self.save_model_directory)

        browse_button = QPushButton("Browse", tab)
        browse_button.clicked.connect(self.browse_model_directory)
        open_button = QPushButton("Open Folder", tab)
        open_button.clicked.connect(lambda: open_path(get_model_directory()))

        path_layout.addWidget(QLabel("Model Folder", tab), 0, 0)
        path_layout.addWidget(self.model_dir_edit, 0, 1)
        path_layout.addWidget(browse_button, 0, 2)
        path_layout.addWidget(open_button, 0, 3)

        list_group = QGroupBox("Models", tab)
        list_layout = QVBoxLayout(list_group)
        self.model_list = QListWidget(tab)
        self.model_list.itemSelectionChanged.connect(self.refresh_model_action_state)
        list_layout.addWidget(self.model_list)
        self.model_download_status_label = QLabel("Ready", tab)
        list_layout.addWidget(self.model_download_status_label)
        self.model_download_progress = QProgressBar(tab)
        self.model_download_progress.setRange(0, 0)
        self.model_download_progress.setVisible(False)
        list_layout.addWidget(self.model_download_progress)

        action_row = QHBoxLayout()
        self.download_button = QPushButton("Download", tab)
        self.download_button.clicked.connect(self.download_selected_model)
        self.delete_button = QPushButton("Delete", tab)
        self.delete_button.clicked.connect(self.delete_selected_model)
        release_button = QPushButton("Release Loaded Models", tab)
        release_button.clicked.connect(self.release_loaded_models_clicked)
        refresh_button = QPushButton("Refresh", tab)
        refresh_button.clicked.connect(self.refresh_model_list)
        action_row.addWidget(self.download_button)
        action_row.addWidget(self.delete_button)
        action_row.addWidget(release_button)
        action_row.addWidget(refresh_button)
        action_row.addStretch(1)

        layout.addWidget(path_group)
        layout.addWidget(list_group)
        layout.addLayout(action_row)
        return tab

    def _build_cache_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        form_layout = QFormLayout()
        self.cache_enabled_checkbox = QCheckBox("Enable cache", tab)
        self.cache_enabled_checkbox.setChecked(bool(self.settings["cache_enabled"]))
        self.cache_enabled_checkbox.toggled.connect(self.save_cache_settings)

        cache_dir_row = QHBoxLayout()
        self.cache_dir_edit = QLineEdit(str(self.settings["cache_dir"]), tab)
        self.cache_dir_edit.editingFinished.connect(self.save_cache_settings)
        cache_browse_button = QPushButton("Browse", tab)
        cache_browse_button.clicked.connect(self.browse_cache_directory)
        cache_open_button = QPushButton("Open Folder", tab)
        cache_open_button.clicked.connect(lambda: open_path(get_cache_directory()))
        cache_dir_row.addWidget(self.cache_dir_edit)
        cache_dir_row.addWidget(cache_browse_button)
        cache_dir_row.addWidget(cache_open_button)

        self.cache_size_box = QSpinBox(tab)
        self.cache_size_box.setRange(100, 20480)
        self.cache_size_box.setValue(int(self.settings["cache_max_size_mb"]))
        self.cache_size_box.valueChanged.connect(self.save_cache_settings)

        self.performance_mode_combo = QComboBox(tab)
        self.performance_mode_combo.addItem("Balanced", "balanced")
        self.performance_mode_combo.addItem("Fastest", "fastest")
        self.performance_mode_combo.addItem("Low Memory", "low_memory")
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

        self.preview_render_mode_combo = QComboBox(tab)
        self.preview_render_mode_combo.addItem("Light Preview", "light")
        self.preview_render_mode_combo.addItem("Exact Preview", "exact")
        self.preview_render_mode_combo.setCurrentIndex(
            max(
                0,
                self.preview_render_mode_combo.findData(
                    str(self.settings.get("preview_render_mode", "light"))
                ),
            )
        )
        self.preview_render_mode_combo.currentIndexChanged.connect(
            self.save_cache_settings
        )

        self.clear_cache_on_close_checkbox = QCheckBox(
            "Clear cache when the plugin closes",
            tab,
        )
        self.clear_cache_on_close_checkbox.setChecked(
            bool(self.settings.get("clear_cache_on_plugin_close", True))
        )
        self.clear_cache_on_close_checkbox.toggled.connect(self.save_cache_settings)

        self.cache_status_label = QLabel("", tab)
        cleanup_button = QPushButton("Cleanup Now", tab)
        cleanup_button.clicked.connect(self.cleanup_cache_clicked)
        clear_button = QPushButton("Clear All Cache", tab)
        clear_button.clicked.connect(self.clear_cache_clicked)

        form_layout.addRow("Cache", self.cache_enabled_checkbox)
        form_layout.addRow("Location", cache_dir_row)
        form_layout.addRow("Max Size (MB)", self.cache_size_box)
        form_layout.addRow("Performance", self.performance_mode_combo)
        form_layout.addRow("Preview Rendering", self.preview_render_mode_combo)
        form_layout.addRow("Close Behavior", self.clear_cache_on_close_checkbox)
        form_layout.addRow("Current Usage", self.cache_status_label)
        form_layout.addRow("", cleanup_button)
        form_layout.addRow("", clear_button)

        layout.addLayout(form_layout)
        layout.addStretch(1)
        return tab

    def _build_help_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        metadata_path = PLUGIN_ROOT / "metadata.txt"
        version_label = QLabel(self._plugin_version_text(metadata_path), tab)
        version_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(version_label)

        for label_text, url in HELP_LINKS.items():
            button = QPushButton(label_text, tab)
            button.clicked.connect(lambda _checked=False, target=url: open_url(target))
            layout.addWidget(button)

        layout.addStretch(1)
        return tab

    @staticmethod
    def _plugin_version_text(metadata_path: Path) -> str:
        version = "unknown"
        if metadata_path.exists():
            for line in metadata_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("version="):
                    version = line.split("=", 1)[1].strip()
                    break
        return f"Geo-SAM Version: {version}"

    def refresh_dependency_status(self) -> None:
        for module_name, installed in dependency_status().items():
            self.dependency_labels[module_name].setText(
                "Installed" if installed else "Missing"
            )

    def install_dependencies_clicked(self) -> None:
        if self._dependency_install_process is not None:
            return

        self.dependency_install_log.clear()
        self.dependency_install_status_label.setText("Installing...")
        self.dependency_install_progress.setVisible(True)
        self.install_dependencies_button.setEnabled(False)
        self.refresh_dependencies_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self._dependency_install_output_buffer = ""
        self._append_dependency_install_log("Starting dependency installation...")
        try:
            command = get_dependency_install_command()
        except RuntimeError as exc:
            self._finish_dependency_install(False, str(exc))
            return

        process = QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setProcessChannelMode(QProcess.MergedChannels)
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
        while "\n" in self._dependency_install_output_buffer:
            line, self._dependency_install_output_buffer = (
                self._dependency_install_output_buffer.split("\n", maxsplit=1)
            )
            self._append_dependency_install_log(line.rstrip("\r"))

    def _dependency_install_finished(self, exit_code: int, _exit_status: int) -> None:
        """Handle completion of the asynchronous dependency install command."""
        if self._dependency_install_process is None:
            return

        if self._dependency_install_output_buffer:
            self._append_dependency_install_log(
                self._dependency_install_output_buffer.rstrip("\r\n")
            )
            self._dependency_install_output_buffer = ""

        output = self.dependency_install_log.toPlainText().strip()
        self._finish_dependency_install(exit_code == 0, output)

    def _dependency_install_error(self, _error: QProcess.ProcessError) -> None:
        """Handle startup or runtime failures from the dependency installer."""
        process = self._dependency_install_process
        if process is None:
            return

        self._read_dependency_install_output()
        error_message = process.errorString().strip() or (
            "Dependency installer process failed to start."
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
            "Installed" if ok else "Failed"
        )
        self._dependency_install_process = None

        if ok and not output.strip():
            self._append_dependency_install_log("Dependency installation completed.")
        elif not ok and not output.strip():
            self._append_dependency_install_log("Dependency installation failed.")
        self.refresh_dependency_status()

    def _append_dependency_install_log(self, message: str) -> None:
        """Append a dependency installation log line to the settings dialog."""
        if not message:
            return
        self.dependency_install_log.appendPlainText(message)
        scrollbar = self.dependency_install_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

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
            "Select Model Directory",
            self.model_dir_edit.text() or str(DEFAULT_MODEL_DIR),
        )
        if directory:
            self.model_dir_edit.setText(directory)
            self.save_model_directory()

    def refresh_model_list(self) -> None:
        self.model_list.clear()
        for row in get_model_status_rows():
            status_text = "Downloaded" if row["downloaded"] else "Missing"
            item = QListWidgetItem(
                f"{row['label']} [{row['model_type']}] - {status_text}",
                self.model_list,
            )
            item.setData(Qt.UserRole, row["model_id"])
        self.refresh_model_action_state()

    def refresh_model_action_state(self) -> None:
        has_selection = self.model_list.currentItem() is not None
        is_downloading = self._model_download_thread is not None
        self.download_button.setEnabled(has_selection and not is_downloading)
        self.delete_button.setEnabled(has_selection and not is_downloading)

    def _selected_model_id(self) -> str | None:
        item = self.model_list.currentItem()
        if item is None:
            return None
        return str(item.data(Qt.UserRole))

    def download_selected_model(self) -> None:
        model_id = self._selected_model_id()
        if model_id is None or self._model_download_thread is not None:
            return
        self.model_download_status_label.setText(f"Downloading {model_id}...")
        self.model_download_progress.setVisible(True)
        self.close_button.setEnabled(False)
        self.model_list.setEnabled(False)
        self.model_dir_edit.setEnabled(False)
        self.download_button.setEnabled(False)
        self.delete_button.setEnabled(False)

        thread = ModelDownloadThread(model_id, self)
        thread.succeeded.connect(self._download_model_succeeded)
        thread.failed.connect(self._download_model_failed)
        thread.finished.connect(self._download_model_finished)
        self._model_download_thread = thread
        thread.start()

    def _download_model_succeeded(self, checkpoint_path: str) -> None:
        """Handle a successful background model download."""
        self.model_download_status_label.setText("Download completed.")
        MessageTool.MessageBar(
            "Geo-SAM",
            f"Model downloaded to {checkpoint_path}",
            level="success",
        )
        self.refresh_model_list()

    def _download_model_failed(self, error_message: str) -> None:
        """Handle a failed background model download."""
        self.model_download_status_label.setText("Download failed.")
        MessageTool.MessageBoxOK(
            error_message,
            title="Model Download Failed",
        )

    def _download_model_finished(self) -> None:
        """Restore UI state after the background model download ends."""
        self.model_download_progress.setVisible(False)
        self.close_button.setEnabled(True)
        self.model_list.setEnabled(True)
        self.model_dir_edit.setEnabled(True)
        thread = self._model_download_thread
        self._model_download_thread = None
        if thread is not None:
            thread.deleteLater()
        self.refresh_model_action_state()

    def delete_selected_model(self) -> None:
        model_id = self._selected_model_id()
        if model_id is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete Model",
            "Delete the selected checkpoint from the local model folder?",
        )
        if answer != QMessageBox.Yes:
            return
        delete_model(model_id)
        self.refresh_model_list()

    def release_loaded_models_clicked(self) -> None:
        """Release loaded GeoSAM model sessions from memory."""
        removed_count = release_runtime_models()
        MessageTool.MessageBar(
            "Geo-SAM",
            f"Released {removed_count} loaded model session(s).",
            level="info",
        )

    def save_cache_settings(self) -> None:
        cache_dir = self.cache_dir_edit.text().strip() or str(DEFAULT_CACHE_DIR)
        self.cache_dir_edit.setText(cache_dir)
        performance_mode = str(
            self.performance_mode_combo.currentData() or "balanced"
        ).strip()
        if performance_mode not in PERFORMANCE_MODE_VALUES:
            performance_mode = "balanced"
        preview_render_mode = str(
            self.preview_render_mode_combo.currentData() or "light"
        ).strip()
        if preview_render_mode not in PREVIEW_RENDER_MODE_VALUES:
            preview_render_mode = "light"
        self.settings = save_plugin_settings({
            "cache_enabled": self.cache_enabled_checkbox.isChecked(),
            "cache_dir": cache_dir,
            "cache_max_size_mb": self.cache_size_box.value(),
            "clear_cache_on_plugin_close": self.clear_cache_on_close_checkbox.isChecked(),
            "performance_mode": performance_mode,
            "preview_render_mode": preview_render_mode,
        })
        self.refresh_cache_status()

    def browse_cache_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Cache Directory",
            self.cache_dir_edit.text() or str(DEFAULT_CACHE_DIR),
        )
        if directory:
            self.cache_dir_edit.setText(directory)
            self.save_cache_settings()

    def refresh_cache_status(self) -> None:
        current_size_mb = get_cache_size_bytes(get_cache_directory()) / (1024 * 1024)
        self.cache_status_label.setText(f"{current_size_mb:.1f} MB")

    def cleanup_cache_clicked(self) -> None:
        removed_count = cleanup_cache()
        self.refresh_cache_status()
        MessageTool.MessageBar(
            "Geo-SAM",
            f"Removed {removed_count} cached file(s).",
            level="info",
        )

    def clear_cache_clicked(self) -> None:
        """Delete all plugin cache files immediately."""
        removed_count = clear_cache()
        self.refresh_cache_status()
        MessageTool.MessageBar(
            "Geo-SAM",
            f"Deleted {removed_count} cached file(s).",
            level="info",
        )
