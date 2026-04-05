"""Settings dialog for the Geo-SAM plugin."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
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
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .geosam_runtime import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_DIR,
    HELP_LINKS,
    PLUGIN_ROOT,
    cleanup_cache,
    delete_model,
    dependency_status,
    download_model,
    get_cache_directory,
    get_cache_size_bytes,
    get_model_directory,
    get_model_status_rows,
    install_dependencies,
    load_plugin_settings,
    open_path,
    open_url,
    save_plugin_settings,
)
from .messageTool import MessageTool


class GeoSamSettingsDialog(QDialog):
    """Standalone settings dialog for the Geo-SAM plugin."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Geo-SAM Settings")
        self.resize(760, 520)
        self.settings = load_plugin_settings()

        self.tab_widget = QTabWidget(self)
        self.tab_widget.addTab(self._build_dependency_tab(), "Dependencies")
        self.tab_widget.addTab(self._build_model_tab(), "Model Management")
        self.tab_widget.addTab(self._build_cache_tab(), "Cache")
        self.tab_widget.addTab(self._build_help_tab(), "Help")

        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.accept)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(self.tab_widget)
        root_layout.addWidget(close_button, alignment=Qt.AlignRight)

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

        button_row = QHBoxLayout()
        refresh_button = QPushButton("Refresh", tab)
        refresh_button.clicked.connect(self.refresh_dependency_status)
        install_button = QPushButton("Install", tab)
        install_button.clicked.connect(self.install_dependencies_clicked)
        button_row.addWidget(refresh_button)
        button_row.addWidget(install_button)
        button_row.addStretch(1)

        layout.addLayout(self.dependency_form)
        layout.addLayout(button_row)
        layout.addStretch(1)
        return tab

    def _build_model_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        path_group = QGroupBox("Storage", tab)
        path_layout = QGridLayout(path_group)
        self.model_repo_edit = QLineEdit(str(self.settings["model_repo_url"]), tab)
        self.model_repo_edit.editingFinished.connect(self.save_model_repository)
        self.model_dir_edit = QLineEdit(str(self.settings["model_store_dir"]), tab)
        self.model_dir_edit.editingFinished.connect(self.save_model_directory)

        browse_button = QPushButton("Browse", tab)
        browse_button.clicked.connect(self.browse_model_directory)
        open_button = QPushButton("Open Folder", tab)
        open_button.clicked.connect(lambda: open_path(get_model_directory()))

        path_layout.addWidget(QLabel("Repository", tab), 0, 0)
        path_layout.addWidget(self.model_repo_edit, 0, 1, 1, 2)
        path_layout.addWidget(QLabel("Model Folder", tab), 1, 0)
        path_layout.addWidget(self.model_dir_edit, 1, 1)
        path_layout.addWidget(browse_button, 1, 2)
        path_layout.addWidget(open_button, 1, 3)

        list_group = QGroupBox("Models", tab)
        list_layout = QVBoxLayout(list_group)
        self.model_list = QListWidget(tab)
        self.model_list.itemSelectionChanged.connect(self.refresh_model_action_state)
        list_layout.addWidget(self.model_list)

        action_row = QHBoxLayout()
        self.download_button = QPushButton("Download", tab)
        self.download_button.clicked.connect(self.download_selected_model)
        self.delete_button = QPushButton("Delete", tab)
        self.delete_button.clicked.connect(self.delete_selected_model)
        refresh_button = QPushButton("Refresh", tab)
        refresh_button.clicked.connect(self.refresh_model_list)
        action_row.addWidget(self.download_button)
        action_row.addWidget(self.delete_button)
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

        self.cache_status_label = QLabel("", tab)
        cleanup_button = QPushButton("Cleanup Now", tab)
        cleanup_button.clicked.connect(self.cleanup_cache_clicked)

        form_layout.addRow("Cache", self.cache_enabled_checkbox)
        form_layout.addRow("Location", cache_dir_row)
        form_layout.addRow("Max Size (MB)", self.cache_size_box)
        form_layout.addRow("Current Usage", self.cache_status_label)
        form_layout.addRow("", cleanup_button)

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
        ok, output = install_dependencies()
        title = "Dependencies Installed" if ok else "Dependency Installation Failed"
        message = output if output else title
        MessageTool.MessageBoxOK(message, title=title)
        self.refresh_dependency_status()

    def save_model_repository(self) -> None:
        self.settings = save_plugin_settings({
            "model_repo_url": self.model_repo_edit.text().strip(),
        })

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
        self.download_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)

    def _selected_model_id(self) -> str | None:
        item = self.model_list.currentItem()
        if item is None:
            return None
        return str(item.data(Qt.UserRole))

    def download_selected_model(self) -> None:
        model_id = self._selected_model_id()
        if model_id is None:
            return
        try:
            checkpoint_path = download_model(model_id)
        except Exception as exc:
            MessageTool.MessageBoxOK(
                str(exc),
                title="Model Download Failed",
            )
            return
        MessageTool.MessageBar(
            "Geo-SAM",
            f"Model downloaded to {checkpoint_path}",
            level="success",
        )
        self.refresh_model_list()

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

    def save_cache_settings(self) -> None:
        cache_dir = self.cache_dir_edit.text().strip() or str(DEFAULT_CACHE_DIR)
        self.cache_dir_edit.setText(cache_dir)
        self.settings = save_plugin_settings({
            "cache_enabled": self.cache_enabled_checkbox.isChecked(),
            "cache_dir": cache_dir,
            "cache_max_size_mb": self.cache_size_box.value(),
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
