"""Settings dialog for the Geo-SAM plugin."""

from __future__ import annotations

import html
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from qgis.PyQt.QtCore import QSize, QThread, Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCloseEvent
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .messageTool import MessageTool
from .i18n import translate
from .dependency_path import (
    clear_all_plugin_managed_site_packages,
    clear_current_plugin_managed_site_packages,
    get_plugin_managed_dependency_stats,
    get_plugin_managed_dependency_stats_dedup,
    get_plugin_managed_site_packages,
    iter_all_plugin_managed_site_packages,
)
from .dependency_installer import (
    build_full_commands,
    build_uv_commands,
    make_plan,
    make_plan_for_copy,
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
    format_dependency_install_command,
    format_dependency_install_commands,
    get_cache_directory,
    get_cache_size_bytes,
    get_model_directory,
    load_plugin_settings,
    open_path,
    open_url,
    save_plugin_settings,
)
from .geosam_runtime import get_loaded_model_ids, release_runtime_models
from .runtime_profile import (
    DEFAULT_PROFILE,
    PROFILE_VALUES,
    TorchProfile,
    commit_switch,
    get_custom_index_url,
    prepare_switch,
    profile_display_name,
    profile_install_status,
    profiles_for_platform,
    set_custom_index_url,
)
from .torch_profile import (
    detect_recommended_profile,
    profile_unavailable_reason,
)

if TYPE_CHECKING:
    from .plugin_settings import DependencyStatusRow


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from terminal output.

    Parameters
    ----------
    text : str
        Raw terminal text that may contain color/cursor escape codes.

    Returns
    -------
    str
        Text with escape sequences removed.

    """
    return _ANSI_ESCAPE_RE.sub("", text)


# SGR color codes uv emits, mapped to CSS for :func:`_ansi_to_html`.
# uv only uses this small set (verified against real PTY output): no 256-color
# or truecolor sequences, so a fixed table is sufficient and dependency-free.
_SGR_STYLE: dict[str, str] = {
    "2": "color:#888;",  # dim/faint
    "32": "color:#008c4a;",  # green (matches plugin's installed-success hex)
    "37": "color:#c0c0c0;",  # white (light gray on dark; readable on light bg)
    "30": "color:#444;",  # black (used inside progress bars)
}

_SGR_SPLIT_RE = re.compile(r"(\x1b\[[0-9;]*m)")


def _ansi_to_html(text: str) -> str:
    """Convert a terminal line with SGR escapes to an HTML fragment.

    Handles uv's color set: dim (``2``), bold (``1``), green (``32``), white
    (``37``), black (``30``), and reset (``0`` / ``39``). Text is HTML-escaped,
    so the fragment is safe to embed.

    Parameters
    ----------
    text : str
        One terminal row that may contain SGR escape sequences.

    Returns
    -------
    str
        HTML fragment (without surrounding tags) ready for :func:`setHtml`.

    """
    out: list[str] = []
    current_style = ""
    current_bold = False
    open_span = False
    # The (style, bold) actually declared by the currently-open span. A
    # mid-line SGR color change (e.g. uv's progress bar going green → dim-gray
    # for the empty portion) must close and reopen the span; if we only track
    # whether *a* span is open, the new color is stored but never emitted and
    # all following text inherits the first color.
    emitted_style = ""
    emitted_bold = False

    def close_span() -> None:
        nonlocal open_span, emitted_style, emitted_bold
        if open_span:
            out.append("</span>")
            open_span = False
            emitted_style = ""
            emitted_bold = False

    def sync_span() -> None:
        """Open/close/reopen the span to match the active attributes."""
        nonlocal open_span, emitted_style, emitted_bold
        if not current_style and not current_bold:
            close_span()
            return
        if (
            open_span
            and current_style == emitted_style
            and current_bold == emitted_bold
        ):
            return  # already showing exactly the desired style
        # Desired style differs from what's open: close + reopen so the new
        # color/bold actually takes effect on subsequent text.
        close_span()
        attrs: list[str] = []
        if current_style:
            attrs.append(current_style)
        if current_bold:
            attrs.append("font-weight:bold;")
        out.append(f'<span style="{"".join(attrs)}">')
        open_span = True
        emitted_style = current_style
        emitted_bold = current_bold

    for token in _SGR_SPLIT_RE.split(text):
        if not token:
            continue
        if _ANSI_ESCAPE_RE.fullmatch(token) is not None:
            code = token[2:-1]
            if code in ("0", ""):
                close_span()
                current_style = ""
                current_bold = False
            elif code == "1":
                current_bold = True
                sync_span()
            elif code == "22":
                current_bold = False
                sync_span()
            elif code in _SGR_STYLE:
                current_style = _SGR_STYLE[code]
                sync_span()
            elif code == "39":
                close_span()
                current_style = ""
            # other SGR codes are ignored (stripped, no style change)
            continue
        out.append(html.escape(token))

    close_span()
    return "".join(out)


# Cursor-control escape CSI sequence, e.g. ``\x1b[9A`` (up 9) or ``\x1b[2K``
# (erase line). Captured so we can interpret cursor motion instead of stripping
# it blindly — uv redraws a multi-row progress display by moving the cursor up
# and rewriting each row.
_CSI_RE = re.compile(r"\x1b\[(\d*)([A-Za-z])")

# Matches a trailing *unterminated* CSI in a chunk: ``ESC`` alone, ``ESC [``,
# ``ESC [ 1`` ... up to but not including the final letter. The worker reads the
# PTY in fixed-size chunks, so a sequence like ``\x1b[37m`` can be split into
# ``\x1b[3`` + ``7m`` across two reads. If the partial tail were written into
# the screen buffer, a snapshot taken at that instant would emit garbage like
# ``\x1b[3`` as a visible row (verified against real uv output). We detect such
# tails and buffer them for the next :meth:`_ProgressTerminal.feed` call.
_PARTIAL_CSI_TAIL_RE = re.compile(r"\x1b(?:\[(?:[0-9;?]*))?$")


class WarningBanner(QFrame):
    """Inline banner combining a QGIS theme warning icon and styled text.

    Uses ``QIcon.pixmap`` with the widget's :func:`devicePixelRatio` so the
    SVG is rasterized at native HiDPI resolution — keeping the icon crisp on
    retina displays (a plain rich-text ``<img>`` renders at 1x and appears
    blurry/jagged).

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.

    """

    _ICON_SIZE = 8
    _WARNING_COLOR = "#b00020"
    _INFO_COLOR = "#0b5394"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icon_label = QLabel(self)
        self._icon_label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        )
        self._icon_label.setContentsMargins(2, 2, 2, 2)
        self._text_label = QLabel(self)
        self._text_label.setWordWrap(True)
        self._text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        layout.addWidget(
            self._icon_label,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )
        layout.addWidget(self._text_label, stretch=1)
        self._text_label.setStyleSheet(
            f"color:{self._WARNING_COLOR}; font-weight:bold;"
        )
        self._refresh_icon()
        self.setPlain("")

    def _refresh_icon(self, icon_name: str = "mIconWarning.svg") -> None:
        """Render a QGIS theme icon at the current device pixel ratio.

        Parameters
        ----------
        icon_name : str
            Theme icon file name, e.g. ``"mIconWarning.svg"`` or
            ``"mIconInfo.svg"``. Falls back to a blank icon when the
            theme lacks the requested asset.

        """
        from qgis.core import QgsApplication

        icon = QgsApplication.getThemeIcon(icon_name)
        if icon.isNull():
            self._icon_label.clear()
            return
        dpr = self.devicePixelRatioF()
        physical_size = QSize(
            int(self._ICON_SIZE * dpr), int(self._ICON_SIZE * dpr)
        )
        pixmap = icon.pixmap(physical_size)
        pixmap.setDevicePixelRatio(dpr)
        self._icon_label.setPixmap(pixmap)

    def setWarning(self, text: str) -> None:
        """Show the banner with a warning icon and red bold text.

        Parameters
        ----------
        text : str
            Plain-text warning message (HTML escaping is applied internally).

        """
        self._refresh_icon("mIconWarning.svg")
        self._icon_label.setVisible(True)
        self._text_label.setStyleSheet(
            f"color:{self._WARNING_COLOR}; font-weight:bold;"
        )
        self._text_label.setText(text)

    def setInfo(self, text: str) -> None:
        """Show the banner with an info icon and blue bold text.

        Used for advisory messages that are not errors (e.g. dedup
        savings, APFS clone caveats). The blue colour and ``mIconInfo``
        icon distinguish it from genuine errors shown via
        :meth:`setWarning`.

        Parameters
        ----------
        text : str
            Plain-text advisory message.

        """
        self._refresh_icon("mIconInfo.svg")
        self._icon_label.setVisible(True)
        self._text_label.setStyleSheet(
            f"color:{self._INFO_COLOR}; font-weight:bold;"
        )
        self._text_label.setText(text)

    def setPlain(self, text: str) -> None:
        """Show plain non-warning text without the icon.

        Parameters
        ----------
        text : str
            Plain-text status message.

        """
        self._icon_label.clear()
        self._icon_label.setVisible(False)
        self._text_label.setStyleSheet("")
        self._text_label.setText(text)

    def clear(self) -> None:
        """Hide all content."""
        self._icon_label.clear()
        self._icon_label.setVisible(False)
        self._text_label.setText("")


class _ProgressTerminal:
    """Minimal terminal emulator that mirrors uv's screen state.

    ``uv`` (and other modern CLIs) render download progress as a multi-row
    block: on every tick it moves the cursor up N rows (``\\x1b[nA``), erases
    each row (``\\x1b[2K``), and rewrites all rows. A naive line splitter that
    only understands ``\\r`` and ``\\n`` therefore re-emits every row on every
    tick, producing a flood of duplicate progress lines.

    This class emulates just enough of the cursor model (rows, ``\\r``,
    ``\\n``, cursor up/down, erase-line) to keep a coherent screen buffer, then
    exposes :meth:`snapshot` so the caller can render the **full screen** on
    each tick — a complete replacement that avoids both the duplicate-line and
    partial-fragment problems of incremental per-line diffing.

    Notes
    -----
    Only the cursor-motion subset that uv uses for progress (``A`` / ``B`` /
    ``K``) is interpreted; SGR color escapes (``m``) are left in the stored
    rows so :func:`_ansi_to_html` can render uv's native colors. Partial CSI
    sequences split across PTY read boundaries are buffered and rejoined.

    """

    def __init__(self) -> None:
        # Parallel lists: row text and whether the row is "finalized" (the
        # cursor has advanced past it via a real ``\n``).
        self._rows: list[str] = [""]
        self._finalized: list[bool] = [False]
        self._cur_row = 0
        self._cur_col = 0
        # Trailing partial CSI held back from the previous chunk. The PTY is
        # read in fixed-size blocks, so an escape like ``\x1b[37m`` can be split
        # (``\x1b[3`` + ``7m``); we rejoin it on the next feed instead of
        # writing the fragment into the screen buffer.
        self._pending = ""
        # Frame-completeness gate (anti-flicker). uv redraws its multi-row
        # progress block by moving the cursor up N rows (``\x1b[nA``), erasing,
        # and rewriting each row. If a snapshot is taken mid-redraw — cursor
        # rewound but rows not yet rewritten — the display flickers (rows
        # vanish or show stale-then-new). ``_high_water_row`` tracks the
        # furthest row the cursor has reached; while ``_cur_row`` is above it
        # the terminal is mid-redraw and :meth:`snapshot_stable` returns the
        # last complete frame instead of a half-drawn one.
        self._high_water_row = 0
        self._stable_snapshot: list[str] | None = None

    def _ensure_row(self, row: int) -> None:
        while len(self._rows) <= row:
            self._rows.append("")
            self._finalized.append(False)

    def feed(self, text: str) -> None:
        """Ingest a chunk of terminal output into the screen buffer.

        Parameters
        ----------
        text : str
            Decoded PTY output. A chunk may end in the middle of a CSI escape
            (e.g. ``\x1b[3``); the fragment is buffered internally and rejoined
            with the next call, so it never appears as a visible screen row.

        """
        # Rejoin any partial escape held from the previous chunk before parsing.
        text = self._pending + text
        self._pending = ""
        # If the chunk ends mid-escape, hold the tail for the next feed() so the
        # fragment is not written into the screen buffer (which would surface as
        # a garbage snapshot line during a fast multi-package download).
        tail = _PARTIAL_CSI_TAIL_RE.search(text)
        if tail is not None:
            self._pending = text[tail.start() :]
            text = text[: tail.start()]

        i = 0
        n = len(text)
        while i < n:
            m = _CSI_RE.match(text, i)
            if m is not None:
                count = int(m.group(1)) if m.group(1) else 1
                cmd = m.group(2)
                if cmd == "A":  # cursor up
                    self._cur_row = max(0, self._cur_row - count)
                    i = m.end()
                    continue
                if cmd == "B":  # cursor down
                    self._cur_row += count
                    self._ensure_row(self._cur_row)
                    i = m.end()
                    continue
                if cmd == "K":  # erase line (2 = whole line)
                    self._ensure_row(self._cur_row)
                    self._rows[self._cur_row] = ""
                    self._finalized[self._cur_row] = False
                    self._cur_col = 0
                    i = m.end()
                    continue
                # SGR (m) and other non-cursor CSI: leave in the stored text
                # so :func:`_ansi_to_html` can render uv's native colors.

            ch = text[i]
            if ch == "\r":
                self._cur_col = 0
            elif ch == "\n":
                # A real newline finalizes the current row (uv writes a
                # completed summary line then ``\r\n`` before starting the
                # next phase). The row above the cursor is now permanent.
                self._finalized[self._cur_row] = True
                self._cur_row += 1
                self._ensure_row(self._cur_row)
                self._cur_col = 0
                # The cursor just landed on a fresh row at the bottom of the
                # written content — update the high-water mark so the
                # frame-completeness gate treats a fully-rewritten frame as
                # stable.
                if self._cur_row > self._high_water_row:
                    self._high_water_row = self._cur_row
            elif ch == "\b":
                if self._cur_col > 0:
                    self._cur_col -= 1
            else:
                self._ensure_row(self._cur_row)
                row = self._rows[self._cur_row]
                if self._cur_col >= len(row):
                    row = row + " " * (self._cur_col - len(row)) + ch
                else:
                    row = row[: self._cur_col] + ch + row[self._cur_col + 1 :]
                self._rows[self._cur_row] = row
                self._cur_col += 1
            i += 1

    def snapshot(self) -> list[str]:
        """Return the current visible screen as a list of non-empty rows.

        Each row keeps its SGR color escapes so the dialog can render uv's
        native colors via :func:`_ansi_to_html`. This is a **full screen
        replacement** — the caller overwrites its entire display with this
        snapshot, which eliminates the duplicate-line and partial-fragment
        problems that incremental per-line diffing caused.

        Trailing spaces are **preserved** (only trailing ``\\r``/``\\n`` are
        stripped): uv pads each multi-package progress line with trailing
        spaces so the speed/total column right-aligns across packages, and
        stripping that padding would scramble the alignment. The render uses
        ``white-space: pre-wrap`` so the spaces render at a uniform width.

        Returns
        -------
        list[str]
            Non-empty screen rows (ANSI escapes preserved, trailing spaces
            preserved), top to bottom.

        """
        return [
            raw.rstrip("\r\n")
            for raw in self._rows
            if _strip_ansi(raw).strip()
        ]

    def snapshot_stable(self) -> list[str]:
        """Return a frame-completeness-gated snapshot to avoid flicker.

        uv redraws its multi-row progress block by rewinding the cursor
        (``\\x1b[nA``) and rewriting each row. A snapshot taken mid-redraw —
        after the rewind but before all rows are rewritten — shows a
        half-drawn screen (rows vanish or briefly show stale content), which
        the user perceives as flicker.

        This method suppresses those mid-redraw snapshots: while the cursor is
        above the high-water row (``_cur_row < _high_water_row``), it returns
        the last fully-drawn frame from the cache instead of the current
        half-drawn screen. Once the cursor returns to the bottom (redraw
        complete), it refreshes the cache.

        The very first frame has no cache, so it is emitted even if mid-redraw
        — a stale frame is better than an empty display.

        Returns
        -------
        list[str]
            The last complete screen snapshot, or the current snapshot if the
            cursor is at the bottom (redraw finished) or no cached frame
            exists yet.

        """
        if (
            self._cur_row < self._high_water_row
            and self._stable_snapshot is not None
        ):
            return self._stable_snapshot
        self._stable_snapshot = self.snapshot()
        return self._stable_snapshot


class InstallProcessWorker(QThread):
    """Background worker that runs a subprocess with a PTY for progress output.

    On Unix (macOS/Linux), a pseudo-terminal is allocated so that uv detects
    a real TTY and emits progress bars (e.g., ``  45% ━━━━━╸━━━━━``).
    ``QProcess`` pipes stdout as a non-TTY, which makes uv suppress progress
    entirely — the UI appears frozen during long downloads.

    On Windows where ``pty`` is unavailable, a regular pipe is used as
    fallback (no progress bars, but output still streams).

    Signals
    ------
    screen_output(list)
        Emitted with the full list of visible screen rows (ANSI escapes
        preserved). The dialog replaces its entire display with this
        snapshot. Throttled to ~6/sec so a fast animation does not flood
        the GUI thread.
    install_finished(int)
        Emitted with the process exit code when the subprocess ends.
    start_failed(str)
        Emitted when the subprocess could not be started.

    """

    screen_output = pyqtSignal(list)
    install_finished = pyqtSignal(int)
    start_failed = pyqtSignal(str)

    def __init__(self, command: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._command = command
        self._process: subprocess.Popen | None = None
        self._cancel_requested = False
        # Throttle transient progress signals so a 60 Hz animation does not
        # flood the GUI thread. Permanent (appended) lines are always emitted
        # immediately; only repeated progress (replace) frames are coalesced.
        self._progress_emit_monotonic: float = 0.0

    def run(self) -> None:
        """Run the subprocess and stream output via signals."""
        try:
            if sys.platform != "win32":
                self._run_with_pty()
            else:
                self._run_with_pipe()
        except Exception as exc:
            self.start_failed.emit(str(exc))

    @property
    def is_cancel_requested(self) -> bool:
        """Return whether cooperative cancellation was requested."""
        return self._cancel_requested

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        """Return an environment that lets uv render progress bars.

        QGIS launches plugins with ``TERM=dumb`` (and sometimes ``CI=1``).
        Even though this worker runs uv behind a pseudo-terminal, uv inspects
        ``TERM`` and refuses to emit spinners or download progress bars when
        it is ``dumb``. The install then appears frozen on the first line
        (``Using CPython ...``) for the entire download — there is no output
        until resolution finishes.

        We force a capable ``TERM`` and a few color/progress controls so the
        progress stream the worker is designed to surface actually arrives.
        Only progress/color related variables are overridden; everything else
        (``PATH``, ``HOME``, ``UV_*``, ``SSL_CERT_FILE``, ...) is inherited
        unchanged from the QGIS process.

        Returns
        -------
        dict[str, str]
            Environment mapping for :class:`subprocess.Popen`.

        """
        env = dict(os.environ)
        # ``dumb`` makes uv (and other modern CLIs) disable all progress
        # rendering. A real PTY can handle the common 256-color profile.
        if env.get("TERM") in (None, "", "dumb"):
            env["TERM"] = "xterm-256color"
        # uv enables a richer progress animation when it thinks a human is
        # watching; these help under PTY without harming pipes.
        env.setdefault("CLICOLOR_FORCE", "1")
        env.pop("NO_COLOR", None)
        return env

    def _run_with_pty(self) -> None:
        """Run the subprocess with a pseudo-terminal on Unix."""
        import select
        import pty

        master_fd, slave_fd = pty.openpty()
        try:
            self._process = subprocess.Popen(
                self._command,
                stdout=slave_fd,
                stderr=slave_fd,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                env=self._subprocess_env(),
            )
        except (OSError, FileNotFoundError) as exc:
            os.close(master_fd)
            os.close(slave_fd)
            self.start_failed.emit(str(exc))
            return
        os.close(slave_fd)

        terminal = _ProgressTerminal()
        while True:
            if self._cancel_requested:
                break

            if self._process.poll() is not None:
                try:
                    remaining = os.read(master_fd, 65536)
                    if remaining:
                        terminal.feed(remaining.decode("utf-8", errors="replace"))
                except OSError:
                    pass
                break

            try:
                ready, _, _ = select.select([master_fd], [], [], 0.1)
            except (OSError, ValueError):
                break

            if ready:
                try:
                    chunk = os.read(master_fd, 4096)
                    if not chunk:
                        break
                    terminal.feed(chunk.decode("utf-8", errors="replace"))
                except OSError:
                    break

            self._emit_snapshot(terminal)

        self._emit_snapshot(terminal, force=True)
        os.close(master_fd)
        exit_code = self._process.wait()
        self.install_finished.emit(exit_code)

    def _emit_snapshot(self, terminal: _ProgressTerminal, *, force: bool = False) -> None:
        """Emit a throttled full-screen snapshot via ``screen_output``.

        Parameters
        ----------
        terminal : _ProgressTerminal
            Terminal emulator to snapshot.
        force : bool, optional
            When ``True``, bypass the throttle (used on process exit so the
            final state is always delivered).

        """
        import time

        now = time.monotonic()
        if not force and now - self._progress_emit_monotonic < 0.15:
            return
        self._progress_emit_monotonic = now
        # Use the frame-completeness-gated snapshot so mid-redraw frames (uv's
        # cursor-up rewind + erase) don't flicker on screen. On ``force`` (process
        # exit) we still want the final frame even if the gate has not settled,
        # so bypass the cache by reading the raw snapshot when forced.
        snapshot = terminal.snapshot() if force else terminal.snapshot_stable()
        self.screen_output.emit(snapshot)

    def _run_with_pipe(self) -> None:
        """Run the subprocess with a regular pipe on Windows."""
        try:
            self._process = subprocess.Popen(
                self._command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=self._subprocess_env(),
            )
        except (OSError, FileNotFoundError) as exc:
            self.start_failed.emit(str(exc))
            return

        terminal = _ProgressTerminal()
        if self._process.stdout is not None:
            for raw_line in self._process.stdout:
                terminal.feed(raw_line)
                self._emit_snapshot(terminal)
        self._emit_snapshot(terminal, force=True)

        exit_code = self._process.wait()
        self.install_finished.emit(exit_code)

    def cancel(self) -> None:
        """Request cooperative cancellation and terminate the subprocess.

        Sets the cancellation flag so :meth:`run` exits its polling loop, then
        terminates the underlying subprocess. Pair with :meth:`wait` on the
        owning thread to guarantee the worker has fully stopped before the
        dialog is destroyed — otherwise Qt aborts with
        ``QThread: Destroyed while thread is still running``.

        """
        self._cancel_requested = True
        if self._process is not None:
            try:
                self._process.terminate()
            except OSError:
                pass


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
        self._install_worker: InstallProcessWorker | None = None
        self._dependency_install_commands: list[list[str]] = []
        self._dependency_install_command_index = 0
        self._missing_dependency_names: list[str] = []
        self._model_download_thread: ModelDownloadThread | None = None
        # Plugin-generated header lines (commands, status) shown above the
        # live uv terminal snapshot. Kept separate so screen snapshots don't
        # clobber them.
        self._install_header_lines: list[str] = []
        # Latest screen snapshot from the worker (uv's live TUI).
        self._install_log_lines: list[str] = []
        # Coalesce rapid install-output signals into periodic render calls so
        # the GUI thread is not flooded with setHtml() invocations (which freeze
        # the main event loop during a fast multi-package download).
        self._render_pending = False
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(150)
        self._render_timer.timeout.connect(self._flush_install_log)

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

    def done(self, result: int) -> None:
        """Stop background threads before the dialog is destroyed.

        Overrides :meth:`QDialog.done` so that any path that closes the dialog
        — the Close button (``accept``), the window close button, or ``ESC``
        (``reject``) — first tears down the install worker and the model
        download thread. Without this, QGIS shutdown garbage-collects the
        dialog while a :class:`QThread` is still running, which makes Qt abort
        with ``QThread: Destroyed while thread is still running``.

        Parameters
        ----------
        result : int
            The dialog result code passed by Qt.

        """
        self._stop_background_threads()
        super().done(result)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop background threads when the window is requested to close.

        Parameters
        ----------
        event : QCloseEvent
            The close event delivered by Qt.

        """
        self._stop_background_threads()
        super().closeEvent(event)

    def _stop_background_threads(self) -> None:
        """Cancel and join all background worker threads owned by the dialog.

        Guarantees that :class:`InstallProcessWorker` and
        :class:`ModelDownloadThread` have fully stopped before the dialog's
        child widgets (and their owned QThreads) are destroyed. Cancels the
        install subprocess cooperatively, then blocks on :meth:`QThread.wait`
        with a bounded timeout so a wedged worker cannot freeze the UI
        indefinitely.

        """
        worker = self._install_worker
        if worker is not None:
            worker.cancel()
            worker.wait(5000)
            self._install_worker = None

        download_thread = self._model_download_thread
        if download_thread is not None:
            download_thread.wait(5000)
            self._model_download_thread = None

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

        # -- Torch runtime profile selector --------------------------------
        profile_group = QGroupBox(self.tr("Torch Runtime"), tab)
        profile_layout = QFormLayout(profile_group)

        self.torch_profile_combo = QComboBox(tab)
        available_profiles = profiles_for_platform()
        recommended = detect_recommended_profile()
        for profile_key in PROFILE_VALUES:
            display_name = profile_display_name(profile_key)
            if profile_key == recommended:
                display_name = self.tr("{name} (Recommended)").format(
                    name=display_name
                )
            self.torch_profile_combo.addItem(display_name, profile_key)
            idx = self.torch_profile_combo.count() - 1
            if profile_key not in available_profiles:
                self.torch_profile_combo.model().item(idx).setEnabled(False)
                reason = profile_unavailable_reason(profile_key)
                if reason:
                    self.torch_profile_combo.setItemData(
                        idx, reason, Qt.ItemDataRole.ToolTipRole
                    )
        # Set current from settings
        current_profile = self.settings.get("torch_profile", DEFAULT_PROFILE)
        profile_idx = self.torch_profile_combo.findData(current_profile)
        if profile_idx >= 0:
            self.torch_profile_combo.setCurrentIndex(profile_idx)
        self.torch_profile_combo.currentIndexChanged.connect(
            self._on_torch_profile_changed
        )

        self.custom_index_url_edit = QLineEdit(tab)
        self.custom_index_url_edit.setPlaceholderText(
            "https://download.pytorch.org/whl/cu128"
        )
        custom_url = get_custom_index_url() or ""
        self.custom_index_url_edit.setText(custom_url)
        self.custom_index_url_edit.editingFinished.connect(
            self._save_custom_index_url
        )

        self.custom_index_url_label = QLabel(self.tr("Custom Index URL:"), tab)
        is_custom = current_profile == "custom"
        self.custom_index_url_label.setVisible(is_custom)
        self.custom_index_url_edit.setVisible(is_custom)

        self.profile_status_label = QLabel(tab)
        self.profile_status_label.setWordWrap(True)
        self.profile_status_label.setTextFormat(Qt.TextFormat.RichText)

        self.switch_runtime_button = QPushButton(
            self.tr("Switch"), tab
        )
        self.switch_runtime_button.clicked.connect(
            self.switch_runtime_clicked
        )
        runtime_row = QHBoxLayout()
        runtime_row.addWidget(self.torch_profile_combo, stretch=1)
        runtime_row.addWidget(self.switch_runtime_button)
        profile_layout.addRow(self.tr("Runtime:"), runtime_row)
        profile_layout.addRow(self.custom_index_url_label, self.custom_index_url_edit)
        profile_layout.addRow(self.profile_status_label)

        # -- Summary labels ------------------------------------------------
        self.dependency_summary_label = WarningBanner(tab)
        self.dependency_storage_label = QLabel(tab)
        self.dependency_storage_label.setWordWrap(True)
        self.dependency_storage_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.dependency_storage_label.setStyleSheet(
            "padding:2px 4px;"
        )
        self.dependency_storage_dedup_label = WarningBanner(tab)

        # -- Dependency table ----------------------------------------------
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

        # -- Action buttons ------------------------------------------------
        button_row = QHBoxLayout()
        self.install_dependencies_button = QPushButton(
            self.tr("Install Missing"), tab
        )
        self.install_dependencies_button.clicked.connect(
            self.install_dependencies_clicked
        )
        self.copy_install_commands_button = QPushButton(
            self.tr("Copy Install Commands"), tab
        )
        self.copy_install_commands_button.clicked.connect(
            self.copy_install_commands_clicked
        )
        self.refresh_dependencies_button = QPushButton(
            self.tr("Refresh Status"), tab
        )
        self.refresh_dependencies_button.clicked.connect(
            self.refresh_dependency_status
        )
        button_row.addWidget(self.install_dependencies_button)
        button_row.addWidget(self.copy_install_commands_button)
        button_row.addWidget(self.refresh_dependencies_button)

        button_row2 = QHBoxLayout()
        self.open_dependency_folder_button = QPushButton(
            self.tr("Browse Packages"), tab
        )
        self.open_dependency_folder_button.clicked.connect(
            self.open_dependency_folder_clicked
        )
        self.clear_current_dependencies_button = QPushButton(
            self.tr("Clear Current Runtime Environment"), tab,
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
        button_row2.addWidget(self.open_dependency_folder_button)
        button_row2.addWidget(self.clear_current_dependencies_button)
        button_row2.addWidget(self.clear_all_dependencies_button)

        # -- Install status and log ----------------------------------------
        self.dependency_install_status_label = QLabel(
            self.tr("Installable dependency status will appear here."),
            tab,
        )
        self.dependency_install_status_label.setWordWrap(True)
        self.dependency_install_progress = QProgressBar(tab)
        self.dependency_install_progress.setRange(0, 0)
        self.dependency_install_progress.setVisible(False)

        self.dependency_install_log = QTextBrowser(tab)
        self.dependency_install_log.setReadOnly(True)
        self.dependency_install_log.setPlaceholderText(
            self.tr("Dependency installation output will appear here.")
        )
        self.dependency_install_log.setMinimumHeight(220)
        self.dependency_install_log.setOpenExternalLinks(True)

        # -- Layout assembly -----------------------------------------------
        layout.addWidget(profile_group)
        layout.addWidget(self.dependency_summary_label)
        layout.addWidget(self.dependency_storage_label)
        layout.addWidget(self.dependency_storage_dedup_label)
        layout.addWidget(self.dependency_table)
        layout.addLayout(button_row)
        layout.addLayout(button_row2)
        layout.addWidget(self.dependency_install_status_label)
        layout.addWidget(self.dependency_install_progress)
        layout.addWidget(self.dependency_install_log)
        return tab

    def _current_torch_profile(self) -> TorchProfile:
        """Return the currently selected torch profile from the combo box."""
        idx = self.torch_profile_combo.currentIndex()
        profile = self.torch_profile_combo.itemData(idx)
        if profile in PROFILE_VALUES:
            return profile  # type: ignore[return-value]
        return DEFAULT_PROFILE

    def _on_torch_profile_changed(self, _index: int) -> None:
        """Handle torch profile combo box selection change.

        Persist the selection and immediately refresh the dependency status
        table + button enablement so the user sees the selected backend's
        install state without a manual "Refresh Status" click.
        """
        profile = self._current_torch_profile()
        save_plugin_settings({"torch_profile": profile})
        self.settings["torch_profile"] = profile
        is_custom = profile == "custom"
        self.custom_index_url_label.setVisible(is_custom)
        self.custom_index_url_edit.setVisible(is_custom)
        self.refresh_dependency_status()

    def _update_profile_status_label(
        self,
        selected_fully_installed: bool,
        selected_missing: list[str],
    ) -> None:
        """Refresh the status label under the backend combo.

        Parameters
        ----------
        selected_fully_installed : bool
            Whether the currently-selected profile has its dependencies
            installed.
        selected_missing : list[str]
            Missing module names for the selected profile (for the hint text).

        """
        from .dependency_path import _runtime_dependency_key

        from .runtime_profile import active_profile

        key = _runtime_dependency_key()
        active = active_profile(key)
        selected = self._current_torch_profile()
        active_text = profile_display_name(active)
        selected_text = profile_display_name(selected)
        if active == selected:
            self.profile_status_label.setText(
                self.tr("Active Runtime: {profile}").format(profile=active_text)
            )
        else:
            self.profile_status_label.setText(
                self.tr(
                    "Active Runtime: {active} | Selected Runtime: {selected}"
                ).format(active=active_text, selected=selected_text)
            )

    def _selected_profile_status(self) -> tuple[bool, list[str]]:
        """Return ``(fully_installed, missing)`` for the combo-selected profile.

        Uses an on-disk dist-info scan (not :mod:`importlib.metadata`) so it
        reports a non-active profile correctly.

        """
        from .dependency_path import _runtime_dependency_key

        key = _runtime_dependency_key()
        return profile_install_status(key, self._current_torch_profile())

    def _save_custom_index_url(self) -> None:
        """Persist the custom index URL from the input field."""
        url = self.custom_index_url_edit.text().strip()
        from .dependency_path import _runtime_dependency_key

        key = _runtime_dependency_key()
        set_custom_index_url(key, url)

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

        browse_button = QPushButton(self.tr("Choose Folder"), tab)
        browse_button.clicked.connect(self.browse_model_directory)
        open_button = QPushButton(self.tr("Show in Folder"), tab)
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
        cache_browse_button = QPushButton(self.tr("Choose Folder"), tab)
        cache_browse_button.clicked.connect(self.browse_cache_directory)
        cache_open_button = QPushButton(self.tr("Show in Folder"), tab)
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
        from .dependency_path import _runtime_dependency_key

        from .runtime_profile import active_profile

        key = _runtime_dependency_key()
        active = active_profile(key)
        selected = self._current_torch_profile()

        # Always use disk-scanning for the status table so that the result
        # is immune to importlib.metadata cache staleness after a profile
        # switch.  ``_rows_for_profile`` reads .dist-info directories from
        # the target folder, which is authoritative for *any* profile
        # (active or not).
        selected_fully_installed, selected_missing = self._selected_profile_status()
        rows = self._rows_for_profile(key, selected)

        installed_count = sum(1 for row in rows if row["installed"])
        missing_count = len(rows) - installed_count
        installable_missing_rows = [
            row for row in rows if not row["installed"] and row["installable"]
        ]
        # The packages "Install Missing" will target: the selected profile's
        # missing set when viewing a non-active profile, otherwise the active
        # profile's installable-missing set.
        self._missing_dependency_names = [
            row["package"] for row in installable_missing_rows
        ]
        self._update_profile_status_label(
            selected_fully_installed, selected_missing
        )
        if not selected_fully_installed and missing_count > 0:
            summary_text = self.tr(
                "Dependencies: {installed} installed, {missing} missing — "
                "click 'Install Missing' to install."
            ).format(installed=installed_count, missing=missing_count)
            self.dependency_summary_label.setWarning(summary_text)
        elif selected != active:
            self.dependency_summary_label.setPlain(
                self.tr(
                    "Selected runtime '{selected}': {installed} installed, "
                    "{missing} missing."
                ).format(
                    selected=profile_display_name(selected),
                    installed=installed_count,
                    missing=missing_count,
                )
            )
        else:
            self.dependency_summary_label.setPlain(
                self.tr(
                    "Dependencies: {installed} installed, {missing} missing."
                ).format(
                    installed=installed_count,
                    missing=missing_count,
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
        total_file_count = sum(row["file_count"] for row in all_stats)
        dedup_stats = get_plugin_managed_dependency_stats_dedup(
            all_dependency_paths,
        )
        storage_lines: list[str] = []
        storage_lines.append(
            self.tr(
                "Current runtime ({profile}): {size} · {files} file(s)"
            ).format(
                profile=profile_display_name(active),
                size=self._format_bytes(current_stats["size_bytes"]),
                files=current_stats["file_count"],
            )
        )
        storage_lines.append(
            self.tr(
                "All runtimes (logical total): {total_size} · "
                "{files} file(s) across {folder_count} folder(s)"
            ).format(
                total_size=self._format_bytes(total_size_bytes),
                files=total_file_count,
                folder_count=len(all_dependency_paths),
            )
        )
        dedup_bytes = dedup_stats["dedup_size_bytes"]
        shared_bytes = dedup_stats["shared_bytes"]
        unique_files = dedup_stats["unique_file_count"]
        dedup_file_count = dedup_stats["file_count"]
        symlink_count = dedup_stats["symlink_count"]
        if shared_bytes > 0:
            storage_lines.append(
                self.tr(
                    "Actual disk usage (deduplicated): {dedup_size} · "
                    "{unique_files} unique file(s)"
                ).format(
                    dedup_size=self._format_bytes(dedup_bytes),
                    unique_files=unique_files,
                )
            )
            self.dependency_storage_dedup_label.setInfo(
                self.tr(
                    "Saved {shared_size} via hardlinks/symlinks "
                    "({unique_files} unique out of {total_files} files, "
                    "{symlink_count} symlink(s))."
                ).format(
                    shared_size=self._format_bytes(shared_bytes),
                    unique_files=unique_files,
                    total_files=dedup_file_count,
                    symlink_count=symlink_count,
                )
            )
        elif dedup_file_count > 0 and platform.system() == "Darwin":
            self.dependency_storage_dedup_label.setInfo(
                self.tr(
                    "On macOS, uv shares files via APFS clonefiles, so "
                    "real usage may be smaller than the logical total."
                )
            )
        elif symlink_count > 0:
            storage_lines.append(
                self.tr(
                    "Actual disk usage (deduplicated): {dedup_size} · "
                    "{symlink_count} symlink(s) skipped"
                ).format(
                    dedup_size=self._format_bytes(dedup_bytes),
                    symlink_count=symlink_count,
                )
            )
            self.dependency_storage_dedup_label.clear()
        else:
            self.dependency_storage_dedup_label.clear()
        self.dependency_storage_label.setText("\n".join(storage_lines))
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

        is_installing = self._install_worker is not None
        self.install_dependencies_button.setEnabled(
            bool(self._missing_dependency_names) and not is_installing
        )
        # Switch Runtime is a pure sys.path swap: only allow it when the
        # selected runtime is fully installed AND differs from the active one.
        # Hint the reason via tooltip when disabled.
        can_switch = (
            not is_installing
            and selected != active
            and selected_fully_installed
        )
        self.switch_runtime_button.setEnabled(can_switch)
        if not can_switch and not is_installing:
            if selected == active:
                hint = self.tr("Already active.")
            elif not selected_fully_installed:
                hint = self.tr(
                    "Install missing dependencies for '{selected}' first."
                ).format(selected=profile_display_name(selected))
            else:
                hint = ""
            self.switch_runtime_button.setToolTip(hint)
        else:
            self.switch_runtime_button.setToolTip("")
        self.copy_install_commands_button.setEnabled(not is_installing)
        self.refresh_dependencies_button.setEnabled(not is_installing)
        self.open_dependency_folder_button.setEnabled(not is_installing)
        self.clear_current_dependencies_button.setEnabled(
            current_stats["exists"] and not is_installing
        )
        self.clear_all_dependencies_button.setEnabled(
            bool(all_dependency_paths) and not is_installing
        )

    def _rows_for_profile(
        self,
        key: str,
        profile: TorchProfile,
    ) -> list[DependencyStatusRow]:
        """Build dependency status rows for a given profile.

        Plugin-managed packages (torch, torchvision, ultralytics, geosam) are
        checked by scanning the profile's :file:`*.dist-info` directories on
        disk, so the result is authoritative regardless of whether *profile*
        is the active one. QGIS runtime-provided packages (rasterio,
        geopandas, shapely, pyproj) are checked via
        :func:`importlib.metadata.distribution` because they live in the QGIS
        Python environment, not in the profile target directory.

        Parameters
        ----------
        key : str
            Runtime dependency key.
        profile : TorchProfile
            Profile to inspect.

        Returns
        -------
        list[DependencyStatusRow]
            One row per plugin runtime dependency.

        """
        import importlib.metadata

        from .plugin_settings import (
            DEPENDENCY_DISTRIBUTIONS,
            RUNTIME_PROVIDED_DEPENDENCY_NAMES,
            dependency_source_text,
            dependency_state,
        )
        from .runtime_profile import profile_target_path

        target = profile_target_path(key, profile)
        installed_versions: dict[str, str] = {}
        if target.exists():
            for child in target.iterdir():
                if not (child.is_dir() and child.name.endswith(".dist-info")):
                    continue
                parts = child.name[:-len(".dist-info")].rsplit("-", 1)
                if len(parts) == 2:
                    name, version = parts
                    installed_versions[name] = version
        rows: list[DependencyStatusRow] = []
        for module_name, distribution_name in DEPENDENCY_DISTRIBUTIONS.items():
            if module_name in RUNTIME_PROVIDED_DEPENDENCY_NAMES:
                try:
                    distribution = importlib.metadata.distribution(
                        distribution_name
                    )
                except importlib.metadata.PackageNotFoundError:
                    version = ""
                else:
                    version = distribution.version
            else:
                version = installed_versions.get(distribution_name, "")
            installed = bool(version)
            state = dependency_state(module_name, installed)
            rows.append(
                {
                    "package": module_name,
                    "distribution": distribution_name,
                    "installed": installed,
                    "version": version,
                    "state": state,
                    "installable": state == "missing_installable",
                    "source": dependency_source_text(module_name, state),
                }
            )
        return rows

    def install_dependencies_clicked(self) -> None:
        if self._install_worker is not None:
            return
        if not self._missing_dependency_names:
            self.refresh_dependency_status()
            self.dependency_install_status_label.setText(
                self.tr("No installable dependencies are missing.")
            )
            return

        self._install_header_lines = []
        self._install_log_lines = []
        self.dependency_install_log.clear()
        missing_dependencies = ", ".join(self._missing_dependency_names)
        self.dependency_install_status_label.setText(
            self.tr("Installing missing dependencies: {dependencies}").format(
                dependencies=missing_dependencies
            )
        )
        self._set_install_buttons_enabled(False)
        self.close_button.setEnabled(False)
        self.dependency_install_progress.setVisible(True)
        self._append_dependency_install_log(
            self.tr("Starting dependency installation: {dependencies}").format(
                dependencies=missing_dependencies
            )
        )

        profile = self._current_torch_profile()
        custom_url = self.custom_index_url_edit.text().strip() or None

        self._append_dependency_install_log(
            self.tr("Target runtime: {profile}").format(
                profile=profile_display_name(profile)
            )
        )

        try:
            plan = make_plan(
                tuple(self._missing_dependency_names),
                profile,
                custom_index_url=custom_url,
            )
            commands = build_uv_commands(plan)
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

    def switch_runtime_clicked(self) -> None:
        """Handle the Switch Runtime button click.

        Switching is a pure ``sys.path`` swap: it persists the new active
        profile and releases loaded models. It does **not** install anything
        — all installation is managed by the "Install Missing" button. The
        button is only enabled when the selected runtime is fully installed,
        so reaching here means the target is ready; we keep a defensive
        check that hints the user to use "Install Missing" if the state has
        changed underneath us.
        """
        profile = self._current_torch_profile()
        answer = QMessageBox.question(
            self,
            self.tr("Switch Torch Runtime"),
            self.tr(
                "Switching the torch runtime will unload loaded models.\n\n"
                "Target runtime: {profile}\n\n"
                "Continue?"
            ).format(profile=profile_display_name(profile)),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        custom_url = self.custom_index_url_edit.text().strip() or None
        try:
            switch_plan = prepare_switch(
                profile,
                install_if_missing=False,
                custom_index_url=custom_url,
            )
        except ValueError as exc:
            # Profile not allowed on this platform.
            QMessageBox.warning(
                self,
                self.tr("Switch Torch Runtime"),
                str(exc),
            )
            return
        except RuntimeError as exc:
            # Not installed: hint to use Install Missing (the button should
            # already be disabled, but guard against races).
            QMessageBox.information(
                self,
                self.tr("Switch Torch Runtime"),
                self.tr(
                    "{error}\n\nUse 'Install Missing' to install dependencies "
                    "for the '{profile}' runtime first."
                ).format(error=str(exc), profile=profile_display_name(profile)),
            )
            return

        try:
            result = commit_switch(switch_plan)
        except Exception as exc:
            self._append_dependency_install_log(str(exc))
            QMessageBox.critical(
                self,
                self.tr("Switch Torch Runtime"),
                self.tr("Switch failed: {error}").format(error=str(exc)),
            )
            self.refresh_dependency_status()
            return
        self._append_dependency_install_log(
            self.tr("Switched to {profile}.").format(
                profile=profile_display_name(result.profile),
            )
        )
        self.refresh_dependency_status()

    def copy_install_commands_clicked(self) -> None:
        """Copy the full install command (pip install uv + uv install) to clipboard."""
        profile = self._current_torch_profile()
        custom_url = self.custom_index_url_edit.text().strip() or None
        packages = self._packages_for_copy()

        try:
            plan = make_plan_for_copy(packages, profile, custom_index_url=custom_url)
            commands = build_full_commands(plan)
            text = format_dependency_install_commands(commands)
            QApplication.clipboard().setText(text)
            self._append_dependency_install_log(
                self.tr("Copied install commands to clipboard.")
            )
        except Exception as exc:
            self._append_dependency_install_log(
                self.tr("Failed to build command: {error}").format(error=str(exc))
            )

    def _packages_for_copy(self) -> tuple[str, ...]:
        """Return the package list to use for copy-command buttons."""
        if self._missing_dependency_names:
            return tuple(self._missing_dependency_names)
        from .plugin_settings import DEPENDENCY_INSTALL_REQUIREMENTS

        return tuple(DEPENDENCY_INSTALL_REQUIREMENTS.keys())

    def _set_install_buttons_enabled(self, enabled: bool) -> None:
        """Enable or disable all action buttons during installation."""
        self.install_dependencies_button.setEnabled(enabled)
        self.switch_runtime_button.setEnabled(enabled)
        self.copy_install_commands_button.setEnabled(enabled)
        self.refresh_dependencies_button.setEnabled(enabled)
        self.open_dependency_folder_button.setEnabled(enabled)
        self.clear_current_dependencies_button.setEnabled(enabled)
        self.clear_all_dependencies_button.setEnabled(enabled)

    def _start_next_dependency_install_command(self) -> None:
        """Start the next pending dependency install command via PTY worker."""
        if self._dependency_install_command_index >= len(
            self._dependency_install_commands
        ):
            output = self._install_log_plain_text()
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
        worker = InstallProcessWorker(command, self)
        worker.screen_output.connect(self._on_screen_output)
        worker.install_finished.connect(self._on_install_worker_finished)
        worker.start_failed.connect(self._on_install_worker_start_failed)
        self._install_worker = worker
        worker.start()

    def _on_screen_output(self, lines: list[str]) -> None:
        """Replace the install log with a full screen snapshot.

        The worker sends the entire visible terminal screen (uv's live TUI)
        as a list of rows. We store it as-is and request a coalesced render.
        This is a **full replacement** — no incremental diffing — so there
        are never duplicates or partial fragments.

        Parameters
        ----------
        lines : list[str]
            Non-empty screen rows (ANSI escapes preserved for color rendering).

        """
        self._install_log_lines = list(lines)
        self._render_install_log()

    def _on_install_worker_finished(self, exit_code: int) -> None:
        """Handle completion of the install worker subprocess."""
        if self._install_worker is None:
            return
        self._install_worker = None
        if exit_code == 0:
            self._start_next_dependency_install_command()
            return
        output = self._install_log_plain_text()
        self._finish_dependency_install(False, output)

    def _on_install_worker_start_failed(self, error_message: str) -> None:
        """Handle subprocess start failure from the install worker."""
        if self._install_worker is None:
            return
        combined_output = self._install_log_plain_text()
        if combined_output:
            combined_output = f"{combined_output}\n{error_message}"
        else:
            combined_output = error_message
        self._finish_dependency_install(False, combined_output)

    def _install_log_plain_text(self) -> str:
        """Return the install log as plain, ANSI-stripped text.

        Used by completion handlers that need the captured output (e.g. to
        report failure). Reads from the in-memory line list — the source of
        truth — rather than the rendered HTML widget, so no SGR escapes or
        HTML tags leak into the reported text.

        Returns
        -------
        str
            Newline-joined log lines with ANSI escapes removed.

        """
        return "\n".join(
            _strip_ansi(line)
            for line in self._install_header_lines + self._install_log_lines
        ).strip()

    def _render_install_log(self) -> None:
        """Request a coalesced re-render of the install log.

        Multiple rapid calls (e.g. during a fast multi-package download) are
        batched into a single :meth:`_flush_install_log` via a 150 ms timer,
        preventing the GUI thread from being flooded with ``setHtml()``
        calls that would otherwise freeze the main event loop.
        """
        if self._render_pending:
            return
        self._render_pending = True
        self._render_timer.start()

    def _flush_install_log(self) -> None:
        """Rebuild the install log widget from header lines + screen snapshot.

        Called by :attr:`_render_timer` (coalesced). The install log is tiny
        (a handful of lines), so re-rendering the whole HTML is cheap.

        Header lines (plugin-generated status/command text) are shown first,
        followed by the live uv terminal snapshot. Each line is converted
        from terminal SGR escapes to styled HTML via :func:`_ansi_to_html`.
        """
        self._render_pending = False
        all_lines = self._install_header_lines + self._install_log_lines
        if all_lines:
            body = "<br>".join(
                _ansi_to_html(line) for line in all_lines
            )
            html_doc = (
                "<html><body><div style='white-space: pre-wrap; "
                "font-family: Menlo, Monaco, Consolas, \"Courier New\", "
                "monospace;'>"
                f"{body}</div></body></html>"
            )
        else:
            html_doc = ""
        log = self.dependency_install_log
        log.setHtml(html_doc)
        scrollbar = log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _append_dependency_install_log(self, message: str) -> None:
        """Append a plugin-generated header line above the live uv output.

        Header lines persist across screen snapshots (they are not part of
        uv's terminal and are never overwritten). Used for status messages,
        commands, and error text.

        """
        if not message:
            return
        self._install_header_lines.append(message)
        self._render_install_log()

    def _finish_dependency_install(self, ok: bool, output: str) -> None:
        """Restore dependency install UI state and report the result.

        When a profile switch is pending and the install succeeded, the
        switch is committed here (sys.path swap + active profile persist +
        model release).
        """
        # Flush any pending render immediately so the final log state is shown.
        if self._render_timer.isActive():
            self._render_timer.stop()
            self._flush_install_log()
        self.dependency_install_progress.setVisible(False)
        self._set_install_buttons_enabled(True)
        self.close_button.setEnabled(True)
        self.dependency_install_status_label.setText(
            self.tr("Dependency installation completed.")
            if ok
            else self.tr("Dependency installation failed.")
        )
        self._install_worker = None
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
            error_msg = str(exc)
            if sys.platform == "win32":
                error_msg += self.tr(
                    "\n\nOn Windows, files may be locked by running processes. "
                    "Please restart QGIS and try again."
                )
            MessageTool.MessageBoxOK(
                error_msg,
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
            error_msg = str(exc)
            if sys.platform == "win32":
                error_msg += self.tr(
                    "\n\nOn Windows, files may be locked by running processes. "
                    "Please restart QGIS and try again."
                )
            MessageTool.MessageBoxOK(
                error_msg,
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
