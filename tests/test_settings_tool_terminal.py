"""Tests for the install-progress terminal emulator in ``settingsTool``.

These tests stub the ``qgis.PyQt`` import surface and the sibling ``tools.*``
modules that ``settingsTool`` imports at module top level, so the pure-Python
terminal emulator (``_ProgressTerminal``) and SGR→HTML converter
(``_ansi_to_html``) can be exercised outside a running QGIS.

Run from the plugin root:

    python3 -m pytest tests/test_settings_tool_terminal.py -v
    # or, without pytest:
    python3 tests/test_settings_tool_terminal.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _install_stub(name: str, attrs: dict[str, object] | None = None) -> types.ModuleType:
    """Register a stub module under ``name`` with the given attributes."""
    mod = types.ModuleType(name)
    for key, value in (attrs or {}).items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def _load_settings_tool():
    """Import ``tools/settingsTool.py`` with minimal QGIS + sibling stubs.

    Returns
    -------
    module
        The loaded ``settingsTool`` module.

    """
    plugin_root = Path(__file__).resolve().parent.parent
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))

    class _Dummy:
        """Placeholder for any Qt class referenced at import time."""

    # ---- qgis.PyQt stub ------------------------------------------------
    QtCore = _install_stub(
        "qgis.PyQt.QtCore",
        {
            "QThread": _Dummy,
            "QTimer": _Dummy,
            "pyqtSignal": lambda *a, **k: None,
            "Qt": types.SimpleNamespace(
                AlignmentFlag=types.SimpleNamespace(AlignRight=0),
            ),
            "QCloseEvent": _Dummy,
        },
    )
    QtGui = _install_stub(
        "qgis.PyQt.QtGui",
        {"QColor": _Dummy, "QCloseEvent": _Dummy},
    )
    qt_widgets_names = (
        "QAbstractItemView QApplication QCheckBox QComboBox QDialog QFileDialog "
        "QFormLayout QGridLayout QGroupBox QHBoxLayout QHeaderView QLabel "
        "QLineEdit QMessageBox QProgressBar QPushButton QSlider QTabBar "
        "QTabWidget QTableWidget QTableWidgetItem QTextBrowser QVBoxLayout QWidget"
    ).split()
    QtWidgets = _install_stub(
        "qgis.PyQt.QtWidgets",
        {name: _Dummy for name in qt_widgets_names},
    )
    pyqt = _install_stub("qgis.PyQt")
    pyqt.QtCore, pyqt.QtGui, pyqt.QtWidgets = QtCore, QtGui, QtWidgets
    _install_stub("qgis", attrs={"PyQt": pyqt})

    # ---- tools package + sibling modules (names only) -----------------
    # settingsTool does ``from .X import a, b, c``; give each sibling an object
    # for every imported name so the ``from ... import`` resolves.
    _install_stub("tools")
    sibling_attrs = {
        "tools.messageTool": ["MessageTool"],
        "tools.i18n": ["translate"],
        "tools.dependency_path": [
            "clear_all_plugin_managed_site_packages",
            "clear_current_plugin_managed_site_packages",
            "get_plugin_managed_dependency_stats",
            "get_plugin_managed_site_packages",
            "iter_all_plugin_managed_site_packages",
        ],
        "tools.dependency_installer": [
            "build_full_commands",
            "build_uv_commands",
            "make_plan",
            "make_plan_for_copy",
        ],
        "tools.model_manager": ["delete_model", "download_model", "get_model_status_rows"],
        "tools.plugin_settings": [
            "DEFAULT_CACHE_DIR",
            "DEFAULT_MODEL_DIR",
            "HELP_LINKS",
            "PERFORMANCE_MODE_VALUES",
            "PLUGIN_ROOT",
            "clear_cache",
            "dependency_status_text",
            "format_dependency_install_command",
            "format_dependency_install_commands",
            "get_cache_directory",
            "get_cache_size_bytes",
            "get_model_directory",
            "load_plugin_settings",
            "open_path",
            "open_url",
            "save_plugin_settings",
        ],
        "tools.geosam_runtime": ["get_loaded_model_ids", "release_runtime_models"],
        "tools.runtime_profile": [
            "DEFAULT_PROFILE",
            "PROFILE_VALUES",
            "TorchProfile",
            "commit_switch",
            "get_custom_index_url",
            "prepare_switch",
            "profile_display_name",
            "profile_install_status",
            "profiles_for_platform",
            "set_custom_index_url",
        ],
        "tools.torch_profile": [
            "detect_recommended_profile",
            "profile_unavailable_reason",
        ],
    }
    for modname, names in sibling_attrs.items():
        _install_stub(modname, {name: _Dummy() for name in names})

    module_path = plugin_root / "tools" / "settingsTool.py"
    spec = importlib.util.spec_from_file_location("tools.settingsTool", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tools.settingsTool"] = module
    spec.loader.exec_module(module)
    return module


st = _load_settings_tool()
_ProgressTerminal = st._ProgressTerminal
_ansi_to_html = st._ansi_to_html
_strip_ansi = st._strip_ansi


# ---------------------------------------------------------------------------
# Regression: partial ANSI escape split across PTY read boundaries
# ---------------------------------------------------------------------------


def test_snapshot_mid_escape_split_does_not_leak_fragment():
    """A snapshot taken between two chunks must not show a partial escape.

    Regression for the download-display flicker: the worker reads the PTY in
    fixed-size blocks, so an escape like ``\\x1b[37m`` can be split into
    ``\\x1b[3`` + ``7m``. The fragment must be buffered, not written into the
    screen buffer — otherwise it appears as a garbage snapshot row.
    """
    term = _ProgressTerminal()
    term.feed("Using CPython...\r\n")
    term.feed("\x1b[3")  # partial escape at end of a read chunk
    snapshot = term.snapshot()
    plain = [_strip_ansi(line) for line in snapshot]
    assert plain == ["Using CPython..."]
    # No row should contain a stray ESC after ANSI stripping.
    assert all("\x1b" not in line for line in plain)


def test_partial_escape_is_rejoined_on_next_feed():
    """A buffered partial escape rejoins the next chunk and renders normally."""
    term = _ProgressTerminal()
    term.feed("\x1b[3")
    term.feed("7m\u2819\x1b[0m Resolving dependencies...")
    plain = [_strip_ansi(line) for line in term.snapshot()]
    assert plain == ["\u2819 Resolving dependencies..."]


def test_partial_escape_right_after_esc_byte():
    """A chunk ending in a bare ``ESC`` byte must not leak it either."""
    term = _ProgressTerminal()
    term.feed("X\r\n\x1b")  # ESC alone at end
    assert term._pending == "\x1b"
    plain = [_strip_ansi(line) for line in term.snapshot()]
    assert plain == ["X"]
    term.feed("[37m\u2819\x1b[0m Hi")
    assert [_strip_ansi(line) for line in term.snapshot()] == ["X", "\u2819 Hi"]


def test_partial_cursor_up_command_rejoined():
    """A split cursor-up sequence (``\\x1b[1`` + ``A``) still moves the cursor."""
    term = _ProgressTerminal()
    term.feed("row1\nrow2\n")
    assert term._cur_row == 2
    term.feed("\x1b[1")  # split mid cursor-up
    assert term._cur_row == 2  # not yet applied
    term.feed("A")  # completes \x1b[1A (cursor up 1)
    assert term._cur_row == 1


def test_full_chunk_unaffected_by_buffering():
    """A complete chunk with no trailing partial escape leaves no pending data."""
    term = _ProgressTerminal()
    term.feed("\x1b[37m\u2819\x1b[0m \x1b[2mResolving dependencies...\x1b[0m")
    assert term._pending == ""
    plain = [_strip_ansi(line) for line in term.snapshot()]
    assert plain == ["\u2819 Resolving dependencies..."]


def test_no_partial_escape_leak_across_many_chunk_sizes():
    """Replay a realistic uv byte stream split many ways: no leak in any frame.

    This is the core regression guard for the multi-package download flicker:
    it would previously surface ``\\x1b[37`` / ``\\x1b[2`` / ``\\x1b[`` rows.
    """
    # A realistic uv progress byte stream with cursor-up redraws, SGR color
    # codes, and multi-row blocks (mirrors real uv pip install output).
    stream = (
        "\x1b[2mUsing CPython 3.9.18\x1b[0m\r\n"
        "\x1b[37m\u280b\x1b[0m \x1b[2mResolving dependencies...\x1b[0m\r"
        "\x1b[2K\x1b[37m\u2819\x1b[0m \x1b[2mResolving dependencies...\x1b[0m\r"
        "\x1b[2K\x1b[2mResolved \x1b[1m1 package\x1b[0m \x1b[2min 447ms\x1b[0m\r\n"
        "\x1b[32m\x1b[30m\x1b[2m------------------------------\x1b[0m\x1b[0m"
        " 16.00 KiB/13.32 MiB         \x1b[1A\r\x1b[2K\x1b[1B\r\x1b[2K"
        "\x1b[37m\u2838\x1b[0m \x1b[2mDownloading numpy...\x1b[0m\r\n"
        "\x1b[32m\x1b[30m\x1b[2m██████████████████████████████\x1b[0m\x1b[0m"
        " 13.32 MiB/13.32 MiB         \x1b[1A\r\x1b[2K\x1b[1B\r\x1b[2K"
        "\x1b[2mInstalled \x1b[1m1 package\x1b[0m\r\n"
        " \x1b[32m+\x1b[39m \x1b[1mnumpy\x1b[0m\x1b[2m==1.26.4\x1b[0m\r\n"
    )

    leaked = 0
    for chunk_size in (1, 2, 3, 5, 7, 13, 37, 53, 100, 4096):
        term = _ProgressTerminal()
        for i in range(0, len(stream), chunk_size):
            term.feed(stream[i : i + chunk_size])
            for line in term.snapshot():
                if "\x1b" in _strip_ansi(line):
                    leaked += 1
    assert leaked == 0, f"partial escape leaked in {leaked} frames"


# ---------------------------------------------------------------------------
# _ansi_to_html sanity
# ---------------------------------------------------------------------------


def test_ansi_to_html_renders_colors_and_bold():
    """SGR dim/green/bold map to the expected CSS styles."""
    out = _ansi_to_html("\x1b[32m+\x1b[39m \x1b[1msix\x1b[0m")
    assert "color:#008c4a" in out
    assert "font-weight:bold" in out


def test_ansi_to_html_handles_complete_sgr_sequence():
    """A complete SGR sequence is consumed (no ESC leaks) and styles applied."""
    out = _ansi_to_html("\x1b[2mdim text\x1b[0m")
    assert "\x1b" not in out
    assert "color:#888" in out
    assert "dim text" in out


def test_ansi_to_html_mid_line_color_change_reopens_span():
    """A color change mid-line must close+reopen the span (progress bar fix).

    Regression for the all-green progress bar: uv emits green for the filled
    portion then black+dim (gray) for the empty portion. The previous
    implementation kept the first span open, so the empty portion inherited
    green and the bar looked uniformly filled. The filled (green) and empty
    (gray) dashes must end up in *separate* spans with the right counts.
    """
    # Real uv bar fragment: 8 green dashes, then black+dim gray dashes.
    frag = "\x1b[32m--------\x1b[30m\x1b[2m----------------------\x1b[0m"
    out = _ansi_to_html(frag)
    import re as _re

    green = _re.search(r'color:#008c4a;">(-*)</span>', out)
    gray = _re.search(r'color:#888;">(-*)</span>', out)
    assert green is not None, f"green portion missing in {out!r}"
    assert gray is not None, f"gray portion missing in {out!r}"
    assert len(green.group(1)) == 8, f"expected 8 green dashes, got {out!r}"
    assert len(gray.group(1)) == 22, f"expected 22 gray dashes, got {out!r}"


def test_ansi_to_html_real_uv_progress_line():
    """A full uv progress line renders package name, bar, and size correctly."""
    line = (
        "\x1b[2mnumpy               \x1b[0m "
        "\x1b[32m--------\x1b[30m\x1b[2m----------------------\x1b[0m\x1b[0m "
        "1.42 MiB/5.06 MiB"
    )
    out = _ansi_to_html(line)
    assert "\x1b" not in out
    assert "numpy" in out
    assert "1.42 MiB/5.06 MiB" in out
    # Both colors must appear (filled green + empty gray).
    assert "color:#008c4a" in out
    assert "color:#888" in out


# ---------------------------------------------------------------------------
# snapshot() preserves trailing spaces (right-alignment of speed column)
# ---------------------------------------------------------------------------


def test_snapshot_preserves_trailing_spaces():
    """snapshot() keeps trailing spaces so uv's speed column stays aligned.

    Regression for the misaligned speed column: uv pads each multi-package
    progress line with trailing spaces so the speed/total values right-align.
    rstrip() destroyed that padding; we now only strip trailing CR/LF.
    """
    term = _ProgressTerminal()
    # Two lines with different total widths; uv pads the shorter one.
    term.feed("\x1b[2mnumpy\x1b[0m 1.42 MiB/5.06 MiB\r\n")
    term.feed("\x1b[2mscipy\x1b[0m 1.33 MiB/28.92 MiB          \r\n")
    snapshot = term.snapshot()
    assert len(snapshot) == 2
    # The scipy line must keep its trailing padding spaces.
    assert snapshot[1].endswith("          "), (
        f"trailing spaces stripped: {snapshot[1]!r}"
    )


def test_snapshot_strips_trailing_newline_only():
    """A line with content + trailing spaces but no newline keeps its spaces."""
    term = _ProgressTerminal()
    term.feed("pkg    3.51 MiB/4.56 MiB   ")  # trailing spaces, no newline
    snapshot = term.snapshot()
    assert snapshot == ["pkg    3.51 MiB/4.56 MiB   "]


# ---------------------------------------------------------------------------
# snapshot_stable() frame-completeness gate (anti-flicker)
# ---------------------------------------------------------------------------


def test_snapshot_stable_holds_last_frame_during_mid_redraw():
    """snapshot_stable() returns the cached frame while uv is mid-redraw.

    Regression for the flicker: uv redraws its multi-row progress block by
    rewinding the cursor (``\\x1b[nA``) and erasing+rewriting each row. A
    snapshot taken after the rewind but before the rewrite shows a
    half-drawn screen. The gate must return the previous complete frame
    until the cursor returns to the bottom.
    """
    term = _ProgressTerminal()
    frame1 = (
        "\x1b[37m\u280b\x1b[0m \x1b[2mPreparing packages...\x1b[0m (2/35)\r\n"
        "\x1b[2mnumpy\x1b[0m 1.0 MiB/5.0 MiB\r\n"
        "\x1b[2mscipy\x1b[0m 0.5 MiB/28.0 MiB\r\n"
    )
    term.feed(frame1)
    stable1 = term.snapshot_stable()
    assert len(stable1) == 3

    # Simulate uv's redraw split: cursor-up 3 + erase spinner row, but the
    # new spinner/package rows not yet rewritten.
    term.feed("\x1b[3A\r\x1b[2K")
    # Cursor is now at row 0 (above the high-water mark of 3).
    gated = term.snapshot_stable()
    # Gate must return the cached frame, not the half-erased screen.
    assert gated == stable1, (
        f"gate returned a mid-redraw frame: {gated!r}"
    )


def test_snapshot_stable_refreshes_when_redraw_completes():
    """snapshot_stable() refreshes the cache once the cursor reaches bottom."""
    term = _ProgressTerminal()
    frame1 = (
        "\x1b[37m\u280b\x1b[0m spin1\r\n"
        "\x1b[2mnumpy\x1b[0m old\r\n"
    )
    term.feed(frame1)
    term.snapshot_stable()

    # Full redraw: cursor-up 2, erase+rewrite both rows back to the bottom.
    redraw = (
        "\x1b[2A\r\x1b[2K\x1b[37m\u2819\x1b[0m spin2\r\n"
        "\r\x1b[2K\x1b[2mnumpy\x1b[0m new\r\n"
    )
    term.feed(redraw)
    refreshed = term.snapshot_stable()
    plain = [_strip_ansi(line) for line in refreshed]
    assert plain == ["\u2819 spin2", "numpy new"], f"stale frame: {plain!r}"


def test_snapshot_stable_emits_first_frame_even_mid_redraw():
    """The very first snapshot is emitted even if mid-redraw (no cache yet)."""
    term = _ProgressTerminal()
    # Cursor-up before any complete frame was ever cached.
    term.feed("only line\r\n\x1b[1A")
    snap = term.snapshot_stable()
    # No cached frame exists, so it must emit the current screen, not empty.
    assert snap == ["only line"]


if __name__ == "__main__":
    # Allow running without pytest.
    sys.exit(pytest.main([__file__, "-v"]))
