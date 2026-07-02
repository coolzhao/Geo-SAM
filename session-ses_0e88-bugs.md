# Geo-SAM 依赖安装系统 Bug 总结

## Bug 1: 退出 QGIS 时崩溃（QThread 析构中仍在运行）

**崩溃栈关键信息：**
```
QThread::~QThread() → QMessageLogger::fatal() → abort()
↑ sipQThread (InstallProcessWorker) 被析构
↑ QWidget::~QWidget() → QObjectPrivate::deleteChildren()
↑ gc_collect_main  ← QGIS unloadPlugin 触发垃圾回收
```

**根因：** `InstallProcessWorker(QThread)` 没有任何线程清理逻辑。当 dialog 关闭或 QGIS 退出时，还在跑子进程的 `QThread` 被 GC 回收 → Qt 触发致命的 `abort()`。

**修复：**
- `InstallProcessWorker` 增加 `_cancel_requested` 协作取消标志，`run()` 循环每 0.1s 检查
- `cancel()` 同时设标志 + `terminate()` 子进程，让 `wait()` 迅速返回
- `GeoSamSettingsDialog` 新增 `done(result)` 和 `closeEvent(event)` 覆盖
- 新增 `_stop_background_threads()`: `cancel()` + `wait(5000)` 安装线程；`wait(5000)` 下载线程

**文件：** `tools/settingsTool.py`

---

## Bug 2: 依赖安装进度不显示（卡在 "Using CPython 3.12.11"）

**现象：** 安装开始后只显示一行 "Using CPython 3.12.11"，然后长时间无输出。

**根因：** QGIS 从 GUI 启动时设了 `TERM=dumb`。即使 worker 用了 PTY，uv 检查 `TERM` 后认为终端不支持渲染，**禁用所有 spinner 和下载进度条**。

**验证（最小化变量测试）：**

| 环境 | bytes | 有 spinner |
|---|---|---|
| `CI=1, TERM=dumb` (QGIS 真实) | 274 | ❌ |
| 仅 `TERM=xterm-256color` | **2920** | ✅ |
| 仅删除 `CI` | 274 | ❌ |

**根因是 `TERM=dumb`，不是 `CI`。**

**修复：**
- `InstallProcessWorker` 新增 `_subprocess_env()` 静态方法
- 若 `TERM` 为空或 `dumb`，设为 `xterm-256color`
- 设 `CLICOLOR_FORCE=1`，移除 `NO_COLOR`
- `_run_with_pty()` 和 `_run_with_pipe()` 均传入此 env

**文件：** `tools/settingsTool.py`

---

## Bug 3: torch 解析失败（No solution found）

**错误信息：**
```
× No solution found when resolving dependencies:
╰─▶ Because torch==2.11.0 has no wheels with a matching platform tag
    (e.g., macosx_13_0_x86_64)
```

**根因：** `plugin_settings.py:96-99` 把 Python 3.10+ 的 torch 锁到了 `torch==2.11.0` / `torchvision==0.26.0`。PyTorch 从 2.3.0 起**停止发布 macOS x86_64 wheel**（只剩 arm64）。而 QGIS Python 的 ABI 是 x86_64（`_sysconfigdata` 编译标志 `-arch x86_64`），在 x86_64 平台上无法解析 torch 2.11.0。

**验证：**
- torch 2.2.2: ✅ 有 `macosx_10_13_x86_64` wheel
- torch 2.3.0+: ❌ 全无 x86_64 macOS wheel

**修复：** 移除 torch 版本锁（`TORCH_RESOLVER_CONSTRAINT_REQUIREMENTS_PY39` 和 `..._PY310_PLUS` 改为空元组），让 uv 自动选可用版本。

**文件：** `tools/plugin_settings.py`

---

## Bug 4: 安装进度条重复、不刷新

**现象：** torch 下载行每帧追加一次，导致 `torch ---- 16 KiB`、`torch ---- 32 KiB` 等大量重复。进度条一直堆积，不刷新。

**根因（多层）：**

1. **uv 的多行 TUI 重绘结构：**
   - uv 用 `\x1b[9A`（光标上移 9 行）+ `\x1b[2K`（清行）+ `\r` 重绘整个下载块
   - 原来的 `_drain_text_buffer` 只认 `\r` 和 `\n`，每次重绘的每一行都当新行 emit

2. **原始行包含 ANSI 颜色码：**
   - `_strip_ansi` 丢弃了 uv 的颜色信息（dim/bold/green/white/black）

3. **多包并行下载时：**
   - uv 同时显示 N 个包的进度行，每行都在变化
   - 原来的 `diff()` 只把最后一行当作瞬时进度，其余当作永久输出
   - 不断变化的字节数 → 不同身份 → 被当永久行 append

**最终方案（快照替换法）：**

抛弃逐行 diff 方法，改用**全屏快照替换**：

- `_ProgressTerminal` 维护完整的屏幕缓冲区（行 + finalized 状态）
- `snapshot()` 返回当前所有非空行（保留 ANSI 颜色码）
- Worker 限流（~6/sec）发出 `screen_output(list)` 信号
- Dialog 用 `_install_header_lines`（插件自身消息）+ `_install_log_lines`（快照）合并渲染
- QTimer（150ms）合并渲染调用，避免主线程冻结

**关键设计决策：**
- 不再追踪每行的 "永久/瞬时" 状态
- 不再使用 `_line_identity` 去重
- 屏幕就是真相 —— 每次快照都是完整的替换

**文件：** `tools/settingsTool.py`

---

## Bug 5: 安装进度无颜色

**现象：** 进度条是默认白色，uv 的 dim/green/bold 颜色全部丢失。

**根因：** `_strip_ansi()` 在 emit 前就丢弃了所有 SGR 颜色码。`QPlainTextEdit.setPlainText()` 也不支持颜色。

**修复：**
- `_ProgressTerminal.feed()` 只消费光标命令（`A`/`B`/`K`），**SGR 颜色码（`m`）保留在存储行中**
- 新增 `_ansi_to_html()` 转换器（无外部依赖），处理 uv 的 7 种 SGR 码：
  - `2` → `color:#888`（dim 灰色）
  - `1` → `font-weight:bold`（粗体）
  - `32` → `color:#008c4a`（绿色，复用插件 hex）
  - `37` → `color:#c0c0c0`（浅灰）
  - `30` → `color:#444`（深灰）
  - `0`/`39` → reset
- `QPlainTextEdit` → `QTextBrowser` + `setHtml()` 渲染
- 新增 `_install_log_plain_text()` 从 `_install_log_lines` + `_strip_ansi()` 读取纯文本

**文件：** `tools/settingsTool.py`

---

## Bug 6: 主进程阻塞/卡住

**现象：** 安装期间 QGIS 主界面完全冻结，无法操作。

**根因：** Worker 线程每秒发数百个 `line_output` 信号，每个信号在主线程触发 `setHtml()` 重建整个 HTML → 事件队列堆积 → 主线程冻结。

**修复：**
- `QTimer`（150ms 单次）合并渲染请求
- `_render_install_log()` 只标记 `_render_pending = True` 并启动定时器
- `_flush_install_log()` 在定时器触发时实际执行 `setHtml()`
- 安装完成时（`_finish_dependency_install`）立即 flush

**文件：** `tools/settingsTool.py`

---

## Bug 7: --python-platform 强制错误架构

**错误信息：**
```
OSError: dlopen(.../libtorch_global_deps.dylib):
mach-o file, but is an incompatible architecture
(have 'x86_64', need 'arm64e' or 'arm64')
```

**根因：** `--python-platform x86_64-apple-darwin` 从 `sysconfig.get_platform()` 推导。`_sysconfigdata` 编译标志 `-arch x86_64` 是 Python 的**编译标志**（QGIS universal2 构建），不代表运行时架构。QGIS 在 arm64 Mac 上**原生以 arm64 运行**。

**修复：**
- 移除 `--python-platform` 参数，只保留 `--python-version`
- uv 自动检测宿主机架构（arm64），装正确的 wheel
- 移除不再使用的 `_sysconfig_to_uv_platform`、`_extract_arch_from_sysconfig_platform` 函数

**文件：** `tools/dependency_installer.py`

---

## UI 改动汇总

| 改动 | 文件 |
|---|---|
| Switch Backend 按钮移入 Backend QGroupBox，与下拉菜单并排 | `tools/settingsTool.py` |
| 移除 "Copy uv Command"，"Copy Full Command" → "Copy Install Commands" | `tools/settingsTool.py` |
| "Clear Current Profile" → "Clear Current Environment" | `tools/settingsTool.py` |
| "Open Folder" → "Browse Packages"（依赖标签页） | `tools/settingsTool.py` |
| "Browse" → "Choose Folder"，"Open Folder" → "Show in Folder"（模型/缓存标签页） | `tools/settingsTool.py` |
| Refresh Status 按钮移到 Copy Install Commands 后面 | `tools/settingsTool.py` |
| Switch Backend 只安装 torch+torchvision（不装全部依赖） | `tools/runtime_profile.py` |
| 9 种语言翻译全部更新并编译为 .qm | `i18n/GeoSAM_*.ts` + `.qm` |

---

## Bug 4 续：进度重复的最终修复 — 全屏快照替换

### 之前的尝试（全部失败）

Bug 4 经历了 5+ 个 session 的多次修复尝试，每次解决一个子问题又暴露新的：

| 尝试 | 方案 | 新问题 |
|---|---|---|
| 1 | 终端模拟器 + `_drain_text_buffer` | 多行 TUI 重绘导致每帧重复 emit 所有行 |
| 2 | `_ProgressTerminal` 终端模拟器 + finalized 行区分 | 下载行未 finalized（无 `\n`），但值变化 → 不同身份 → 重复 |
| 3 | `_line_identity` 归一化下载数值 (`16 KiB` → `#`) | 多包并行时只有最后一行是"瞬时"，其余被当永久行 append |
| 4 | `_DOWNLOAD_LINE_RE` 检测下载行 + 按包身份替换 | 终端缓冲区碎片化导致 `torchvision ----` 这样的部分写入 |
| 5 | `_line_identity` + 按包身份匹配替换 | 终端模拟器的方法本质上就是错的 — 逐行 diff 无法正确处理多行 TUI |

### 最终方案：全屏快照替换（抛弃逐行 diff）

**核心思路：** 屏幕就是真相。不 diff，不追踪每行状态，不区分永久/瞬时。每次快照都是完整的替换。

**架构变更：**

```
之前（diff 方法，失败）：
Worker → terminal.diff() → line_output(str, bool) 信号 → Dialog 逐行 append/replace
                                                          ↓
                                                   _line_identity 去重
                                                   _replace_last_log_line
                                                   _committed_identities 追踪
                                                   _last_log_line_is_progress 标志

之后（快照方法）：
Worker → terminal.snapshot() → screen_output(list) 信号 → Dialog 全量替换
                                                          ↓
                                                   _install_header_lines (插件消息)
                                                   + _install_log_lines (屏幕快照)
                                                   → _flush_install_log() → setHtml()
```

**具体改动：**

1. **Worker 信号变更：**
   - 移除 `line_output = pyqtSignal(str, bool)`
   - 新增 `screen_output = pyqtSignal(list)`
   - `_emit_terminal_diff()` → `_emit_snapshot(terminal, force=False)`
   - 限流 ~6/sec（`force=True` 用于进程退出时）

2. **终端模拟器简化：**
   - `_ProgressTerminal.snapshot()` 返回当前所有非空行（保留 ANSI）
   - 移除 `diff()` 方法
   - 移除 `_committed_identities`、`_last_progress_identity`
   - `feed()` 仅消费光标命令（`A`/`B`/`K`），SGR 颜色码（`m`）保留在行中

3. **对话框重写：**
   - 新增 `_on_screen_output(lines)` 替代 `_on_install_line_output`
   - `_install_header_lines`：插件生成的消息（命令、状态、错误）
   - `_install_log_lines`：worker 的屏幕快照（每次完整替换）
   - `_flush_install_log()` 合并两者渲染 HTML
   - 移除 `_replace_last_log_line`、`_on_install_line_output`

4. **清理死代码：**
   - 移除 `_line_identity`、`_SPINNER_RE`、`_PROGRESS_VALUE_RE`、`_DOWNLOAD_LINE_RE`
   - 移除 `_last_log_line_is_progress`、`_committed_identities`、`_last_progress_identity`

**文件：** `tools/settingsTool.py`

---

## Bug 8: 下载进度快照中出现部分 ANSI 转义碎片（闪烁/乱码行）

**现象：** 多包快速下载时，安装日志中会短暂闪现 `\x1b[37`、`\x1b[2`、`\x1b[` 这样的乱码行（持续一两帧后消失）。

**根因：** `InstallProcessWorker._run_with_pty()` 以 4096 字节的固定块读取 PTY。当某次 read 恰好把一条 ANSI 转义序列切成两半（例如 `\x1b[37m` 被切成 `\x1b[3` + `7m`）时，`_ProgressTerminal.feed()` 的旧实现会把这个不完整的 `\x1b[3` 当作普通字符写进屏幕缓冲区。若此刻恰好触发快照（worker 每 0.15s emit 一次），这一行的快照就会是 `\x1b[3`，经 `_strip_ansi()` 后仍是 `\x1b[3`（因为 `_strip_ansi` 只匹配**完整**的 CSI），从而作为乱码行显示。

**验证（最小化变量测试）：**

用真实捕获的 numpy 下载字节流（含 `\x1b[1A`/`\x1b[2K`/`\x1b[1B` 多行重绘 + SGR 颜色码）以 9 种不同块大小（1/2/3/5/7/13/37/53/100/4096）回放进终端模拟器，每帧检查快照：

| 实现 | 泄漏帧数（共 9083 帧） |
|---|---|
| 旧实现 | **1841** 帧出现乱码（`'\x1b[37'`、`'⠼ \x1b[2'`、`'\x1b['` 等） |
| 新实现 | **0** |

**修复（`tools/settingsTool.py`）：**

- 新增正则 `_PARTIAL_CSI_TAIL_RE = re.compile(r"\x1b(?:\[(?:[0-9;?]*))?$")` —— 匹配位于块尾的未终结 CSI（单独的 `ESC`、`ESC [`、`ESC [ 1` … 直到缺少终结字母）。
- `_ProgressTerminal` 新增 `self._pending` 缓冲区。
- `feed()` 在解析前：先把上一块的 `_pending` 拼到当前块开头；再用 `_PARTIAL_CSI_TAIL_RE` 检测当前块尾部的不完整转义，将其存入 `_pending` 留待下次 `feed()`，本次只解析前半段。
- 更新类/方法文档字符串，说明部分转义会跨块缓存重组（旧的"each CSI is self-delimiting"注释是错的）。

**回归测试：** `tests/test_settings_tool_terminal.py`（8 个用例）

- `test_snapshot_mid_escape_split_does_not_leak_fragment` —— 在 `\x1b[3` 切分点取快照，断言不出现乱码行
- `test_partial_escape_is_rejoined_on_next_feed` —— 缓存的碎片在下一块正确重组并正常渲染
- `test_partial_escape_right_after_esc_byte` —— 块尾正好是单独 `ESC` 字节
- `test_partial_cursor_up_command_rejoined` —— 切分的 `\x1b[1` + `A` 仍能正确移动光标
- `test_full_chunk_unaffected_by_buffering` —— 完整块（无尾部碎片）`_pending` 为空
- `test_no_partial_escape_leak_across_many_chunk_sizes` —— 用 10 种块大小回放真实流，断言 0 泄漏
- `test_ansi_to_html_renders_colors_and_bold` / `test_ansi_to_html_handles_complete_sgr_sequence` —— 颜色/粗体渲染

**测试隔离：** 插件根目录的 `__init__.py` 会让 pytest 把整个插件当包导入（触发 `from .tools.dependency_path import ...`，只在 QGIS 内可用）。新增 `conftest.py`（`collect_ignore_glob`）让 pytest 不收集插件包，测试通过 `importlib.util` + qgis/sibling 模块桩加载 `settingsTool.py`，可在 QGIS 之外运行。

**文件：** `tools/settingsTool.py`、`tests/test_settings_tool_terminal.py`、`conftest.py`

---

## 关键教训

1. **不能信 `_sysconfigdata` 的编译标志来推断运行时架构** — universal2 二进制的 sysconfig 可能与实际运行架构不一致
2. **uv 的 `TERM=dumb` 检测比 `CI` 更敏感** — 仅改 `TERM` 即可恢复进度条
3. **多行 TUI 不适合逐行 diff** — 全屏快照替换是正确的架构（5 次失败尝试的教训）
4. **QPlainTextEdit + cursor 操作极易出 bug** — `setHtml()` + 列表驱动渲染更可靠
5. **Systematic debugging 的 3+ 失败法则** — 当同一个子问题修了 3+ 次仍有问题时，应质疑架构而非继续修补
6. **逐行信号 + setHtml() 会冻结主线程** — 必须限流 + QTimer 合并渲染
7. **终端模拟器的 SGR 码不应在 feed() 中丢弃** — 应保留在行中，渲染时转 HTML
8. **PTY 分块读取会切断转义序列** — 终端模拟器必须缓存并重组跨块的不完整 CSI，否则快照会出现 `\x1b[37` 这类乱码行
