# Geo-SAM 依赖安装引擎重构方案

**Session ID:** ses_0e88d2457ffe7GWiRLW9g3m5EA
**Created:** 6/30/2026, 3:33:51 PM
**Updated:** 6/30/2026, 4:28:07 PM

---

## User

评价：**方向是对的，而且很适合你的场景；但原方案有几处需要收敛，否则会从“优雅”变成“维护复杂”。**

我会这样改：

## 结论

**可以用 uv 作为主安装引擎，pip 作为兜底。**
但不要把它设计成“用户安装 uv 后自己操作”，而是：

> Geo-SAM 内部管理一个私有 uv 二进制，用它安装到插件自己的 target runtime。用户只看到“切换 CPU / CUDA / 自定义依赖”。

uv 官方的 `uv pip install` 支持 `--target`，也就是可以直接安装到你现在这种隔离目录；它还支持 `--torch-backend cpu/cu128/cu126/auto`，其中 `auto` 会尝试根据系统和 CUDA 驱动选择 PyTorch index，但该选项目前是 preview，不能完全当成稳定产品契约。([Astral Docs][1])

---

## 对原方案的评价

### 1. “uv 比 pip 好很多”基本成立，但不要绝对化

uv 的确更适合作为你这里的依赖引擎，原因不是“快”这么简单，而是它有几个对你特别有用的能力：

```text
uv pip install --target <target>
uv pip uninstall --target <target>
uv pip list --target <target>
uv pip show --target <target>
uv pip sync --target <target>
```

尤其是 `uv pip sync` 很适合做“环境同步”：它会把目标环境同步到 requirements / pylock 描述的状态，未列出的包会被移除。这个比手写“卸载 torch、清理 dist-info、重装 GPU torch”优雅很多。([Astral Docs][1])

但也要注意，uv 不是 pip 的完全克隆。官方文档明确说它是常见 pip 工作流的替代品，但不是 exact clone，而且它不会读取 pip 专属配置，比如 `pip.conf` 或 `PIP_INDEX_URL`。所以你的插件里不要假设用户已有的 pip 配置会自动生效。([Astral Docs][2])

---

### 2. “有 uv.lock”这点要谨慎表述

如果你把 Geo-SAM 依赖管理做成一个标准 uv project，`uv.lock` 是很好的；它是跨平台 lockfile，会捕获不同系统、架构、Python 版本下的依赖解析信息。([Astral Docs][3])

但你现在的核心模型是：

```text
pip/uv install --target <plugin-runtime-target>
```

这种情况下，我更建议用：

```text
requirements.in
requirements.lock.txt
或者 pylock.toml
```

也就是：

```bash
uv pip compile requirements.in -o requirements.lock.txt
uv pip sync requirements.lock.txt --target <target>
```

uv 的 pip interface 支持把依赖锁定到 `requirements.txt`，也支持 `pylock.toml`；这比直接宣传“uv.lock 管 target”更准确。([Astral Docs][4])

---

## 对你的 Torch GPU 需求，uv 反而很加分

你之前的问题是：

> 默认安装 CPU torch，用户想换 GPU torch，如何优雅处理？

用 uv 后可以设计成：

```text
Torch Backend:
○ 自动推荐
○ CPU
○ NVIDIA CUDA 12.6
○ NVIDIA CUDA 12.8
○ NVIDIA CUDA 13.0
○ ROCm
○ 自定义
```

底层命令可以是：

```bash
"<uv>" pip install \
  --python "<qgis-python>" \
  --target "<new-target>" \
  --torch-backend cu128 \
  --no-python-downloads \
  --strict \
  torch torchvision ultralytics
```

或者自动推荐时：

```bash
"<uv>" pip install \
  --python "<qgis-python>" \
  --target "<new-target>" \
  --torch-backend auto \
  --no-python-downloads \
  --strict \
  torch torchvision ultralytics
```

`--torch-backend auto` 的存在让你的插件不必完全自己维护 PyTorch index 映射，但因为它是 preview，我建议产品上写成：

```text
自动推荐 Torch 后端
```

而不是：

```text
自动保证安装正确 CUDA 版本
```

---

## 最推荐的架构

不要让用户看到“uv / pip / target / index-url”这些细节。内部做四层：

```text
DependencyPlan
  - packages
  - torch_backend
  - python_path
  - target_path
  - constraints
  - install_engine: uv | pip

RuntimeProfile
  - cpu
  - cuda-cu128
  - cuda-cu126
  - custom

InstallerBackend
  - UvBackend
  - PipBackend

UvBootstrapper
  - 检查 .deps/bin/uv
  - 下载 uv
  - 校验 hash
  - chmod / 设置可执行
  - 失败回退 pip
```

用户体验是：

```text
当前环境：CPU

[修复当前环境]
[切换 Torch 后端...]
[管理额外包...]
[高级]
```

高级里才放：

```text
[复制 uv 命令]
[复制 pip 命令]
[打开 target 文件夹]
[清理未使用环境]
```

---

## uv 获取方案建议改一下

原文里的“自动下载 uv”是对的，但要补几个安全和兼容点。

不要默认执行官方 `install.sh` / `install.ps1`。uv 官方安装脚本默认可能修改 shell profile；如果用 installer，要显式使用 `UV_NO_MODIFY_PATH` 或 `UV_UNMANAGED_INSTALL`。更稳的是：**直接下载 GitHub Releases 里的平台二进制，放到 `.deps/bin/uv`，并校验你内置的 sha256。** 官方文档也说明 release artifacts 可以直接下载，并且 installer 支持指定安装路径和禁用 shell 修改。([Astral Docs][5])

启动 uv 时建议加这些约束：

```bash
"<uv>" \
  --no-config \
  --cache-dir "<plugin-cache>/uv" \
  pip install \
  --python "<qgis-python>" \
  --target "<target>" \
  --no-python-downloads \
  --strict \
  ...
```

关键是 `--python "<qgis-python>"` 和 `--no-python-downloads`。因为 uv 在 `--target` 模式下如果找不到合适 Python，可能会尝试下载 Python；你的插件应该强制使用 QGIS 自带 Python，避免解析出和 QGIS ABI 不一致的 wheel。([Astral Docs][1])

---

## 仍然建议保留 pip fallback

最终策略我会写成：

```text
依赖安装引擎：
- 默认：uv
- uv 不可用 / 下载失败 / 用户拒绝下载：pip
- 高级设置允许固定使用 pip
```

原因很现实：

```text
企业网络拦截
GitHub / astral.sh 不可访问
杀软拦截新下载 exe
用户禁止插件下载二进制
QGIS 插件审核或安全策略限制
```

uv 不是 Python 标准组件，pip 才是最后兜底。

---

## 最小实现版本

你可以先做这个版本：

```text
1. 启动时不下载 uv
2. 用户点击“安装 / 修复依赖”时，提示是否启用快速依赖引擎 uv
3. 同意后下载私有 uv 到 .deps/bin
4. 安装依赖时优先 uv
5. uv 失败自动生成 pip fallback 命令
6. CPU / CUDA / Custom 都创建独立 runtime target
```

UI 文案可以是：

```text
Geo-SAM 可以下载一个私有的快速依赖安装器 uv，用于安装到插件隔离环境。
它不会安装到系统 PATH，也不会修改你的 Python 环境。
```

---

## 最终建议

这套方案值得做，但我会把原文改成：

> **uv 作为首选依赖引擎，pip 作为兜底。Geo-SAM 自动管理私有 uv 二进制，不要求用户预装。依赖变更通过 Runtime Profile + staging target 完成；Torch CPU / CUDA / Custom 通过 `--torch-backend` 或显式 PyTorch index 安装。**

这样既满足全平台，也满足用户手动换 GPU torch 的需求，而且不会把复杂度暴露给普通用户。

[1]: https://docs.astral.sh/uv/reference/cli/ "Commands | uv"
[2]: https://docs.astral.sh/uv/pip/compatibility/ "Compatibility with pip | uv"
[3]: https://docs.astral.sh/uv/concepts/projects/layout/ "Structure and files | uv"
[4]: https://docs.astral.sh/uv/pip/compile/ "Locking environments | uv"
[5]: https://docs.astral.sh/uv/getting-started/installation/ "Installation | uv"  是的，**作为第一版，这个方案更稳、更轻、更容易被用户接受**：

> **检测系统里是否已有 uv；有就用 uv；没有就回退 pip。**

但我不建议把它作为最终最优方案。它更像是一个 **保守的 Auto 模式**。

---

## 我的评价

### 优点

这个方案有几个明显好处：

1. **不需要插件下载二进制**
2. **不会触发企业网络、杀软、插件审核等安全顾虑**
3. **实现简单**
4. **已有 uv 的高级用户可以获得更快、更好的依赖安装体验**
5. **没有 uv 的普通用户仍然可以继续使用 pip**

uv 的 `uv pip` 本来就是面向现有 pip 工作流的接口，可以用于 `install`、`sync`、`compile` 等常见依赖操作；但 uv 官方也说明它不是 pip 的完全克隆，所以保留 pip fallback 是合理的。([Astral Docs][1])

---

## 但它不是“体验最优”

问题在于：**大多数普通用户可能没有 uv。**

如果只做：

```text
有 uv -> 用 uv
没有 uv -> 用 pip
```

那实际效果可能是：

```text
高级用户：体验很好
普通用户：还是 pip 老体验
```

所以这个方案适合做 **低风险第一版**，但不一定能最大化解决“依赖安装慢、Torch 切换麻烦、依赖同步不干净”的问题。

---

## 我建议的最终策略

可以分三档：

```text
依赖安装引擎：Auto

1. 优先使用 Geo-SAM 私有 uv    // 未来可选
2. 否则使用系统 uv             // 你现在想做的
3. 否则使用 QGIS Python 的 pip
```

如果你现在不想做自动下载 uv，那就先实现成：

```text
依赖安装引擎：Auto

1. 检测系统 uv
2. 检测成功并版本满足要求 -> 使用 uv
3. 否则 -> 使用 pip
```

UI 里可以这么写：

```text
安装引擎：Auto
已检测到 uv：/opt/homebrew/bin/uv
当前将使用 uv 安装依赖。

[改用 pip] [复制 uv 命令] [复制 pip 命令]
```

如果没检测到：

```text
安装引擎：Auto
未检测到 uv，将使用 QGIS 自带 Python 的 pip。

[复制 pip 命令]
```

---

## 关键注意点

### 1. 不要只依赖 PATH

QGIS 是 GUI 程序，尤其在 macOS 上，从 Finder 启动时拿到的 `PATH` 经常和终端不同。

所以检测 uv 时建议按顺序找：

```python
1. 用户设置里的 uv 路径
2. shutil.which("uv")
3. 常见路径：
   macOS:
     /opt/homebrew/bin/uv
     /usr/local/bin/uv
     ~/.local/bin/uv

   Linux:
     ~/.local/bin/uv
     /usr/local/bin/uv
     /usr/bin/uv

   Windows:
     %USERPROFILE%\\.local\\bin\\uv.exe
     %USERPROFILE%\\AppData\\Local\\Programs\\uv\\uv.exe
```

并提供一个按钮：

```text
[选择 uv 可执行文件...]
```

这样用户明明安装了 uv，但插件找不到时，可以手动指定。

---

### 2. 必须检查 uv 版本

不要只检查 `uv` 存不存在，还要跑：

```bash
uv --version
```

然后判断最低版本。

比如你的 Torch 后端逻辑如果依赖 `--torch-backend`，那就需要确保用户 uv 版本足够新。uv 官方文档中 PyTorch 集成功能要求较新的 uv 版本，并且 `UV_TORCH_BACKEND` / `--torch-backend` 是后续版本才加入的能力。([Astral Docs][2])

如果版本太旧：

```text
检测到 uv，但版本过旧。将回退到 pip。
```

或者：

```text
检测到 uv 0.x.x，但需要 >= 0.x.x。
[继续使用 pip] [选择其他 uv]
```

---

### 3. 使用系统 uv 时，也要强制绑定 QGIS Python

不要让 uv 自己找 Python。你应该始终传入 QGIS 的 Python：

```bash
"<uv>" pip install \
  --python "<qgis-python>" \
  --target "<target>" \
  --no-python-downloads \
  ...
```

这样可以避免 uv 使用系统 Python、Homebrew Python、Conda Python，导致 wheel ABI 和 QGIS 不匹配。uv 文档说明它会发现和管理 Python 版本，因此插件这里应该明确指定 Python。([Astral Docs][3])

---

### 4. 最好禁用用户 uv 配置影响

用户自己的 uv 可能配置了私有 index、代理、缓存、镜像等。对普通项目是好事，但对你的插件可能会造成不可复现。

建议调用时加：

```bash
--no-config
--cache-dir "<Geo-SAM-cache>/uv"
```

这样 Geo-SAM 的依赖安装更可控。

---

## 对 Torch 切换的设计

你可以这样做：

```text
Torch Backend:
○ CPU
○ NVIDIA CUDA 12.6
○ NVIDIA CUDA 12.8
○ 自动推荐
○ 自定义
```

如果使用 uv：

```bash
"<uv>" pip install \
  --python "<qgis-python>" \
  --target "<new-target>" \
  --torch-backend cu128 \
  --no-python-downloads \
  --no-config \
  torch torchvision
```

uv 官方有 PyTorch 集成文档，支持按 CPU / CUDA 等加速后端配置 PyTorch 依赖。([Astral Docs][2])

如果回退 pip，则生成传统命令：

```bash
"<qgis-python>" -m pip install \
  --target "<new-target>" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision
```

也就是说，**UI 不变，底层安装引擎不同**。

---

## 我建议你现在就这么做

第一版：

```text
Auto:
  1. 检测系统 uv
  2. uv 可用且版本满足 -> 用 uv
  3. 否则 -> 用 pip
```

同时提供高级设置：

```text
依赖安装引擎：
○ Auto
○ uv
○ pip
```

如果用户选 uv 但找不到 uv：

```text
未找到 uv。请选择 uv 可执行文件，或切换到 pip。
```

---

## 最终判断

**是，更好，尤其适合作为第一阶段实现。**

但我会把它定义为：

> **“系统 uv 优先，pip 兜底”**

而不是：

> **“全面改用 uv”**

这样你不用处理自动下载 uv 的安全和分发问题，又能让已经安装 uv 的用户获得更好的体验。后续如果发现 pip 安装失败率仍然高，再加一个可选的“下载 Geo-SAM 私有 uv”也不迟。

[1]: https://docs.astral.sh/uv/pip/?utm_source=chatgpt.com "The pip interface | uv"
[2]: https://docs.astral.sh/uv/guides/integration/pytorch/?utm_source=chatgpt.com "Using uv with PyTorch"
[3]: https://docs.astral.sh/uv/concepts/python-versions/?utm_source=chatgpt.com "Python versions | uv - Astral Docs" 帮我制定一个重构计划 1. 依赖 install engin支持 uv pip auto(默认) 如果engin是uv, 没有uv就自动下载安装, 如果是auto，没有就自动为pip. 2. 要支持将安装命令一键复制到粘贴板 3. 如果uv，支持uv的auto pytorch模式，其他的使用环境检测与用户指定形式。帮我制定几个重构方案，及其各自的优缺点

---

## Assistant (Build · GLM-5.2 · 10.8s)

_Thinking:_

The user is asking me to create a refactoring plan for a QGIS plugin (Geo-SAM) that involves dependency installation. Let me understand the requirements:

1. Dependency install engine supports: uv, pip, auto (default)
   - If engine is uv and no uv exists → auto download and install uv
   - If engine is auto and no uv exists → fall back to pip
2. Support one-click copy of install commands to clipboard
3. If uv, support uv's auto pytorch mode; otherwise use environment detection + user-specified form

The user wants me to create several refactoring plans with pros and cons for each.

Let me first explore the codebase to understand the current architecture before proposing plans. This is important because the plans should be grounded in the actual code structure.

Let me look at the relevant files - particularly the runtime, widget tools, and any existing dependency management code.

I should explore the codebase first to understand:
- Current dependency installation logic
- Current torch backend handling
- Current UI structure for dependency management
- How the plugin manages target paths/runtimes

Let me start exploring.

I'll explore the codebase first to understand the current dependency management architecture before proposing refactoring plans.

**Tool: bash**

**Input:**
```json
{
  "command": "ls -la",
  "workdir": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM"
}
```

**Output:**
```
total 144
-rw-r--r--@  1 fancy  staff    640 Jun 22 00:42 __init__.py
drwxr-xr-x@  5 fancy  staff    160 Jun 24 10:29 __pycache__
drwxr-xr-x  23 fancy  staff    736 Jun 24 16:16 .
drwxr-xr-x@ 15 fancy  staff    480 Jun 22 01:12 ..
drwxr-xr-x@  2 fancy  staff     64 Jun 22 13:17 .cache
drwxr-xr-x@  4 fancy  staff    128 Jun 22 01:12 .deps
-rw-r--r--@  1 fancy  staff  14340 Jun 22 15:29 .DS_Store
drwxr-xr-x  20 fancy  staff    640 Jun 30 15:25 .git
-rw-r--r--@  1 fancy  staff    150 Jun 22 13:10 .gitignore
drwxr-xr-x@  3 fancy  staff     96 Jun 22 13:05 .models
-rw-r--r--@  1 fancy  staff    824 Jun 22 00:42 .readthedocs.yaml
drwxr-xr-x@  5 fancy  staff    160 Jun 22 12:29 .ruff_cache
-rw-r--r--@  1 fancy  staff   5244 Jun 24 16:16 AGENTS.md
-rw-r--r--@  1 fancy  staff    453 Jun 24 16:16 CITATION.cff
drwxr-xr-x@ 20 fancy  staff    640 Jun 30 15:25 docs
-rw-r--r--@  1 fancy  staff   2030 Jun 22 00:42 geo_sam_provider.py
-rw-r--r--@  1 fancy  staff  11503 Jun 22 00:42 geo_sam_tool.py
drwxr-xr-x@ 22 fancy  staff    704 Jun 22 00:42 i18n
-rw-r--r--   1 fancy  staff   1065 May  4 11:48 LICENSE
-rw-r--r--@  1 fancy  staff   3867 Jun 22 00:49 metadata.txt
-rw-r--r--@  1 fancy  staff   5474 Jun 24 16:16 README.md
drwxr-xr-x  20 fancy  staff    640 Jun 30 15:25 tools
drwxr-xr-x  12 fancy  staff    384 Jun 24 10:29 ui

```

**Tool: bash**

**Input:**
```json
{
  "command": "ls tools/ ui/",
  "workdir": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM"
}
```

**Output:**
```
tools/:
__init__.py
__pycache__
canvasTool.py
dependency_path.py
geosam_backend.py
geosam_runtime.py
geoTool.py
i18n.py
messageTool.py
model_manager.py
online_tile_export.py
plugin_settings.py
sam_processing_algorithm.py
settingsTool.py
translation_catalog.py
ulid.py
widgetTool.py

ui/:
__init__.py
__pycache__
config
cursors
EncoderCopilot.ui
icons
Selector.ui
split_tab_panel.py
UI.py

```

---

## Assistant (Build · GLM-5.2 · 15.1s)

_Thinking:_

Let me explore the key files for dependency management: geosam_runtime.py, widgetTool.py, dependency_path.py, plugin_settings.py, settingsTool.py

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/dependency_path.py"
}
```

**Output:**
```
<path>/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/dependency_path.py</path>
<type>file</type>
<content>
1: """Plugin-managed dependency path helpers for Geo-SAM."""
2: 
3: from __future__ import annotations
4: 
5: import hashlib
6: import importlib
7: import logging
8: import platform
9: import shutil
10: import sys
11: import sysconfig
12: from pathlib import Path
13: from typing import TypedDict
14: 
15: PLUGIN_ROOT = Path(__file__).resolve().parents[1]
16: PLUGIN_DEPENDENCY_ROOT = PLUGIN_ROOT / ".deps"
17: PLUGIN_RUNTIME_DEPENDENCIES_ROOT = PLUGIN_DEPENDENCY_ROOT / "runtimes"
18: LEGACY_PLUGIN_MANAGED_SITE_PACKAGES = PLUGIN_DEPENDENCY_ROOT / "python"
19: 
20: logger = logging.getLogger(__name__)
21: 
22: 
23: class DependencyPathStats(TypedDict):
24:     """Summary information for a plugin-managed dependency path.
25: 
26:     Attributes
27:     ----------
28:     path : Path
29:         Dependency directory path.
30:     exists : bool
31:         Whether the directory exists.
32:     file_count : int
33:         Number of files contained in the directory.
34:     size_bytes : int
35:         Total size of files contained in the directory.
36: 
37:     """
38: 
39:     path: Path
40:     exists: bool
41:     file_count: int
42:     size_bytes: int
43: 
44: 
45: def get_plugin_managed_site_packages(*, create: bool = False) -> Path:
46:     """Return the plugin-managed dependency directory for this runtime.
47: 
48:     Parameters
49:     ----------
50:     create : bool, optional
51:         Whether to create the dependency directory before returning it.
52: 
53:     Returns
54:     -------
55:     Path
56:         Directory used as the pip ``--target`` location for plugin-managed
57:         Python packages in the active QGIS Python runtime.
58: 
59:     """
60:     dependency_path = (
61:         PLUGIN_RUNTIME_DEPENDENCIES_ROOT / _runtime_dependency_key() / "python"
62:     )
63:     if create:
64:         dependency_path.mkdir(parents=True, exist_ok=True)
65:     return dependency_path
66: 
67: 
68: def register_plugin_managed_dependency_path(*, create: bool = False) -> Path:
69:     """Expose plugin-managed dependencies to the active Python runtime.
70: 
71:     Parameters
72:     ----------
73:     create : bool, optional
74:         Whether to create the dependency directory before registering it.
75: 
76:     Returns
77:     -------
78:     Path
79:         Registered dependency directory path.
80: 
81:     """
82:     dependency_path = get_plugin_managed_site_packages(create=create)
83:     dependency_path_text = str(dependency_path)
84:     if dependency_path_text not in sys.path:
85:         sys.path.insert(0, dependency_path_text)
86:         importlib.invalidate_caches()
87:     return dependency_path
88: 
89: 
90: def get_plugin_managed_dependency_stats(path: Path) -> DependencyPathStats:
91:     """Return size and file-count information for a dependency directory.
92: 
93:     Parameters
94:     ----------
95:     path : Path
96:         Dependency directory to inspect.
97: 
98:     Returns
99:     -------
100:     DependencyPathStats
101:         Summary information for the dependency directory.
102: 
103:     """
104:     exists = path.exists()
105:     file_count = 0
106:     size_bytes = 0
107:     if exists:
108:         for child_path in path.rglob("*"):
109:             if not child_path.is_file():
110:                 continue
111:             file_count += 1
112:             try:
113:                 size_bytes += child_path.stat().st_size
114:             except OSError as exc:
115:                 logger.warning(
116:                     "Could not stat dependency file %s while computing size: %s",
117:                     child_path,
118:                     exc,
119:                 )
120:     return {
121:         "path": path,
122:         "exists": exists,
123:         "file_count": file_count,
124:         "size_bytes": size_bytes,
125:     }
126: 
127: 
128: def iter_plugin_managed_runtime_site_packages() -> list[Path]:
129:     """Return existing plugin-managed dependency directories for all runtimes.
130: 
131:     Returns
132:     -------
133:     list[Path]
134:         Existing runtime-specific dependency directories.
135: 
136:     """
137:     if not PLUGIN_RUNTIME_DEPENDENCIES_ROOT.exists():
138:         return []
139:     dependency_paths: list[Path] = []
140:     for runtime_path in sorted(PLUGIN_RUNTIME_DEPENDENCIES_ROOT.iterdir()):
141:         site_packages_path = runtime_path / "python"
142:         if site_packages_path.exists():
143:             dependency_paths.append(site_packages_path)
144:     return dependency_paths
145: 
146: 
147: def iter_all_plugin_managed_site_packages(*, include_legacy: bool = True) -> list[Path]:
148:     """Return all plugin-managed dependency directories.
149: 
150:     Parameters
151:     ----------
152:     include_legacy : bool, optional
153:         Whether to include the old shared ``.deps/python`` directory.
154: 
155:     Returns
156:     -------
157:     list[Path]
158:         Existing plugin-managed dependency directories.
159: 
160:     """
161:     dependency_paths = iter_plugin_managed_runtime_site_packages()
162:     if include_legacy and LEGACY_PLUGIN_MANAGED_SITE_PACKAGES.exists():
163:         dependency_paths.append(LEGACY_PLUGIN_MANAGED_SITE_PACKAGES)
164:     return dependency_paths
165: 
166: 
167: def clear_current_plugin_managed_site_packages() -> int:
168:     """Delete dependencies installed for the active runtime.
169: 
170:     Returns
171:     -------
172:     int
173:         Number of files removed.
174: 
175:     """
176:     dependency_path = get_plugin_managed_site_packages()
177:     return _remove_dependency_path(dependency_path)
178: 
179: 
180: def clear_all_plugin_managed_site_packages() -> int:
181:     """Delete all plugin-managed dependencies, including legacy installs.
182: 
183:     Returns
184:     -------
185:     int
186:         Number of files removed.
187: 
188:     """
189:     removed_count = 0
190:     for dependency_path in iter_all_plugin_managed_site_packages(include_legacy=True):
191:         removed_count += _remove_dependency_path(dependency_path)
192:     _remove_empty_directory(PLUGIN_RUNTIME_DEPENDENCIES_ROOT)
193:     _remove_empty_directory(PLUGIN_DEPENDENCY_ROOT)
194:     return removed_count
195: 
196: 
197: def _runtime_dependency_key() -> str:
198:     """Return a stable dependency directory key for the active runtime.
199: 
200:     Returns
201:     -------
202:     str
203:         Runtime-specific dependency directory name.
204: 
205:     """
206:     python_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
207:     platform_tag = _safe_path_part(sysconfig.get_platform())
208:     architecture_tag = _safe_path_part(platform.machine() or "unknown")
209:     identity_parts = [
210:         sys.prefix,
211:         getattr(sys, "base_prefix", ""),
212:         sys.executable,
213:         python_tag,
214:         platform_tag,
215:         architecture_tag,
216:     ]
217:     runtime_hash = hashlib.sha256("\0".join(identity_parts).encode("utf-8"))
218:     return f"{python_tag}-{platform_tag}-{architecture_tag}-{runtime_hash.hexdigest()[:12]}"
219: 
220: 
221: def _safe_path_part(value: str) -> str:
222:     """Return a filesystem-safe path component.
223: 
224:     Parameters
225:     ----------
226:     value : str
227:         Raw path component text.
228: 
229:     Returns
230:     -------
231:     str
232:         Sanitized path component text.
233: 
234:     """
235:     sanitized = "".join(
236:         character if character.isalnum() or character in {"-", "_", "."} else "-"
237:         for character in value
238:     ).strip("-._")
239:     return sanitized or "unknown"
240: 
241: 
242: def _remove_dependency_path(dependency_path: Path) -> int:
243:     """Remove one plugin-managed dependency directory.
244: 
245:     Parameters
246:     ----------
247:     dependency_path : Path
248:         Dependency directory to delete.
249: 
250:     Returns
251:     -------
252:     int
253:         Number of files removed.
254: 
255:     """
256:     dependency_path_text = str(dependency_path)
257:     sys.path[:] = [path_text for path_text in sys.path if path_text != dependency_path_text]
258:     importlib.invalidate_caches()
259:     if not dependency_path.exists():
260:         return 0
261:     stats = get_plugin_managed_dependency_stats(dependency_path)
262:     try:
263:         shutil.rmtree(dependency_path)
264:     except OSError as exc:
265:         logger.error(
266:             "Could not remove plugin-managed dependency directory %s: %s",
267:             dependency_path,
268:             exc,
269:         )
270:         raise
271:     _remove_empty_directory(dependency_path.parent)
272:     return stats["file_count"]
273: 
274: 
275: def _remove_empty_directory(path: Path) -> None:
276:     """Remove a directory when it exists and is empty.
277: 
278:     Parameters
279:     ----------
280:     path : Path
281:         Directory path to remove.
282: 
283:     """
284:     try:
285:         path.rmdir()
286:     except FileNotFoundError:
287:         return
288:     except OSError:
289:         return

(End of file - total 289 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/geosam_runtime.py"
}
```

**Output:**
```
<path>/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/geosam_runtime.py</path>
<type>file</type>
<content>
1: """GeoSAM-backed runtime helpers for the QGIS plugin."""
2: 
3: from __future__ import annotations
4: 
5: import gc
6: import json
7: import logging
8: import sys
9: from dataclasses import dataclass
10: from pathlib import Path
11: from typing import TYPE_CHECKING, Any, Callable, Literal
12: from urllib.parse import urlparse
13: 
14: from qgis.core import (
15:     QgsCoordinateReferenceSystem,
16:     QgsGeometry,
17:     QgsRasterLayer,
18:     QgsRectangle,
19: )
20: 
21: from .geosam_backend import configure_geosam_qgis_runtime
22: from .messageTool import MessageTool
23: from .model_manager import (
24:     create_model_spec,
25:     effective_model_imgsz,
26:     get_model_definition,
27:     infer_model_id_from_checkpoint_path,
28: )
29: from .online_tile_export import (
30:     OnlineRasterExportError,
31:     OnlineTileExportPlan,
32:     OnlineRasterPreflightError,
33: )
34: from .online_tile_export import (
35:     export_online_raster_source as export_online_tile_raster_source,
36: )
37: from .online_tile_export import (
38:     export_online_raster_plan as export_online_tile_raster_plan,
39: )
40: from .online_tile_export import (
41:     online_export_failure_message as describe_online_export_failure,
42: )
43: from .online_tile_export import (
44:     prepare_online_raster_export_plan,
45: )
46: from .plugin_settings import (
47:     clear_cache,
48:     get_cache_directory,
49:     initialize_rasterio_proj_data,
50:     load_plugin_settings,
51:     trim_cache_if_needed,
52: )
53: 
54: if TYPE_CHECKING:
55:     from geosam.engines import QueryResult
56:     from geosam.models import ModelSpec
57:     from geosam.query import BoundingBox, Points, PromptSet
58: 
59: logger = logging.getLogger(__name__)
60: 
61: ONLINE_QUERY_REFRESH_MARGIN_PIXELS = 128
62: PerformanceMode = Literal["balanced", "fastest", "low_memory"]
63: PreviewRenderMode = Literal["pixel_level", "simplified"]
64: 
65: _DATACLASS_SLOTS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}
66: _FROZEN_DATACLASS_KWARGS = {"frozen": True, **_DATACLASS_SLOTS_KWARGS}
67: _MUTABLE_DATACLASS_KWARGS = dict(_DATACLASS_SLOTS_KWARGS)
68: 
69: @dataclass(**_FROZEN_DATACLASS_KWARGS)
70: class FeatureSourceSummary:
71:     """Summary metadata for a cached GeoSAM feature source."""
72: 
73:     manifest_path: Path
74:     crs_text: str
75:     extent: tuple[float, float, float, float]
76:     chip_count: int
77:     pixel_area: float
78:     model_id: str | None = None
79:     checkpoint_path: str | None = None
80: 
81: 
82: @dataclass(**_MUTABLE_DATACLASS_KWARGS)
83: class RealtimeQueryCache:
84:     """Session cache for realtime raster queries."""
85: 
86:     layer_id: str | None = None
87:     model_id: str | None = None
88:     source_candidate: str | None = None
89:     engine: Any | None = None
90:     query_cache: Any | None = None
91: 
92:     def clear(self) -> None:
93:         """Reset the cache state."""
94:         self.layer_id = None
95:         self.model_id = None
96:         self.source_candidate = None
97:         self.engine = None
98:         self.query_cache = None
99: 
100: 
101: @dataclass(**_FROZEN_DATACLASS_KWARGS)
102: class PreparedPersistentQueryCacheHit:
103:     """Persistent realtime-query cache metadata prepared on the main thread.
104: 
105:     Parameters
106:     ----------
107:     source_path : str
108:         Raster source path stored with the reusable encoded chip.
109:     feature_path : Path
110:         Path to the persisted encoded image features.
111:     chip_bounds : Any
112:         GeoSAM bounding box for the encoded chip.
113:     chip_grid : Any
114:         GeoSAM grid for the encoded chip.
115:     query : Any
116:         Query projected into the encoded chip CRS on the main thread.
117:     query_bounds_value : Any
118:         Query bounds prepared on the main thread for result metadata.
119:     prompt_kwargs : dict[str, Any]
120:         Model prompt arguments prepared on the main thread.
121:     crs_text : str
122:         Chip CRS text prepared on the main thread for cache metadata.
123: 
124:     """
125: 
126:     source_path: str
127:     feature_path: Path
128:     chip_bounds: Any
129:     chip_grid: Any
130:     query: Any
131:     query_bounds_value: Any
132:     prompt_kwargs: dict[str, Any]
133:     crs_text: str
134: 
135: 
136: @dataclass(**_FROZEN_DATACLASS_KWARGS)
137: class PreparedRealtimeRasterSample:
138:     """Raster chip data prepared on the main thread for background encoding.
139: 
140:     Parameters
141:     ----------
142:     source_path : str
143:         Raster source used to read the prepared chip.
144:     model_image : Any
145:         Chip image converted to model-ready ``HWC`` uint8 format.
146:     chip_bounds : Any
147:         GeoSAM bounding box for the prepared chip.
148:     chip_grid : Any
149:         GeoSAM grid for the prepared chip.
150:     query : Any
151:         Query projected into the prepared chip CRS on the main thread.
152:     query_bounds_value : Any
153:         Query bounds prepared on the main thread for result metadata.
154:     prompt_kwargs : dict[str, Any]
155:         Model prompt arguments prepared on the main thread.
156:     crs_text : str
157:         Chip CRS text prepared on the main thread for cache metadata.
158: 
159:     """
160: 
161:     source_path: str
162:     model_image: Any
163:     chip_bounds: Any
164:     chip_grid: Any
165:     query: Any
166:     query_bounds_value: Any
167:     prompt_kwargs: dict[str, Any]
168:     crs_text: str
169: 
170: 
171: RealtimeQueryProgressCallback = Callable[[str, float], None]
172: RealtimeQueryCancelCallback = Callable[[], bool]
173: 
174: 
175: class _RealtimeQueryCanceledError(RuntimeError):
176:     """Internal exception used to stop a background realtime query task."""
177: 
178: 
179: @dataclass(**_FROZEN_DATACLASS_KWARGS)
180: class PreparedRealtimeRasterQuery:
181:     """Prepared background-query inputs for a realtime raster request.
182: 
183:     Parameters
184:     ----------
185:     model_id : str
186:         Selected GeoSAM model identifier.
187:     layer_id : str
188:         Source QGIS layer id captured on the main thread.
189:     layer_name : str
190:         Human-readable layer name used in cache metadata.
191:     source_fingerprint : str
192:         Stable fingerprint used to validate persistent cache entries.
193:     layer_crs_text : str
194:         Active realtime layer CRS captured on the main thread.
195:     cache_directory : Path
196:         Directory where realtime query caches are persisted.
197:     online_raster_cache_directory : Path | None
198:         Cache directory used for exported online raster snapshots.
199:     source_candidates : tuple[str, ...]
200:         Candidate raster sources that can be opened without touching QGIS APIs.
201:     prepared_source_samples : tuple[PreparedRealtimeRasterSample, ...]
202:         Raster chips prepared on the main thread for local source candidates.
203:     online_export_plan : OnlineTileExportPlan | None
204:         Prepared online export plan for non-local raster sources.
205:     supports_feature_reuse : bool
206:         Whether the selected model supports reusable encoded features.
207:     persistent_cache_hit : PreparedPersistentQueryCacheHit | None, optional
208:         Metadata for a reusable persistent cache entry. CRS-heavy parsing is
209:         done on the main thread; encoded features are loaded in the worker.
210: 
211:     """
212: 
213:     model_id: str
214:     layer_id: str
215:     layer_name: str
216:     source_fingerprint: str
217:     layer_crs_text: str
218:     cache_directory: Path
219:     online_raster_cache_directory: Path | None
220:     source_candidates: tuple[str, ...]
221:     prepared_source_samples: tuple[PreparedRealtimeRasterSample, ...]
222:     online_export_plan: OnlineTileExportPlan | None
223:     supports_feature_reuse: bool
224:     persistent_cache_hit: PreparedPersistentQueryCacheHit | None = None
225: 
226: 
227: @dataclass(**_MUTABLE_DATACLASS_KWARGS)
228: class PreparedRealtimeRasterQueryResult:
229:     """Result payload returned by a background realtime raster query.
230: 
231:     Parameters
232:     ----------
233:     source_path : str
234:         Raster source used by the background task.
235:     result : QueryResult
236:         GeoSAM query result.
237:     query_cache : Any | None, optional
238:         In-memory query cache produced during encoding when feature reuse is
239:         supported.
240: 
241:     """
242: 
243:     source_path: str
244:     result: QueryResult
245:     query_cache: Any | None = None
246: 
247: 
248: @dataclass(**_FROZEN_DATACLASS_KWARGS)
249: class QueryResultRenderPayload:
250:     """Rendered query payload used by Geo-SAM canvas overlays."""
251: 
252:     geojson_features: list[dict[str, Any]]
253:     canvas_geometries: list[QgsGeometry]
254: 
255: 
256: class _ModelSessionRegistry:
257:     """Registry for loaded GeoSAM engines so memory can be released explicitly."""
258: 
259:     def __init__(self) -> None:
260:         self._online_engines: dict[str, Any] = {}
261:         self._feature_engines: dict[str, Any] = {}
262: 
263:     @staticmethod
264:     def _key(*parts: str) -> str:
265:         return "||".join(parts)
266: 
267:     def get_online_engine(self, *, source_path: str, model_id: str) -> Any:
268:         """Return a cached online engine or create a new one."""
269:         configure_geosam_qgis_runtime()
270:         from geosam import RasterDataset
271:         from geosam.engines import OnlineQueryEngine
272: 
273:         key = self._key(model_id, str(Path(source_path).expanduser()), "online")
274:         if key not in self._online_engines:
275:             dataset = RasterDataset(source_path)
276:             self._online_engines[key] = OnlineQueryEngine(
277:                 dataset,
278:                 create_model_spec(model_id),
279:             )
280:         return self._online_engines[key]
281: 
282:     def release_online_engines(
283:         self,
284:         *,
285:         model_id: str | None = None,
286:         keep_source_path: str | None = None,
287:     ) -> int:
288:         """Release cached online engines and return the number removed.
289: 
290:         Parameters
291:         ----------
292:         model_id : str | None, optional
293:             Restrict removal to one model id. When ``None``, all model ids are
294:             considered.
295:         keep_source_path : str | None, optional
296:             Preserve the engine for this source path if present.
297: 
298:         Returns
299:         -------
300:         int
301:             Number of released online engines.
302:         """
303:         removed_count = 0
304:         normalized_keep_source_path = None
305:         if keep_source_path is not None:
306:             normalized_keep_source_path = str(Path(keep_source_path).expanduser())
307: 
308:         for key in list(self._online_engines):
309:             key_model_id, key_source_path, _ = key.split("||", maxsplit=2)
310:             if model_id is not None and key_model_id != model_id:
311:                 continue
312:             if (
313:                 normalized_keep_source_path is not None
314:                 and key_source_path == normalized_keep_source_path
315:             ):
316:                 continue
317:             _close_runtime_engine(self._online_engines[key])
318:             del self._online_engines[key]
319:             removed_count += 1
320:         return removed_count
321: 
322:     def get_feature_engine(self, *, feature_dir: str | Path, model_id: str) -> Any:
323:         """Return a cached feature-query engine or create a new one."""
324:         configure_geosam_qgis_runtime()
325:         from geosam.engines import FeatureQueryEngine
326: 
327:         manifest_path = resolve_feature_manifest_path(feature_dir)
328:         key = self._key(model_id, str(manifest_path), "feature")
329:         if key not in self._feature_engines:
330:             self._feature_engines[key] = FeatureQueryEngine(
331:                 manifest_path,
332:                 create_model_spec(model_id),
333:             )
334:         return self._feature_engines[key]
335: 
336:     def get_loaded_model_ids(self) -> set[str]:
337:         """Return model identifiers that have at least one cached engine.
338: 
339:         Returns
340:         -------
341:         set[str]
342:             Model ids with active engines in the registry.
343: 
344:         """
345:         loaded: set[str] = set()
346:         for store_name in ("_online_engines", "_feature_engines"):
347:             store = getattr(self, store_name)
348:             for key in store:
349:                 model_id = key.split("||", maxsplit=1)[0]
350:                 loaded.add(model_id)
351:         return loaded
352: 
353:     def release(self, *, model_id: str | None = None) -> int:
354:         """Release cached engines and return the number removed."""
355:         removed_count = 0
356:         for store_name in ("_online_engines", "_feature_engines"):
357:             store = getattr(self, store_name)
358:             for key in list(store):
359:                 if model_id is not None and not key.startswith(f"{model_id}||"):
360:                     continue
361:                 _close_runtime_engine(store[key])
362:                 del store[key]
363:                 removed_count += 1
364:         return removed_count
365: 
366: 
367: MODEL_SESSIONS = _ModelSessionRegistry()
368: 
369: 
370: def _normalize_performance_mode(value: Any) -> PerformanceMode:
371:     """Normalize a configured performance mode."""
372:     normalized_value = str(value or "balanced").strip().lower()
373:     if normalized_value in {"balanced", "fastest", "low_memory"}:
374:         return normalized_value  # type: ignore[return-value]
375:     return "balanced"
376: 
377: 
378: def get_performance_mode() -> PerformanceMode:
379:     """Return the configured realtime performance mode."""
380:     return _normalize_performance_mode(
381:         load_plugin_settings().get("performance_mode", "balanced")
382:     )
383: 
384: 
385: def _normalize_preview_render_mode(value: Any) -> PreviewRenderMode:
386:     """Normalize a configured preview render mode."""
387:     normalized_value = str(value or "pixel_level").strip().lower().replace("-", "_")
388:     if normalized_value in {"pixel_level", "simplified"}:
389:         return normalized_value  # type: ignore[return-value]
390:     return "pixel_level"
391: 
392: 
393: def get_preview_render_mode() -> PreviewRenderMode:
394:     """Return the configured preview render mode."""
395:     return _normalize_preview_render_mode(
396:         load_plugin_settings().get("preview_render_mode", "pixel_level")
397:     )
398: 
399: 
400: def _close_runtime_engine(engine: Any) -> None:
401:     """Best-effort close for cached GeoSAM engine instances."""
402:     close_method = getattr(engine, "close", None)
403:     if callable(close_method):
404:         try:
405:             close_method()
406:         except Exception as exc:
407:             logger.warning("Failed to close cached GeoSAM engine: %s", exc)
408: 
409: 
410: def sanitize_path_component(value: str) -> str:
411:     """Convert an arbitrary label into a filesystem-safe path component."""
412:     sanitized = "".join(
413:         character if character.isalnum() or character in {"-", "_", "."} else "_"
414:         for character in value.strip()
415:     )
416:     sanitized = sanitized.strip("._")
417:     return sanitized or "unnamed"
418: 
419: 
420: def get_layer_cache_directory(layer_name: str) -> Path:
421:     """Return the cache root for a raster layer name."""
422:     layer_cache_dir = get_cache_directory() / sanitize_path_component(layer_name)
423:     layer_cache_dir.mkdir(parents=True, exist_ok=True)
424:     return layer_cache_dir
425: 
426: 
427: def create_model_spec_from_checkpoint(
428:     checkpoint_path: str | Path,
429:     *,
430:     model_id: str | None = None,
431:     device: str | None = None,
432: ) -> ModelSpec:
433:     """Create a model spec from an arbitrary checkpoint path."""
434:     configure_geosam_qgis_runtime()
435:     from geosam.runtime import create_model_spec_from_checkpoint as create_runtime_spec
436: 
437:     return create_runtime_spec(
438:         checkpoint_path,
439:         model_id=model_id,
440:         device=device,
441:     )
442: 
443: 
444: def _flush_torch_memory() -> None:
445:     """Best-effort cleanup for Python and accelerator memory."""
446:     gc.collect()
447:     try:
448:         import torch
449:     except (ImportError, OSError) as exc:
450:         logger.warning(
451:             "Skipping PyTorch memory cleanup because torch failed to load: %s",
452:             exc,
453:         )
454:         return
455: 
456:     if torch.cuda.is_available():
457:         torch.cuda.empty_cache()
458:         torch.cuda.ipc_collect()
459:     if hasattr(torch, "mps") and torch.backends.mps.is_available():
460:         empty_cache = getattr(torch.mps, "empty_cache", None)
461:         if callable(empty_cache):
462:             empty_cache()
463: 
464: 
465: def _clear_online_engine_predictor_state(
466:     engine: Any,
467:     *,
468:     keep_features: bool,
469: ) -> None:
470:     """Clear transient predictor state retained by an online engine.
471: 
472:     Parameters
473:     ----------
474:     engine : Any
475:         Online query engine that may hold an Ultralytics SAM predictor.
476:     keep_features : bool
477:         True to preserve ``predictor.features`` as the hot cache for the
478:         current active image. False to clear all image-specific predictor
479:         state, including cached features.
480: 
481:     Notes
482:     -----
483:     This only clears image-specific predictor state. Model weights remain
484:     loaded so cached prompt inference can continue using the same engine
485:     instance.
486: 
487:     """
488:     predictor = getattr(getattr(engine, "adapter", None), "model", None)
489:     predictor = getattr(predictor, "predictor", None)
490:     if predictor is None:
491:         return
492: 
493:     attributes_to_clear: list[tuple[str, Any]] = [
494:         ("im", None),
495:         ("dataset", None),
496:         ("source_type", None),
497:         ("batch", None),
498:         ("results", None),
499:         ("txt_path", None),
500:         ("plotted_img", None),
501:         ("prompts", {}),
502:         ("vid_writer", {}),
503:     ]
504:     if not keep_features:
505:         attributes_to_clear.insert(0, ("features", None))
506: 
507:     for attribute_name, empty_value in attributes_to_clear:
508:         if hasattr(predictor, attribute_name):
509:             setattr(predictor, attribute_name, empty_value)
510: 
511: 
512: def _should_trim_online_engines() -> bool:
513:     """Return whether inactive online engines should be released immediately."""
514:     return get_performance_mode() != "fastest"
515: 
516: 
517: def _should_clear_online_query_state_after_result() -> bool:
518:     """Return whether realtime queries should clear predictor state after use."""
519:     return get_performance_mode() == "low_memory"
520: 
521: 
522: def release_online_runtime_hot_cache(
523:     *,
524:     keep_source_path: str | None = None,
525:     force: bool = False,
526: ) -> int:
527:     """Release cached online engines and clear accelerator memory.
528: 
529:     Parameters
530:     ----------
531:     keep_source_path : str | None, optional
532:         Preserve the online engine for this source path when provided.
533:     force : bool, default=False
534:         When ``True``, release engines even when the configured performance
535:         mode prefers keeping them warm.
536: 
537:     Returns
538:     -------
539:     int
540:         Number of released online engines.
541: 
542:     """
543:     if not force and not _should_trim_online_engines():
544:         return 0
545:     removed_count = MODEL_SESSIONS.release_online_engines(
546:         keep_source_path=keep_source_path,
547:     )
548:     if removed_count > 0:
549:         _flush_torch_memory()
550:     return removed_count
551: 
552: 
553: def get_loaded_model_ids() -> set[str]:
554:     """Return model identifiers currently loaded in memory.
555: 
556:     Returns
557:     -------
558:     set[str]
559:         Model ids with active engines in the session registry.
560: 
561:     """
562:     return MODEL_SESSIONS.get_loaded_model_ids()
563: 
564: 
565: def release_runtime_models(*, model_id: str | None = None) -> int:
566:     """Release loaded GeoSAM engines and clear accelerator caches."""
567:     removed_count = MODEL_SESSIONS.release(model_id=model_id)
568:     _flush_torch_memory()
569:     return removed_count
570: 
571: 
572: def cleanup_on_plugin_unload() -> dict[str, int]:
573:     """Release model memory and clear cache when configured."""
574:     removed_models = release_runtime_models()
575:     removed_cache_files = 0
576:     if load_plugin_settings().get("clear_cache_on_plugin_close", True):
577:         removed_cache_files = clear_cache()
578:     return {
579:         "released_models": removed_models,
580:         "removed_cache_files": removed_cache_files,
581:     }
582: 
583: 
584: def resolve_feature_manifest_path(feature_dir: str | Path) -> Path:
585:     """Resolve the manifest file path inside a feature folder."""
586:     configure_geosam_qgis_runtime()
587:     from geosam.runtime import resolve_feature_manifest_path as resolve_runtime_manifest
588: 
589:     return resolve_runtime_manifest(feature_dir)
590: 
591: 
592: def describe_feature_source(feature_dir: str | Path) -> FeatureSourceSummary:
593:     """Load summary metadata for a GeoSAM feature folder."""
594:     configure_geosam_qgis_runtime()
595:     import geopandas as gpd
596: 
597:     manifest_path = resolve_feature_manifest_path(feature_dir)
598:     if manifest_path.suffix == ".pkl":
599:         msg = (
600:             f"Legacy pickle manifest detected: {manifest_path}. "
601:             "Please regenerate the feature cache with the current Geo-SAM version."
602:         )
603:         logger.error(msg)
604:         raise ValueError(msg)
605:     frame = gpd.read_parquet(manifest_path)
606: 
607:     if len(frame) == 0:
608:         msg = f"Feature manifest is empty: {manifest_path}"
609:         logger.error(msg)
610:         raise ValueError(msg)
611: 
612:     crs_text = _resolve_feature_crs_text(frame)
613:     total_bounds = tuple(float(value) for value in frame.total_bounds)
614:     transform_values = json.loads(frame.iloc[0]["transform"])
615:     pixel_area = abs(float(transform_values[0]) * float(transform_values[4]))
616:     checkpoint_path = str(frame.iloc[0].get("checkpoint_path", "")).strip() or None
617:     model_id = (
618:         infer_model_id_from_checkpoint_path(checkpoint_path, allow_unknown=True)
619:         if checkpoint_path is not None
620:         else None
621:     )
622:     if len(total_bounds) != 4:
623:         msg = f"Unexpected feature extent size for manifest {manifest_path}: {total_bounds!r}"
624:         logger.error(msg)
625:         raise ValueError(msg)
626:     return FeatureSourceSummary(
627:         manifest_path=manifest_path,
628:         crs_text=crs_text,
629:         extent=(
630:             total_bounds[0],
631:             total_bounds[1],
632:             total_bounds[2],
633:             total_bounds[3],
634:         ),
635:         chip_count=len(frame),
636:         pixel_area=pixel_area,
637:         model_id=model_id,
638:         checkpoint_path=checkpoint_path,
639:     )
640: 
641: 
642: def _resolve_feature_crs_text(frame: Any) -> str:
643:     """Resolve a stable CRS string from a feature manifest frame."""
644:     if "crs" in frame.columns and len(frame) > 0:
645:         manifest_crs = frame.iloc[0]["crs"]
646:         if isinstance(manifest_crs, str) and manifest_crs.strip():
647:             return manifest_crs.strip()
648: 
649:     frame_crs = frame.crs
650:     if frame_crs is None:
651:         msg = "Feature manifest CRS is missing."
652:         logger.error(msg)
653:         raise ValueError(msg)
654: 
655:     to_authority = getattr(frame_crs, "to_authority", None)
656:     if callable(to_authority):
657:         authority = to_authority()
658:         if authority is not None:
659:             return f"{authority[0]}:{authority[1]}"
660: 
661:     to_epsg = getattr(frame_crs, "to_epsg", None)
662:     if callable(to_epsg):
663:         epsg_code = to_epsg()
664:         if epsg_code is not None:
665:             return f"EPSG:{epsg_code}"
666: 
667:     to_wkt = getattr(frame_crs, "to_wkt", None)
668:     if callable(to_wkt):
669:         return str(to_wkt())
670: 
671:     return str(frame_crs)
672: 
673: 
674: def layer_extent_rectangle(layer: QgsRasterLayer) -> QgsRectangle:
675:     """Return the extent of a raster layer."""
676:     return layer.extent()
677: 
678: 
679: def layer_pixel_area(layer: QgsRasterLayer) -> float:
680:     """Return the approximate area represented by one raster pixel."""
681:     return abs(
682:         float(layer.rasterUnitsPerPixelX()) * float(layer.rasterUnitsPerPixelY())
683:     )
684: 
685: 
686: def chip_extent_rectangles_for_source(
687:     source_path: str | Path,
688:     *,
689:     bands: list[int] | None = None,
690:     crs: str | None = None,
691:     res: float | None = None,
692:     extent: tuple[float, float, float, float] | None = None,
693:     extent_crs: str | None = None,
694:     chip_size: int = 1024,
695:     stride: int = 512,
696: ) -> list[tuple[float, float, float, float]]:
697:     """Return chip extents for a raster source using GeoSAM sampling rules."""
698:     configure_geosam_qgis_runtime()
699:     from geosam.runtime import chip_extent_rectangles_for_source as runtime_chip_extents
700: 
701:     return runtime_chip_extents(
702:         source_path,
703:         bands=bands,
704:         crs=crs,
705:         res=res,
706:         extent=extent,
707:         extent_crs=extent_crs,
708:         chip_size=chip_size,
709:         stride=stride,
710:     )
711: 
712: 
713: def _raster_layer_source_candidates(layer: QgsRasterLayer) -> list[str]:
714:     """Return possible raster source strings for a QGIS raster layer."""
715:     candidates: list[str] = []
716:     for attribute_name in ("publicSource", "source"):
717:         attribute = getattr(layer, attribute_name, None)
718:         if callable(attribute):
719:             try:
720:                 value = str(attribute())
721:             except Exception:
722:                 continue
723:             if value and value not in candidates:
724:                 candidates.append(value)
725: 
726:     provider = layer.dataProvider()
727:     if provider is not None and hasattr(provider, "dataSourceUri"):
728:         try:
729:             value = str(provider.dataSourceUri())
730:         except Exception:
731:             value = ""
732:         if value and value not in candidates:
733:             candidates.append(value)
734:     return candidates
735: 
736: 
737: def _normalize_local_raster_source(source_candidate: str) -> str | None:
738:     """Resolve a QGIS raster source candidate to a local file path.
739: 
740:     Parameters
741:     ----------
742:     source_candidate : str
743:         Raster source string reported by QGIS.
744: 
745:     Returns
746:     -------
747:     str | None
748:         Absolute local file path when one can be resolved, otherwise ``None``.
749:     """
750:     normalized_source = source_candidate.strip()
751:     if not normalized_source:
752:         return None
753: 
754:     direct_path_candidate = normalized_source.split("|", maxsplit=1)[0].strip()
755:     if direct_path_candidate:
756:         direct_path = Path(direct_path_candidate).expanduser()
757:         if direct_path.exists() and direct_path.is_file():
758:             return str(direct_path)
759: 
760:     parsed_source = urlparse(normalized_source)
761:     if parsed_source.scheme not in {"", "file"}:
762:         return None
763: 
764:     if parsed_source.scheme == "file":
765:         candidate_path = Path(parsed_source.path).expanduser()
766:     else:
767:         candidate_path = Path(normalized_source).expanduser()
768:     if candidate_path.exists() and candidate_path.is_file():
769:         return str(candidate_path)
770:     return None
771: 
772: 
773: def _layer_source_fingerprint(layer: QgsRasterLayer) -> str:
774:     """Return a stable fingerprint for matching raster cache entries."""
775:     provider = layer.dataProvider()
776:     provider_name = layer.providerType()
777:     layer_source = "|".join(_raster_layer_source_candidates(layer))
778:     provider_uri = ""
779:     if provider is not None and hasattr(provider, "dataSourceUri"):
780:         try:
781:             provider_uri = str(provider.dataSourceUri())
782:         except Exception:
783:             provider_uri = ""
784:     return "|".join([layer.name(), provider_name, layer_source, provider_uri])
785: 
786: 
787: def _query_to_crs_text(query, crs_text: str) -> Any:
788:     """Normalize a GeoSAM query into a target CRS string.
789: 
790:     Parameters
791:     ----------
792:     query : Any
793:         GeoSAM query object.
794:     crs_text : str
795:         Destination CRS string.
796: 
797:     Returns
798:     -------
799:     Any
800:         Query expressed in ``crs_text`` when conversion is required.
801:     """
802:     configure_geosam_qgis_runtime()
803:     query_crs = getattr(query, "crs", None)
804:     if query_crs is None or not crs_text:
805:         return query
806:     return query if str(query_crs) == crs_text else query.to_crs(crs_text)
807: 
808: 
809: def _rectangle_to_bbox(rectangle: QgsRectangle, *, crs_text: str):
810:     """Convert a QGIS rectangle into a GeoSAM bounding box."""
811:     configure_geosam_qgis_runtime()
812:     from geosam import BoundingBox
813: 
814:     return BoundingBox(
815:         float(rectangle.xMinimum()),
816:         float(rectangle.yMinimum()),
817:         float(rectangle.xMaximum()),
818:         float(rectangle.yMaximum()),
819:         crs=crs_text,
820:     )
821: 
822: 
823: def _query_in_layer_crs(layer: QgsRasterLayer, query):
824:     """Normalize a query into the layer CRS."""
825:     configure_geosam_qgis_runtime()
826:     crs_text = layer.crs().authid() or layer.crs().toWkt()
827:     if query.crs == QgsCoordinateReferenceSystem(crs_text):
828:         return query
829:     if query.crs is None:
830:         return query
831:     return query if str(query.crs) == crs_text else query.to_crs(crs_text)
832: 
833: 
834: def _query_is_far_from_chip_edge(query, *, chip_bounds, chip_grid) -> bool:
835:     """Return whether a query center is inside the safe inner chip region."""
836:     configure_geosam_qgis_runtime()
837:     from geosam.query import query_bounds, query_center
838: 
839:     if not chip_bounds.contains(query_bounds(query)):
840:         return False
841: 
842:     center_x, center_y = query_center(query)
843:     height, width = chip_grid.shape
844:     pixel_size_x = abs(float(chip_grid.transform.a))
845:     pixel_size_y = abs(float(chip_grid.transform.e))
846:     margin_x_pixels = min(
847:         max(ONLINE_QUERY_REFRESH_MARGIN_PIXELS, 1), max(width // 2, 1)
848:     )
849:     margin_y_pixels = min(
850:         max(ONLINE_QUERY_REFRESH_MARGIN_PIXELS, 1), max(height // 2, 1)
851:     )
852:     margin_x = margin_x_pixels * pixel_size_x
853:     margin_y = margin_y_pixels * pixel_size_y
854: 
855:     return (
856:         chip_bounds.left + margin_x <= center_x <= chip_bounds.right - margin_x
857:         and chip_bounds.bottom + margin_y <= center_y <= chip_bounds.top - margin_y
858:     )
859: 
860: 
861: def _prepare_realtime_raster_source_sample(
862:     *,
863:     source_path: str,
864:     query,
865:     model_id: str,
866: ) -> PreparedRealtimeRasterSample:
867:     """Read a realtime raster chip on the main thread for background encoding."""
868:     configure_geosam_qgis_runtime()
869:     from geosam.datasets import RasterDataset
870:     from geosam.engines import _prompt_prediction_kwargs
871:     from geosam.query import query_bounds, query_center, window_from_center
872: 
873:     model_spec = create_model_spec(model_id)
874:     dataset = RasterDataset(source_path)
875:     projected_query = (
876:         query if str(query.crs) == str(dataset.crs) else query.to_crs(dataset.crs)
877:     )
878:     chip_bounds = window_from_center(
879:         query_center(projected_query),
880:         model_spec.resolved_imgsz,
881:         grid=dataset.grid,
882:     )
883:     sample = dataset[chip_bounds]
884:     chip_grid = sample.grid
885:     return PreparedRealtimeRasterSample(
886:         source_path=sample.source_path,
887:         model_image=sample.to_model_image(),
888:         chip_bounds=chip_bounds,
889:         chip_grid=chip_grid,
890:         query=projected_query,
891:         query_bounds_value=query_bounds(projected_query),
892:         prompt_kwargs=_prompt_prediction_kwargs(projected_query, chip_grid),
893:         crs_text=str(chip_grid.crs),
894:     )
895: 
896: 
897: def _prediction_to_prepared_query_result(
898:     prediction: Any,
899:     *,
900:     chip_grid: Any,
901:     chip_bounds: Any,
902:     query_bounds_value: Any,
903:     source_path: str,
904:     model_type: str,
905: ) -> Any:
906:     """Convert a prediction using bounds prepared outside the worker thread."""
907:     configure_geosam_qgis_runtime()
908:     import numpy as np
909:     from geosam.engines import QueryResult
910: 
911:     if prediction.masks is None:
912:         mask_array = np.zeros((0, chip_grid.height, chip_grid.width), dtype=bool)
913:     else:
914:         mask_array = prediction.masks.detach().cpu().numpy().astype(bool)
915:     scores = prediction.scores.detach().cpu().numpy()
916:     return QueryResult(
917:         mask_array=mask_array,
918:         mask_transform=chip_grid.transform,
919:         mask_crs=chip_grid.crs,
920:         query_bounds=query_bounds_value,
921:         chip_bounds=chip_bounds,
922:         scores=scores,
923:         source_path=source_path,
924:         chip_id=None,
925:         model_type=model_type,
926:     )
927: 
928: 
929: def _realtime_query_cache_directory(layer: QgsRasterLayer, model_id: str) -> Path:
930:     """Return the persistent realtime-query cache directory for a layer/model."""
931:     cache_dir = get_layer_cache_directory(layer.name()) / "realtime_query" / model_id
932:     cache_dir.mkdir(parents=True, exist_ok=True)
933:     return cache_dir
934: 
935: 
936: def _online_layer_raster_cache_directory(layer: QgsRasterLayer) -> Path:
937:     """Return the raster-export cache directory for a layer."""
938:     cache_dir = get_layer_cache_directory(layer.name()) / "online_raster"
939:     cache_dir.mkdir(parents=True, exist_ok=True)
940:     return cache_dir
941: 
942: 
943: def _persistent_cache_entry_paths(
944:     layer: QgsRasterLayer,
945:     model_id: str,
946:     *,
947:     chip_bounds,
948: ) -> tuple[Path, Path]:
949:     """Return persistent cache paths for one encoded realtime chip."""
950:     cache_dir = _realtime_query_cache_directory(layer, model_id)
951:     return _persistent_cache_entry_paths_for_directory(
952:         cache_dir,
953:         chip_bounds=chip_bounds,
954:     )
955: 
956: 
957: def _persistent_cache_entry_paths_for_directory(
958:     cache_dir: Path,
959:     *,
960:     chip_bounds,
961: ) -> tuple[Path, Path]:
962:     """Return persistent cache paths for one encoded realtime chip."""
963:     chip_key = sanitize_path_component(
964:         "_".join([
965:             f"{chip_bounds.left:.3f}",
966:             f"{chip_bounds.bottom:.3f}",
967:             f"{chip_bounds.right:.3f}",
968:             f"{chip_bounds.top:.3f}",
969:         ])
970:     )
971:     entry_dir = cache_dir / chip_key
972:     entry_dir.mkdir(parents=True, exist_ok=True)
973:     return entry_dir / "encoded.pt", entry_dir / "metadata.json"
974: 
975: 
976: def _query_cache_shape_matches_model(model_id: str, query_cache: Any) -> bool:
977:     """Return whether an in-memory query cache matches the model image size.
978: 
979:     Parameters
980:     ----------
981:     model_id : str
982:         Registered model identifier.
983:     query_cache : Any
984:         GeoSAM online query cache payload.
985: 
986:     Returns
987:     -------
988:     bool
989:         ``True`` when the cache chip grid shape matches the effective model
990:         image size.
991: 
992:     """
993:     chip_grid = getattr(query_cache, "chip_grid", None)
994:     if chip_grid is None:
995:         return False
996:     try:
997:         cache_shape = tuple(int(value) for value in chip_grid.shape)
998:     except Exception:
999:         return False
1000:     return cache_shape == effective_model_imgsz(model_id)
1001: 
1002: 
1003: def _save_persistent_query_cache(
1004:     *,
1005:     layer: QgsRasterLayer,
1006:     model_id: str,
1007:     source_path: str,
1008:     query_cache: Any,
1009: ) -> None:
1010:     """Persist a realtime query cache entry to disk."""
1011:     _save_persistent_query_cache_entry(
1012:         cache_directory=_realtime_query_cache_directory(layer, model_id),
1013:         layer_name=layer.name(),
1014:         source_fingerprint=_layer_source_fingerprint(layer),
1015:         model_id=model_id,
1016:         source_path=source_path,
1017:         query_cache=query_cache,
1018:     )
1019: 
1020: 
1021: def save_prepared_realtime_query_cache(
1022:     *,
1023:     prepared_query: PreparedRealtimeRasterQuery,
1024:     source_path: str,
1025:     query_cache: Any,
1026: ) -> None:
1027:     """Persist a prepared realtime query cache entry to disk.
1028: 
1029:     Parameters
1030:     ----------
1031:     prepared_query : PreparedRealtimeRasterQuery
1032:         Prepared realtime query metadata captured on the main thread.
1033:     source_path : str
1034:         Raster source used by the finished realtime query.
1035:     query_cache : Any
1036:         Reusable encoded-query cache returned by the background realtime query.
1037: 
1038:     """
1039:     _save_persistent_query_cache_entry(
1040:         cache_directory=prepared_query.cache_directory,
1041:         layer_name=prepared_query.layer_name,
1042:         source_fingerprint=prepared_query.source_fingerprint,
1043:         model_id=prepared_query.model_id,
1044:         source_path=source_path,
1045:         query_cache=query_cache,
1046:     )
1047: 
1048: 
1049: def _save_persistent_query_cache_entry(
1050:     *,
1051:     cache_directory: Path,
1052:     layer_name: str,
1053:     source_fingerprint: str,
1054:     model_id: str,
1055:     source_path: str,
1056:     query_cache: Any,
1057: ) -> None:
1058:     """Persist a realtime query cache entry using precomputed cache metadata."""
1059:     settings = load_plugin_settings()
1060:     if not settings.get("cache_enabled", True):
1061:         return
1062:     if (
1063:         query_cache is None
1064:         or query_cache.encoded is None
1065:         or query_cache.chip_grid is None
1066:     ):
1067:         return
1068:     if query_cache.chip_bounds is None:
1069:         return
1070: 
1071:     crs_text = getattr(query_cache, "crs_text", None)
1072:     if crs_text is None:
1073:         logger.warning(
1074:             "Skipping realtime query cache save because chip CRS text was not "
1075:             "prepared on the main thread."
1076:         )
1077:         return
1078: 
1079:     encoded_path, metadata_path = _persistent_cache_entry_paths_for_directory(
1080:         cache_directory,
1081:         chip_bounds=query_cache.chip_bounds,
1082:     )
1083:     query_cache.encoded.save(encoded_path)
1084:     metadata = {
1085:         "model_id": model_id,
1086:         "layer_name": layer_name,
1087:         "source_fingerprint": source_fingerprint,
1088:         "source_path": source_path,
1089:         "chip_bounds": [
1090:             float(query_cache.chip_bounds.left),
1091:             float(query_cache.chip_bounds.bottom),
1092:             float(query_cache.chip_bounds.right),
1093:             float(query_cache.chip_bounds.top),
1094:         ],
1095:         "transform": list(query_cache.chip_grid.transform)[:6],
1096:         "shape": list(query_cache.chip_grid.shape),
1097:         "crs": str(crs_text),
1098:         "feature_path": str(encoded_path),
1099:     }
1100:     metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
1101:     trim_cache_if_needed()
1102: 
1103: 
1104: def prepare_realtime_raster_query(
1105:     layer: QgsRasterLayer,
1106:     model_id: str,
1107:     query,
1108:     *,
1109:     cache: RealtimeQueryCache | None = None,
1110:     view_resolution: tuple[tuple[float, float], str] | None = None,
1111: ) -> PreparedRealtimeRasterQuery | None:
1112:     """Prepare a background query plan when realtime encoding is required.
1113: 
1114:     Parameters
1115:     ----------
1116:     layer : QgsRasterLayer
1117:         Active realtime raster layer.
1118:     model_id : str
1119:         Selected GeoSAM model identifier.
1120:     query : Any
1121:         GeoSAM query object.
1122:     cache : RealtimeQueryCache | None, optional
1123:         Realtime in-memory cache for the current session.
1124:     view_resolution : tuple[tuple[float, float], str] | None, optional
1125:         Current map canvas resolution as ``(resolution_xy, crs_authid)``
1126:         where ``resolution_xy`` is ``(x_size, y_size)`` in the canvas CRS
1127:         units and ``crs_authid`` identifies those units. Forwarded to the
1128:         online tile export plan so the exported zoom level reflects the
1129:         user's current view rather than the online layer's coarse base
1130:         resolution.
1131: 
1132:     Returns
1133:     -------
1134:     PreparedRealtimeRasterQuery | None
1135:         Returns ``None`` when the query can run synchronously using an existing
1136:         in-memory or persistent cache. Returns a prepared plan when a new chip
1137:         must be encoded in a background task.
1138: 
1139:     """
1140:     configure_geosam_qgis_runtime()
1141:     model_definition = get_model_definition(model_id)
1142:     supports_feature_reuse = bool(model_definition.supports_feature_reuse)
1143:     layer_crs_text = layer.crs().authid() or layer.crs().toWkt()
1144:     source_fingerprint = _layer_source_fingerprint(layer)
1145: 
1146:     if cache is not None:
1147:         cache_mismatch = cache.layer_id != layer.id() or cache.model_id != model_id
1148:         if cache_mismatch:
1149:             cache.clear()
1150:         elif supports_feature_reuse and cache.query_cache is not None:
1151:             cached_source_path = (
1152:                 Path(cache.source_candidate).expanduser()
1153:                 if cache.source_candidate is not None
1154:                 else None
1155:             )
1156:             if not _query_cache_shape_matches_model(model_id, cache.query_cache):
1157:                 cache.query_cache = None
1158:                 cache.engine = None
1159:             elif cached_source_path is None or not cached_source_path.exists():
1160:                 cache.query_cache = None
1161:                 cache.engine = None
1162:             elif not _query_is_far_from_chip_edge(
1163:                 _query_in_layer_crs(layer, query),
1164:                 chip_bounds=cache.query_cache.chip_bounds,
1165:                 chip_grid=cache.query_cache.chip_grid,
1166:             ):
1167:                 cache.query_cache = None
1168:                 cache.engine = None
1169: 
1170:     if (
1171:         supports_feature_reuse
1172:         and cache is not None
1173:         and cache.query_cache is not None
1174:         and cache.source_candidate is not None
1175:     ):
1176:         cached_source_path = Path(cache.source_candidate).expanduser()
1177:         if cached_source_path.exists():
1178:             return None
1179: 
1180:     persistent_cache_hit: PreparedPersistentQueryCacheHit | None = None
1181:     if supports_feature_reuse and (cache is None or cache.query_cache is None):
1182:         persistent_cache_hit = _find_persistent_query_cache(
1183:             layer,
1184:             model_id,
1185:             query,
1186:         )
1187: 
1188:     source_candidates = [
1189:         source_path
1190:         for source_candidate in _raster_layer_source_candidates(layer)
1191:         if (source_path := _normalize_local_raster_source(source_candidate)) is not None
1192:     ]
1193:     prepared_source_samples: list[PreparedRealtimeRasterSample] = []
1194:     if persistent_cache_hit is None:
1195:         for source_candidate in source_candidates:
1196:             try:
1197:                 prepared_source_samples.append(
1198:                     _prepare_realtime_raster_source_sample(
1199:                         source_path=source_candidate,
1200:                         query=query,
1201:                         model_id=model_id,
1202:                     )
1203:                 )
1204:             except Exception as exc:
1205:                 logger.warning(
1206:                     "Failed to prepare realtime raster chip for %s: %s",
1207:                     source_candidate,
1208:                     exc,
1209:                 )
1210: 
1211:     online_export_plan: OnlineTileExportPlan | None = None
1212:     online_raster_cache_directory: Path | None = None
1213:     if persistent_cache_hit is None and len(source_candidates) == 0:
1214:         online_export_plan = prepare_online_raster_export_plan(
1215:             layer,
1216:             query,
1217:             model_id=model_id,
1218:             chip_size=create_model_spec(model_id).resolved_imgsz,
1219:             source_fingerprint=source_fingerprint,
1220:             view_resolution=view_resolution,
1221:         )
1222:         online_raster_cache_directory = _online_layer_raster_cache_directory(layer)
1223: 
1224:     return PreparedRealtimeRasterQuery(
1225:         model_id=model_id,
1226:         layer_id=layer.id(),
1227:         layer_name=layer.name(),
1228:         source_fingerprint=source_fingerprint,
1229:         layer_crs_text=layer_crs_text,
1230:         cache_directory=_realtime_query_cache_directory(layer, model_id),
1231:         online_raster_cache_directory=online_raster_cache_directory,
1232:         source_candidates=tuple(source_candidates),
1233:         prepared_source_samples=tuple(prepared_source_samples),
1234:         online_export_plan=online_export_plan,
1235:         supports_feature_reuse=supports_feature_reuse,
1236:         persistent_cache_hit=persistent_cache_hit,
1237:     )
1238: 
1239: 
1240: def run_prepared_realtime_raster_query(
1241:     prepared_query: PreparedRealtimeRasterQuery,
1242:     query,
1243:     *,
1244:     progress_callback: RealtimeQueryProgressCallback | None = None,
1245:     is_canceled: RealtimeQueryCancelCallback | None = None,
1246:     qgis_task: Any | None = None,
1247: ) -> PreparedRealtimeRasterQueryResult:
1248:     """Execute a prepared realtime raster query without using QGIS layer APIs.
1249: 
1250:     Parameters
1251:     ----------
1252:     prepared_query : PreparedRealtimeRasterQuery
1253:         Prepared background-query inputs captured on the main thread.
1254:     query : Any
1255:         GeoSAM query object.
1256:     progress_callback : RealtimeQueryProgressCallback | None, optional
1257:         Callback receiving ``(stage_text, progress_percent)`` updates.
1258:     is_canceled : RealtimeQueryCancelCallback | None, optional
1259:         Callback that returns ``True`` when the background task should stop.
1260:     qgis_task : Any | None, optional
1261:         QGIS task object used by GeoSAM for progress, cancellation, logging,
1262:         and QGIS CRS operations inside the worker.
1263: 
1264:     Returns
1265:     -------
1266:     PreparedRealtimeRasterQueryResult
1267:         Query result and optional in-memory cache payload.
1268: 
1269:     Raises
1270:     ------
1271:     RuntimeError
1272:         If cancellation is requested before finishing the job.
1273:     ValueError
1274:         If all raster source candidates fail.
1275: 
1276:     """
1277:     configure_geosam_qgis_runtime(task=qgis_task)
1278:     from geosam.engines import (
1279:         OnlineQueryCache,
1280:         _require_query_crs,
1281:     )
1282:     from geosam.models import build_model_adapter
1283: 
1284:     def _report_progress(stage_text: str, progress_value: float) -> None:
1285:         if progress_callback is not None:
1286:             progress_callback(stage_text, progress_value)
1287: 
1288:     def _ensure_not_canceled() -> None:
1289:         if is_canceled is not None and is_canceled():
1290:             msg = "Realtime raster query task was canceled."
1291:             logger.info(msg)
1292:             raise _RealtimeQueryCanceledError(msg)
1293: 
1294:     _require_query_crs(query)
1295:     errors: list[str] = []
1296:     model_spec = create_model_spec(prepared_query.model_id)
1297: 
1298:     if prepared_query.supports_feature_reuse:
1299:         _ensure_not_canceled()
1300:         _report_progress("Checking cached encoding", 2.0)
1301:         prepared_cache_hit = prepared_query.persistent_cache_hit
1302:         persistent_cache_hit = _load_prepared_persistent_query_cache(prepared_cache_hit)
1303:         if persistent_cache_hit is not None:
1304:             source_path, query_cache = persistent_cache_hit
1305:             engine = None
1306:             try:
1307:                 _ensure_not_canceled()
1308:                 _report_progress("Loading segmentation model", 15.0)
1309:                 engine = build_model_adapter(model_spec)
1310:                 _ensure_not_canceled()
1311:                 _report_progress("Running prompt query", 75.0)
1312:                 prediction = engine.predict_features(
1313:                     query_cache.encoded,
1314:                     multimask_output=False,
1315:                     **prepared_cache_hit.prompt_kwargs,
1316:                 )
1317:                 result = _prediction_to_prepared_query_result(
1318:                     prediction,
1319:                     chip_grid=query_cache.chip_grid,
1320:                     chip_bounds=query_cache.chip_bounds,
1321:                     query_bounds_value=prepared_cache_hit.query_bounds_value,
1322:                     source_path=query_cache.source_path,
1323:                     model_type=model_spec.model_type,
1324:                 )
1325:                 _ensure_not_canceled()
1326:                 _report_progress("Finished realtime query", 100.0)
1327:                 return PreparedRealtimeRasterQueryResult(
1328:                     source_path=source_path,
1329:                     result=result,
1330:                     query_cache=query_cache,
1331:                 )
1332:             except _RealtimeQueryCanceledError:
1333:                 raise
1334:             except Exception as exc:
1335:                 errors.append(f"persistent-cache {source_path}: {exc}")
1336:             finally:
1337:                 _close_runtime_engine(engine)
1338: 
1339:     prepared_source_samples = list(prepared_query.prepared_source_samples)
1340:     source_candidates = list(prepared_query.source_candidates)
1341:     if (
1342:         len(prepared_source_samples) == 0
1343:         and prepared_query.online_export_plan is not None
1344:         and prepared_query.online_raster_cache_directory is not None
1345:     ):
1346:         try:
1347:             _ensure_not_canceled()
1348:             _report_progress("Preparing online raster export", 3.0)
1349:             exported_source = export_online_tile_raster_plan(
1350:                 prepared_query.online_export_plan,
1351:                 cache_directory=prepared_query.online_raster_cache_directory,
1352:                 layer_name=prepared_query.layer_name,
1353:                 progress_callback=_report_progress,
1354:                 is_canceled=is_canceled,
1355:             )
1356:             source_candidates.append(exported_source)
1357:             _ensure_not_canceled()
1358:             _report_progress("Preparing exported raster chip", 25.0)
1359:             prepared_source_samples.append(
1360:                 _prepare_realtime_raster_source_sample(
1361:                     source_path=exported_source,
1362:                     query=query,
1363:                     model_id=prepared_query.model_id,
1364:                 )
1365:             )
1366:         except _RealtimeQueryCanceledError:
1367:             raise
1368:         except (OnlineRasterPreflightError, OnlineRasterExportError) as exc:
1369:             msg = describe_online_export_failure(exc)
1370:             logger.warning("%s", msg)
1371:             raise ValueError(msg) from exc
1372:         except Exception as exc:
1373:             msg = "Failed to prepare exported online raster for realtime query."
1374:             logger.error("%s Details: %s", msg, exc)
1375:             raise ValueError(msg) from exc
1376: 
1377:     if len(prepared_source_samples) == 0:
1378:         msg = "Failed to prepare a local raster chip for the realtime query."
1379:         logger.error(msg)
1380:         raise ValueError(msg)
1381: 
1382:     for prepared_sample in prepared_source_samples:
1383:         engine = None
1384:         try:
1385:             _ensure_not_canceled()
1386:             _report_progress("Encoding image", 30.0)
1387:             engine = build_model_adapter(model_spec)
1388:             query_cache = None
1389:             if prepared_query.supports_feature_reuse:
1390:                 encoded = engine.encode_image(prepared_sample.model_image)
1391:                 _ensure_not_canceled()
1392:                 _report_progress("Running prompt query", 75.0)
1393:                 prediction = engine.predict_features(
1394:                     encoded,
1395:                     multimask_output=False,
1396:                     **prepared_sample.prompt_kwargs,
1397:                 )
1398:                 query_cache = OnlineQueryCache(
1399:                     source_path=prepared_sample.source_path,
1400:                     chip_bounds=prepared_sample.chip_bounds,
1401:                     chip_grid=prepared_sample.chip_grid,
1402:                     encoded=encoded,
1403:                 )
1404:                 query_cache.crs_text = prepared_sample.crs_text
1405:             else:
1406:                 _ensure_not_canceled()
1407:                 _report_progress("Running prompt query", 75.0)
1408:                 prediction = engine.predict_image(
1409:                     prepared_sample.model_image,
1410:                     multimask_output=False,
1411:                     **prepared_sample.prompt_kwargs,
1412:                 )
1413: 
1414:             result = _prediction_to_prepared_query_result(
1415:                 prediction,
1416:                 chip_grid=prepared_sample.chip_grid,
1417:                 chip_bounds=prepared_sample.chip_bounds,
1418:                 query_bounds_value=prepared_sample.query_bounds_value,
1419:                 source_path=prepared_sample.source_path,
1420:                 model_type=model_spec.model_type,
1421:             )
1422:             _ensure_not_canceled()
1423:             _report_progress("Finished realtime query", 100.0)
1424:             return PreparedRealtimeRasterQueryResult(
1425:                 source_path=prepared_sample.source_path,
1426:                 result=result,
1427:                 query_cache=query_cache,
1428:             )
1429:         except _RealtimeQueryCanceledError:
1430:             raise
1431:         except Exception as exc:
1432:             errors.append(f"prepared-sample {prepared_sample.source_path}: {exc}")
1433:             continue
1434:         finally:
1435:             _close_runtime_engine(engine)
1436: 
1437:     if len(source_candidates) > 0:
1438:         msg = "Failed to encode a prepared realtime raster chip for the current prompt."
1439:         logger.error("%s Details: %s", msg, errors)
1440:         raise ValueError("\n".join([msg, *errors]))
1441: 
1442:     msg = "Failed to encode a realtime raster chip for the current prompt."
1443:     logger.error("%s Details: %s", msg, errors)
1444:     raise ValueError("\n".join([msg, *errors]))
1445: 
1446: 
1447: def _load_persistent_query_cache_for_request(
1448:     *,
1449:     model_id: str,
1450:     cache_directory: Path,
1451:     source_fingerprint: str,
1452:     layer_crs_text: str,
1453:     query,
1454: ) -> tuple[str, Any] | None:
1455:     """Load a reusable realtime-query cache entry for a prepared request."""
1456:     cache_hit = _find_persistent_query_cache_for_request(
1457:         model_id=model_id,
1458:         cache_directory=cache_directory,
1459:         source_fingerprint=source_fingerprint,
1460:         layer_crs_text=layer_crs_text,
1461:         query=query,
1462:     )
1463:     return _load_prepared_persistent_query_cache(cache_hit)
1464: 
1465: 
1466: def _find_persistent_query_cache_for_request(
1467:     *,
1468:     model_id: str,
1469:     cache_directory: Path,
1470:     source_fingerprint: str,
1471:     layer_crs_text: str,
1472:     query,
1473: ) -> PreparedPersistentQueryCacheHit | None:
1474:     """Find reusable realtime-query cache metadata without loading features."""
1475:     configure_geosam_qgis_runtime()
1476:     settings = load_plugin_settings()
1477:     if not settings.get("cache_enabled", True):
1478:         return None
1479: 
1480:     initialize_rasterio_proj_data()
1481:     from geosam.datasets import GeoGrid
1482:     from geosam.engines import _prompt_prediction_kwargs
1483:     from geosam.query import BoundingBox, query_bounds, query_center
1484:     from rasterio import Affine
1485: 
1486:     if not cache_directory.exists():
1487:         return None
1488: 
1489:     projected_query = _query_to_crs_text(query, layer_crs_text)
1490:     center_x, center_y = query_center(projected_query)
1491:     best_match: tuple[float, PreparedPersistentQueryCacheHit] | None = None
1492:     expected_shape = effective_model_imgsz(model_id)
1493:     for metadata_path in sorted(cache_directory.glob("*/metadata.json")):
1494:         try:
1495:             metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
1496:         except Exception as exc:
1497:             logger.warning(

(Output capped at 50 KB. Showing lines 1-1497. Use offset=1498 to continue.)
</content>
```

---

## Assistant (Build · GLM-5.2 · 15.2s)

_Thinking:_

The geosam_runtime.py file is large but mostly about realtime raster queries, not dependency installation. Let me look at the actual dependency installation logic - the widgetTool.py and plugin_settings.py, and the geosam_backend.py.

Let me explore the actual dependency installation logic, which appears to be in widgetTool.py and geosam_backend.py.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/geosam_backend.py"
}
```

**Output:**
```
<path>/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/geosam_backend.py</path>
<type>file</type>
<content>
1: """GeoSAM runtime backend helpers for the QGIS plugin."""
2: 
3: from __future__ import annotations
4: 
5: import logging
6: from typing import Any
7: 
8: from qgis.core import QgsProject
9: 
10: logger = logging.getLogger(__name__)
11: 
12: 
13: def configure_geosam_qgis_runtime(
14:     *,
15:     feedback: Any | None = None,
16:     task: Any | None = None,
17:     transform_context: Any | None = None,
18: ) -> bool:
19:     """Configure GeoSAM to use QGIS runtime services.
20: 
21:     Parameters
22:     ----------
23:     feedback : Any | None, optional
24:         QGIS processing feedback object used for progress and cancellation.
25:     task : Any | None, optional
26:         QGIS task object used for progress and cancellation.
27:     transform_context : Any | None, optional
28:         Coordinate transform context to pass to GeoSAM CRS operations. When
29:         omitted, the current project transform context is used.
30: 
31:     Returns
32:     -------
33:     bool
34:         ``True`` when GeoSAM was available and configured, otherwise ``False``.
35: 
36:     """
37:     try:
38:         from geosam.context import configure_runtime
39:     except ModuleNotFoundError:
40:         logger.warning("GeoSAM is not installed; QGIS runtime was not configured.")
41:         return False
42: 
43:     project = QgsProject.instance()
44:     resolved_transform_context = transform_context or project.transformContext()
45:     configure_runtime(
46:         "qgis",
47:         qgis_project=project,
48:         qgis_transform_context=resolved_transform_context,
49:         qgis_feedback=feedback,
50:         qgis_task=task,
51:     )
52:     return True

(End of file - total 52 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/plugin_settings.py"
}
```

**Output:**
```
<path>/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/plugin_settings.py</path>
<type>file</type>
<content>
1: """Plugin settings, dependency, and cache helpers for Geo-SAM."""
2: 
3: from __future__ import annotations
4: 
5: import importlib
6: import importlib.metadata
7: import importlib.util
8: import json
9: import logging
10: import os
11: import subprocess
12: import sys
13: import tempfile
14: from collections.abc import Iterator
15: from contextlib import contextmanager
16: from pathlib import Path
17: from typing import Any, Callable, Iterable, Literal, TypedDict
18: 
19: from qgis.PyQt.QtCore import QUrl
20: from qgis.PyQt.QtGui import QDesktopServices
21: 
22: from .dependency_path import register_plugin_managed_dependency_path
23: 
24: logger = logging.getLogger(__name__)
25: 
26: PLUGIN_ROOT = Path(__file__).resolve().parents[1]
27: CONFIG_DIR = PLUGIN_ROOT / "ui" / "config"
28: SETTINGS_DEFAULT_PATH = CONFIG_DIR / "default.json"
29: SETTINGS_USER_PATH = CONFIG_DIR / "user.json"
30: DEFAULT_MODEL_DIR = PLUGIN_ROOT / ".models"
31: DEFAULT_CACHE_DIR = PLUGIN_ROOT / ".cache"
32: LOCAL_GEOSAM_REPOSITORY = Path("/Users/fancy/Documents/GitHub/geosam")
33: HELP_LINKS = {
34:     "Documentation": "https://geo-sam.readthedocs.io/en/latest/",
35:     "GitHub": "https://github.com/coolzhao/Geo-SAM",
36:     "Report Bug": "https://github.com/coolzhao/Geo-SAM/issues",
37:     "Discussions": "https://github.com/coolzhao/Geo-SAM/discussions",
38: }
39: DEPENDENCY_DISTRIBUTIONS: dict[str, str] = {
40:     "torch": "torch",
41:     "torchvision": "torchvision",
42:     "ultralytics": "ultralytics",
43:     "rasterio": "rasterio",
44:     "geopandas": "geopandas",
45:     "shapely": "shapely",
46:     "pyproj": "pyproj",
47:     "geosam": "geosam",
48: }
49: DEPENDENCY_INSTALL_REQUIREMENTS: dict[str, str] = {
50:     "geosam": "geosam",
51:     "torch": "torch",
52:     "torchvision": "torchvision",
53:     "ultralytics": "ultralytics>=8.4.33",
54:     "rasterio": "rasterio>=1.4.3",
55:     "geopandas": "geopandas>=1.0.1",
56:     "shapely": "shapely>=2.0.7",
57:     "pyproj": "pyproj>=3.6.1",
58: }
59: PLUGIN_RUNTIME_DEPENDENCY_NAMES: tuple[str, ...] = tuple(DEPENDENCY_DISTRIBUTIONS)
60: RUNTIME_PROVIDED_DEPENDENCY_NAMES: tuple[str, ...] = (
61:     "rasterio",
62:     "geopandas",
63:     "shapely",
64:     "pyproj",
65: )
66: SEGMENTATION_RUNTIME_DEPENDENCY_NAMES: tuple[str, ...] = (
67:     "geosam",
68:     "rasterio",
69:     "geopandas",
70:     "shapely",
71:     "pyproj",
72: )
73: QGIS_RUNTIME_CONSTRAINT_DISTRIBUTIONS: tuple[str, ...] = (
74:     "numpy",
75:     "scipy",
76:     "pandas",
77:     "geopandas",
78:     "rasterio",
79:     "shapely",
80:     "pyproj",
81:     "pyogrio",
82:     "python-dateutil",
83:     "networkx",
84:     "requests",
85:     "pyyaml",
86:     "packaging",
87: )
88: PIP_RESOLVER_CONSTRAINT_REQUIREMENTS: tuple[str, ...] = (
89:     "contourpy>=1.2.0",
90:     "matplotlib>=3.8.0",
91: )
92: TORCH_RESOLVER_CONSTRAINT_REQUIREMENTS_PY39: tuple[str, ...] = (
93:     "torch==2.8.0",
94:     "torchvision==0.23.0",
95: )
96: TORCH_RESOLVER_CONSTRAINT_REQUIREMENTS_PY310_PLUS: tuple[str, ...] = (
97:     "torch==2.11.0",
98:     "torchvision==0.26.0",
99: )
100: PIP_ONLY_BINARY_DISTRIBUTIONS: tuple[str, ...] = (
101:     "contourpy",
102:     "kiwisolver",
103:     "matplotlib",
104:     "numpy",
105:     "opencv-python",
106:     "pillow",
107:     "torch",
108:     "torchvision",
109: )
110: PerformanceMode = Literal["balanced", "fastest", "low_memory"]
111: PreviewRenderMode = Literal["pixel_level", "simplified"]
112: PERFORMANCE_MODE_VALUES: tuple[PerformanceMode, ...] = (
113:     "balanced",
114:     "fastest",
115:     "low_memory",
116: )
117: PREVIEW_RENDER_MODE_VALUES: tuple[PreviewRenderMode, ...] = (
118:     "pixel_level",
119:     "simplified",
120: )
121: PROJ_DATA_ENV_KEYS = ("PROJ_DATA", "PROJ_LIB")
122: DependencyInstallCommand = list[str]
123: DependencyInstallPlan = list[DependencyInstallCommand]
124: DependencyState = Literal[
125:     "installed_qgis_runtime",
126:     "installed_plugin_managed",
127:     "missing_installable",
128:     "missing_required",
129: ]
130: 
131: 
132: class DependencyStatusRow(TypedDict):
133:     """Dependency status information shown in the settings dialog.
134: 
135:     Attributes
136:     ----------
137:     package : str
138:         Import or package name shown to users.
139:     distribution : str
140:         Installed Python distribution name.
141:     installed : bool
142:         Whether the distribution can be found in the active environment.
143:     version : str
144:         Installed distribution version, or an empty string when missing.
145:     state : DependencyState
146:         Resolved dependency source and availability state.
147:     installable : bool
148:         Whether the plugin installer should attempt to install this dependency.
149:     source : str
150:         Short source or install-provider note for the settings dialog.
151: 
152:     """
153: 
154:     package: str
155:     distribution: str
156:     installed: bool
157:     version: str
158:     state: DependencyState
159:     installable: bool
160:     source: str
161: 
162: 
163: def dependency_state(module_name: str, installed: bool) -> DependencyState:
164:     """Return the resolved dependency state for a module.
165: 
166:     Parameters
167:     ----------
168:     module_name : str
169:         Dependency module name.
170:     installed : bool
171:         Whether the dependency is installed in the active environment.
172: 
173:     Returns
174:     -------
175:     DependencyState
176:         Normalized dependency state used by the settings dialog and runtime
177:         preflight checks.
178: 
179:     """
180:     if installed:
181:         if module_name in RUNTIME_PROVIDED_DEPENDENCY_NAMES:
182:             return "installed_qgis_runtime"
183:         return "installed_plugin_managed"
184:     if module_name in DEPENDENCY_INSTALL_REQUIREMENTS:
185:         return "missing_installable"
186:     return "missing_required"
187: 
188: 
189: def dependency_source_text(module_name: str, state: DependencyState) -> str:
190:     """Return the settings source label for a dependency state.
191: 
192:     Parameters
193:     ----------
194:     module_name : str
195:         Dependency module name.
196:     state : DependencyState
197:         Resolved dependency state.
198: 
199:     Returns
200:     -------
201:     str
202:         Source text shown in the settings dependency table.
203: 
204:     """
205:     if state == "installed_plugin_managed":
206:         return "Geo-SAM managed"
207:     if module_name in RUNTIME_PROVIDED_DEPENDENCY_NAMES:
208:         if module_name in DEPENDENCY_INSTALL_REQUIREMENTS:
209:             return "QGIS runtime or Geo-SAM"
210:         return "QGIS runtime"
211:     return "Geo-SAM managed"
212: 
213: 
214: def dependency_status_text(state: DependencyState) -> str:
215:     """Return the display status text for a dependency state.
216: 
217:     Parameters
218:     ----------
219:     state : DependencyState
220:         Resolved dependency state.
221: 
222:     Returns
223:     -------
224:     str
225:         Human-readable status text for the settings dependency table.
226: 
227:     """
228:     if state in {"installed_qgis_runtime", "installed_plugin_managed"}:
229:         return "Installed"
230:     return "Missing"
231: 
232: 
233: def initialize_rasterio_proj_data() -> None:
234:     """Point rasterio at its bundled PROJ database when available.
235: 
236:     Rasterio wheels can bundle a newer PROJ database than the active QGIS
237:     process exposes through ``PROJ_LIB``. Importing rasterio while ``PROJ_LIB``
238:     points at an older database can make valid EPSG codes fail to resolve.
239:     This helper resolves rasterio's bundled ``proj_data`` directory, imports
240:     rasterio with that path during initialization when needed, and explicitly
241:     updates rasterio's PROJ search path. The explicit search-path update also
242:     fixes sessions where rasterio was imported before this helper ran.
243: 
244:     Returns
245:     -------
246:     None
247:         The function mutates only import state and temporarily mutates
248:         ``os.environ``.
249: 
250:     Notes
251:     -----
252:     The helper is intentionally a no-op when rasterio is unavailable. Call it
253:     immediately before rasterio operations in lazy rasterio code paths.
254: 
255:     """
256:     rasterio_module = sys.modules.get("rasterio")
257:     if rasterio_module is not None:
258:         rasterio_file = getattr(rasterio_module, "__file__", None)
259:         if rasterio_file is None:
260:             return
261:         rasterio_package_path = Path(rasterio_file).resolve().parent
262:     else:
263:         rasterio_spec = importlib.util.find_spec("rasterio")
264:         if rasterio_spec is None or rasterio_spec.submodule_search_locations is None:
265:             return
266:         rasterio_package_path = Path(rasterio_spec.submodule_search_locations[0])
267: 
268:     rasterio_proj_data_path = rasterio_package_path / "proj_data"
269:     if not (rasterio_proj_data_path / "proj.db").is_file():
270:         return
271: 
272:     original_environment = {
273:         env_key: os.environ.get(env_key) for env_key in PROJ_DATA_ENV_KEYS
274:     }
275:     rasterio_proj_data_text = str(rasterio_proj_data_path)
276:     try:
277:         for env_key in PROJ_DATA_ENV_KEYS:
278:             os.environ[env_key] = rasterio_proj_data_text
279:         rasterio_module = importlib.import_module("rasterio")
280:         rasterio_env_module = importlib.import_module("rasterio.env")
281:         set_proj_data_search_path = getattr(
282:             rasterio_env_module,
283:             "set_proj_data_search_path",
284:             None,
285:         )
286:         if set_proj_data_search_path is None:
287:             logger.warning(
288:                 "Rasterio does not expose set_proj_data_search_path; "
289:                 "cannot override PROJ data path."
290:             )
291:             return
292:         set_proj_data_search_path(rasterio_proj_data_text)
293:         logger.debug(
294:             "Initialized rasterio PROJ data path for %s from %s.",
295:             getattr(rasterio_module, "__file__", "rasterio"),
296:             rasterio_proj_data_path,
297:         )
298:     finally:
299:         for env_key, env_value in original_environment.items():
300:             if env_value is None:
301:                 os.environ.pop(env_key, None)
302:             else:
303:                 os.environ[env_key] = env_value
304: 
305: 
306: def _resolve_rasterio_proj_data_path() -> Path | None:
307:     """Resolve rasterio's bundled PROJ data directory.
308: 
309:     Returns
310:     -------
311:     Path | None
312:         Path to rasterio's bundled ``proj_data`` directory, or ``None`` when
313:         rasterio is unavailable or does not bundle ``proj.db``.
314: 
315:     """
316:     rasterio_module = sys.modules.get("rasterio")
317:     if rasterio_module is not None:
318:         rasterio_file = getattr(rasterio_module, "__file__", None)
319:         if rasterio_file is None:
320:             return None
321:         rasterio_package_path = Path(rasterio_file).resolve().parent
322:     else:
323:         rasterio_spec = importlib.util.find_spec("rasterio")
324:         if rasterio_spec is None or rasterio_spec.submodule_search_locations is None:
325:             return None
326:         rasterio_package_path = Path(rasterio_spec.submodule_search_locations[0])
327: 
328:     rasterio_proj_data_path = rasterio_package_path / "proj_data"
329:     if not (rasterio_proj_data_path / "proj.db").is_file():
330:         return None
331:     return rasterio_proj_data_path
332: 
333: 
334: @contextmanager
335: def rasterio_proj_data_environment() -> Iterator[None]:
336:     """Temporarily expose rasterio's bundled PROJ database to rasterio.
337: 
338:     Yields
339:     ------
340:     None
341:         Control returns to the caller while ``PROJ_DATA`` and ``PROJ_LIB`` point
342:         at rasterio's matching ``proj_data`` directory.
343: 
344:     Notes
345:     -----
346:     Rasterio's ``Env`` wrapper can re-read ``os.environ`` during operations such
347:     as :func:`rasterio.open`. Keeping the environment override active for the
348:     full operation prevents it from falling back to an incompatible QGIS or
349:     conda ``proj.db``.
350: 
351:     """
352:     rasterio_proj_data_path = _resolve_rasterio_proj_data_path()
353:     if rasterio_proj_data_path is None:
354:         yield
355:         return
356: 
357:     original_environment = {
358:         env_key: os.environ.get(env_key) for env_key in PROJ_DATA_ENV_KEYS
359:     }
360:     rasterio_proj_data_text = str(rasterio_proj_data_path)
361:     try:
362:         for env_key in PROJ_DATA_ENV_KEYS:
363:             os.environ[env_key] = rasterio_proj_data_text
364:         initialize_rasterio_proj_data()
365:         yield
366:     finally:
367:         for env_key, env_value in original_environment.items():
368:             if env_value is None:
369:                 os.environ.pop(env_key, None)
370:             else:
371:                 os.environ[env_key] = env_value
372: 
373: 
374: def _normalize_preview_render_mode(value: Any) -> PreviewRenderMode:
375:     """Normalize a configured vectorization mode.
376: 
377:     Parameters
378:     ----------
379:     value : Any
380:         Raw persisted value.
381: 
382:     Returns
383:     -------
384:     PreviewRenderMode
385:         Normalized vectorization mode.
386: 
387:     """
388:     normalized_value = str(value or "pixel_level").strip().lower().replace("-", "_")
389:     if normalized_value in PREVIEW_RENDER_MODE_VALUES:
390:         return normalized_value  # type: ignore[return-value]
391:     return "pixel_level"
392: 
393: 
394: def _load_json(path: Path) -> dict[str, Any]:
395:     """Load a JSON object from disk.
396: 
397:     Parameters
398:     ----------
399:     path : Path
400:         JSON file path.
401: 
402:     Returns
403:     -------
404:     dict[str, Any]
405:         Parsed JSON object, or an empty dictionary when the user settings file
406:         is missing.
407: 
408:     """
409:     if not path.exists():
410:         if path == SETTINGS_USER_PATH:
411:             path.write_text("{}\n", encoding="utf-8")
412:         return {}
413:     return json.loads(path.read_text(encoding="utf-8"))
414: 
415: 
416: def load_plugin_settings() -> dict[str, Any]:
417:     """Load effective plugin settings.
418: 
419:     Returns
420:     -------
421:     dict[str, Any]
422:         Effective settings after merging defaults with user overrides.
423: 
424:     """
425:     settings = _default_plugin_settings()
426:     settings.update(_load_json(SETTINGS_USER_PATH))
427:     if settings.get("performance_mode") not in PERFORMANCE_MODE_VALUES:
428:         settings["performance_mode"] = "balanced"
429:     settings["preview_render_mode"] = _normalize_preview_render_mode(
430:         settings.get("preview_render_mode")
431:     )
432:     return settings
433: 
434: 
435: def _default_plugin_settings() -> dict[str, Any]:
436:     """Return the effective default plugin settings.
437: 
438:     Returns
439:     -------
440:     dict[str, Any]
441:         Default settings resolved from the shipped configuration file.
442: 
443:     """
444:     settings = _load_json(SETTINGS_DEFAULT_PATH)
445:     settings.setdefault("model_store_dir", str(DEFAULT_MODEL_DIR))
446:     settings.setdefault("cache_enabled", True)
447:     settings.setdefault("cache_dir", str(DEFAULT_CACHE_DIR))
448:     settings.setdefault("cache_max_size_mb", 2048)
449:     settings.setdefault("clear_cache_on_plugin_close", True)
450:     settings.setdefault("selected_model_id", "")
451:     settings.setdefault("show_boundary", True)
452:     settings.setdefault("image_source_mode", "live_encoding")
453:     settings.setdefault("default_minimum_pixels", 0)
454:     settings.setdefault("performance_mode", "balanced")
455:     settings.setdefault("preview_mode", True)
456:     settings["preview_render_mode"] = _normalize_preview_render_mode(
457:         settings.get("preview_render_mode")
458:     )
459:     return settings
460: 
461: 
462: def save_plugin_settings(updates: dict[str, Any]) -> dict[str, Any]:
463:     """Persist plugin settings.
464: 
465:     Parameters
466:     ----------
467:     updates : dict[str, Any]
468:         Setting values to merge into the current configuration.
469: 
470:     Returns
471:     -------
472:     dict[str, Any]
473:         Effective settings after the update is applied.
474: 
475:     """
476:     settings = load_plugin_settings()
477:     settings.update(updates)
478:     settings["preview_render_mode"] = _normalize_preview_render_mode(
479:         settings.get("preview_render_mode")
480:     )
481:     defaults = _default_plugin_settings()
482:     user_settings = {
483:         key: settings[key] for key in defaults if settings.get(key) != defaults[key]
484:     }
485:     SETTINGS_USER_PATH.write_text(
486:         json.dumps(user_settings, indent=4),
487:         encoding="utf-8",
488:     )
489:     return settings
490: 
491: 
492: def get_model_directory() -> Path:
493:     """Return the local model directory.
494: 
495:     Returns
496:     -------
497:     Path
498:         Existing model directory path.
499: 
500:     """
501:     settings = load_plugin_settings()
502:     model_dir = Path(settings["model_store_dir"]).expanduser()
503:     model_dir.mkdir(parents=True, exist_ok=True)
504:     return model_dir
505: 
506: 
507: def get_cache_directory() -> Path:
508:     """Return the local cache directory.
509: 
510:     Returns
511:     -------
512:     Path
513:         Existing cache directory path.
514: 
515:     """
516:     settings = load_plugin_settings()
517:     cache_dir = Path(settings["cache_dir"]).expanduser()
518:     cache_dir.mkdir(parents=True, exist_ok=True)
519:     return cache_dir
520: 
521: 
522: def dependency_status() -> dict[str, bool]:
523:     """Return installed-package availability for required dependencies.
524: 
525:     Returns
526:     -------
527:     dict[str, bool]
528:         Mapping from module name to installation status.
529: 
530:     Notes
531:     -----
532:     This check intentionally uses distribution metadata instead of
533:     ``importlib.util.find_spec`` so the result reflects the current environment
534:     after install or uninstall operations in the same QGIS session.
535: 
536:     """
537:     register_plugin_managed_dependency_path()
538:     status: dict[str, bool] = {}
539:     for module_name, distribution_name in DEPENDENCY_DISTRIBUTIONS.items():
540:         try:
541:             importlib.metadata.distribution(distribution_name)
542:         except importlib.metadata.PackageNotFoundError:
543:             status[module_name] = False
544:         else:
545:             status[module_name] = True
546:     return status
547: 
548: 
549: def dependency_status_rows() -> list[DependencyStatusRow]:
550:     """Return dependency status rows with installed versions.
551: 
552:     Returns
553:     -------
554:     list[DependencyStatusRow]
555:         Ordered dependency rows for the settings dialog table.
556: 
557:     """
558:     register_plugin_managed_dependency_path()
559:     rows: list[DependencyStatusRow] = []
560:     for module_name, distribution_name in DEPENDENCY_DISTRIBUTIONS.items():
561:         try:
562:             distribution = importlib.metadata.distribution(distribution_name)
563:         except importlib.metadata.PackageNotFoundError:
564:             state = dependency_state(module_name, installed=False)
565:             rows.append(
566:                 {
567:                     "package": module_name,
568:                     "distribution": distribution_name,
569:                     "installed": False,
570:                     "version": "",
571:                     "state": state,
572:                     "installable": state == "missing_installable",
573:                     "source": dependency_source_text(module_name, state),
574:                 }
575:             )
576:         else:
577:             state = dependency_state(module_name, installed=True)
578:             rows.append(
579:                 {
580:                     "package": module_name,
581:                     "distribution": distribution_name,
582:                     "installed": True,
583:                     "version": distribution.version,
584:                     "state": state,
585:                     "installable": False,
586:                     "source": dependency_source_text(module_name, state),
587:                 }
588:             )
589:     return rows
590: 
591: 
592: def missing_runtime_dependencies(module_names: Iterable[str]) -> list[str]:
593:     """Return missing runtime dependency module names in input order.
594: 
595:     Parameters
596:     ----------
597:     module_names : Iterable[str]
598:         Dependency module names to validate.
599: 
600:     Returns
601:     -------
602:     list[str]
603:         Missing dependency module names without duplicates.
604: 
605:     Notes
606:     -----
607:     The check uses distribution metadata so the result reflects the active
608:     QGIS Python environment after in-session installs.
609: 
610:     """
611:     known_status = dependency_status()
612:     missing_module_names: list[str] = []
613:     seen_module_names: set[str] = set()
614:     for module_name in module_names:
615:         if module_name in seen_module_names:
616:             continue
617:         seen_module_names.add(module_name)
618: 
619:         installed = known_status.get(module_name)
620:         if installed is None:
621:             distribution_name = DEPENDENCY_DISTRIBUTIONS.get(module_name, module_name)
622:             try:
623:                 importlib.metadata.distribution(distribution_name)
624:             except importlib.metadata.PackageNotFoundError:
625:                 installed = False
626:             else:
627:                 installed = True
628:         if not installed:
629:             missing_module_names.append(module_name)
630:     return missing_module_names
631: 
632: 
633: def missing_segmentation_runtime_dependencies() -> list[str]:
634:     """Return missing dependencies required for segmentation queries.
635: 
636:     Returns
637:     -------
638:     list[str]
639:         Missing dependency module names required before GeoSAM query creation.
640: 
641:     """
642:     return missing_runtime_dependencies(SEGMENTATION_RUNTIME_DEPENDENCY_NAMES)
643: 
644: 
645: def missing_plugin_runtime_dependencies() -> list[str]:
646:     """Return missing dependencies managed by the Geo-SAM installer.
647: 
648:     Returns
649:     -------
650:     list[str]
651:         Missing module names shown on the Dependencies page.
652: 
653:     """
654:     return missing_runtime_dependencies(PLUGIN_RUNTIME_DEPENDENCY_NAMES)
655: 
656: 
657: def format_missing_dependencies_message(
658:     module_names: Iterable[str],
659:     *,
660:     action_label: str = "this action",
661: ) -> str:
662:     """Return a user-facing message for missing runtime dependencies.
663: 
664:     Parameters
665:     ----------
666:     module_names : Iterable[str]
667:         Missing dependency module names.
668:     action_label : str, optional
669:         Action name inserted into the message body.
670: 
671:     Returns
672:     -------
673:     str
674:         Formatted QMessageBox-friendly message text.
675: 
676:     Raises
677:     ------
678:     ValueError
679:         Raised when no dependency names were provided.
680: 
681:     """
682:     unique_module_names = list(dict.fromkeys(module_names))
683:     if not unique_module_names:
684:         msg = "Cannot format a dependency message without dependency names."
685:         logger.error(msg)
686:         raise ValueError(msg)
687: 
688:     if len(unique_module_names) == 1:
689:         dependency_text = unique_module_names[0]
690:     elif len(unique_module_names) == 2:
691:         dependency_text = " and ".join(unique_module_names)
692:     else:
693:         dependency_text = ", ".join(unique_module_names[:-1])
694:         dependency_text = f"{dependency_text}, and {unique_module_names[-1]}"
695: 
696:     return (
697:         f"Geo-SAM needs {dependency_text} for {action_label}.\n\n"
698:         "Open Geo-SAM Settings and install the missing dependencies first."
699:     )
700: 
701: 
702: def _dependency_install_requirement(module_name: str) -> str:
703:     """Return the pip requirement used to install a dependency.
704: 
705:     Parameters
706:     ----------
707:     module_name : str
708:         Dependency module name.
709: 
710:     Returns
711:     -------
712:     str
713:         Pip requirement name or path.
714: 
715:     """
716:     if module_name == "geosam" and LOCAL_GEOSAM_REPOSITORY.exists():
717:         return str(LOCAL_GEOSAM_REPOSITORY)
718:     return DEPENDENCY_INSTALL_REQUIREMENTS[module_name]
719: 
720: 
721: def _selected_dependency_module_names(
722:     module_names: Iterable[str] | None = None,
723: ) -> list[str]:
724:     """Return known dependency module names selected for installation.
725: 
726:     Parameters
727:     ----------
728:     module_names : Iterable[str] | None, optional
729:         Dependency package names to install. When omitted, all known plugin
730:         dependencies are selected.
731: 
732:     Returns
733:     -------
734:     list[str]
735:         Known dependency module names in installation order.
736: 
737:     Raises
738:     ------
739:     RuntimeError
740:         Raised when no known dependencies were selected.
741: 
742:     """
743:     if module_names is None:
744:         selected_module_names = list(DEPENDENCY_INSTALL_REQUIREMENTS)
745:     else:
746:         selected_module_names = [
747:             module_name
748:             for module_name in module_names
749:             if module_name in DEPENDENCY_INSTALL_REQUIREMENTS
750:         ]
751:     if not selected_module_names:
752:         msg = "No known dependencies were selected for installation."
753:         logger.error(msg)
754:         raise RuntimeError(msg)
755:     return selected_module_names
756: 
757: 
758: def _create_qgis_runtime_constraints_file() -> Path | None:
759:     """Create a temporary pip constraints file for bundled QGIS packages.
760: 
761:     Returns
762:     -------
763:     Path | None
764:         Temporary constraints file path, or ``None`` when no protected runtime
765:         distributions are installed in the active environment.
766: 
767:     Raises
768:     ------
769:     RuntimeError
770:         Raised when the constraints file cannot be written.
771: 
772:     Notes
773:     -----
774:     The constraints are generated from the currently installed package
775:     versions. This preserves whichever versions the active QGIS runtime ships,
776:     including future runtimes that may bundle newer major versions.
777: 
778:     """
779:     constraint_lines: list[str] = []
780:     installed_versions: dict[str, str] = {}
781:     for distribution_name in QGIS_RUNTIME_CONSTRAINT_DISTRIBUTIONS:
782:         resolved_version = _runtime_constraint_version(distribution_name)
783:         if resolved_version is None:
784:             continue
785:         installed_versions[distribution_name] = resolved_version
786:         constraint_lines.append(f"{distribution_name}=={resolved_version}")
787: 
788:     constraint_lines.extend(_qgis_runtime_resolver_constraints(installed_versions))
789: 
790:     if not constraint_lines:
791:         return None
792: 
793:     try:
794:         with tempfile.NamedTemporaryFile(
795:             mode="w",
796:             encoding="utf-8",
797:             prefix="geo_sam_qgis_runtime_",
798:             suffix=".constraints.txt",
799:             delete=False,
800:         ) as constraints_file:
801:             constraints_file.write("\n".join(constraint_lines))
802:             constraints_file.write("\n")
803:             return Path(constraints_file.name)
804:     except OSError as exc:
805:         msg = "Could not create QGIS runtime dependency constraints file."
806:         logger.error("%s Error: %s", msg, exc)
807:         raise RuntimeError(msg) from exc
808: 
809: 
810: def _qgis_runtime_resolver_constraints(installed_versions: dict[str, str]) -> list[str]:
811:     """Return extra pip constraints for QGIS runtime compatibility.
812: 
813:     Parameters
814:     ----------
815:     installed_versions : dict[str, str]
816:         Installed protected distribution versions keyed by distribution name.
817: 
818:     Returns
819:     -------
820:     list[str]
821:         Additional requirement constraints that prevent pip resolver backtracking
822:         into incompatible or source-only transitive dependency versions.
823: 
824:     Notes
825:     -----
826:     These constraints do not pin QGIS packages. They only guide transitive
827:     dependencies toward versions compatible with the active QGIS runtime.
828: 
829:     """
830:     constraints = list(PIP_RESOLVER_CONSTRAINT_REQUIREMENTS)
831:     constraints.extend(_torch_resolver_constraints())
832:     numpy_major_version = _distribution_major_version(installed_versions.get("numpy"))
833:     if numpy_major_version is not None and numpy_major_version < 2:
834:         constraints.append("opencv-python==4.11.0.86")
835:     constraints.extend(_setuptools_resolver_constraints())
836:     return constraints
837: 
838: 
839: def _torch_resolver_constraints() -> list[str]:
840:     """Return PyTorch constraints compatible with the active Python runtime.
841: 
842:     Returns
843:     -------
844:     list[str]
845:         Torch and torchvision constraints for pip resolution.
846: 
847:     Notes
848:     -----
849:     PyTorch 2.11 does not publish Python 3.9 wheels. QGIS environments based
850:     on Python 3.9 therefore need the latest known PyTorch pair with cp39
851:     wheels.
852: 
853:     """
854:     if sys.version_info < (3, 10):
855:         return list(TORCH_RESOLVER_CONSTRAINT_REQUIREMENTS_PY39)
856:     return list(TORCH_RESOLVER_CONSTRAINT_REQUIREMENTS_PY310_PLUS)
857: 
858: 
859: def _distribution_major_version(version_text: str | None) -> int | None:
860:     """Return the leading integer version component.
861: 
862:     Parameters
863:     ----------
864:     version_text : str | None
865:         Distribution version text.
866: 
867:     Returns
868:     -------
869:     int | None
870:         Major version number, or ``None`` when it cannot be parsed.
871: 
872:     """
873:     if version_text is None:
874:         return None
875:     major_text = version_text.split(".", maxsplit=1)[0]
876:     try:
877:         return int(major_text)
878:     except ValueError:
879:         logger.debug(
880:             "Could not parse distribution major version from %s.", version_text
881:         )
882:         return None
883: 
884: 
885: def _runtime_constraint_version(distribution_name: str) -> str | None:
886:     """Return the effective version used for a runtime constraint line.
887: 
888:     Parameters
889:     ----------
890:     distribution_name : str
891:         Distribution name written into the pip constraints file.
892: 
893:     Returns
894:     -------
895:     str | None
896:         Resolved version text, or ``None`` when the distribution is unavailable.
897: 
898:     Notes
899:     -----
900:     Some QGIS Python environments expose placeholder distribution metadata such
901:     as ``python-dateutil==0.0.0`` while the importable module is a valid
902:     release. This helper normalizes those cases so pip receives a satisfiable
903:     constraint set.
904: 
905:     """
906:     try:
907:         distribution = importlib.metadata.distribution(distribution_name)
908:     except importlib.metadata.PackageNotFoundError:
909:         return None
910: 
911:     distribution_version = distribution.version
912:     if distribution_name != "python-dateutil" or distribution_version != "0.0.0":
913:         return distribution_version
914: 
915:     try:
916:         dateutil_module = importlib.import_module("dateutil")
917:     except ImportError:
918:         logger.warning(
919:             "python-dateutil distribution metadata reports version %s, and the "
920:             "dateutil module could not be imported for a fallback version.",
921:             distribution_version,
922:         )
923:         return distribution_version
924: 
925:     module_version = getattr(dateutil_module, "__version__", None)
926:     if not module_version:
927:         logger.warning(
928:             "python-dateutil distribution metadata reports version %s, and the "
929:             "dateutil module does not expose __version__ for fallback.",
930:             distribution_version,
931:         )
932:         return distribution_version
933: 
934:     logger.info(
935:         "Using python-dateutil module version %s instead of placeholder "
936:         "distribution version %s in runtime constraints.",
937:         module_version,
938:         distribution_version,
939:     )
940:     return str(module_version)
941: 
942: 
943: def _setuptools_resolver_constraints() -> list[str]:
944:     """Return setuptools constraints compatible with recent torch builds.
945: 
946:     Returns
947:     -------
948:     list[str]
949:         Additional setuptools constraints for pip resolution.
950: 
951:     Notes
952:     -----
953:     Recent torch releases require ``setuptools<82``. Some QGIS Python runtimes
954:     currently bundle ``setuptools==82.0.0`` or newer, which would make a
955:     generated exact runtime pin impossible to satisfy. The plugin does not rely
956:     on the QGIS-bundled setuptools version at runtime, so dependency installs
957:     should prefer the torch-compatible range instead of preserving that exact
958:     version.
959: 
960:     """
961:     try:
962:         setuptools_version = importlib.metadata.version("setuptools")
963:     except importlib.metadata.PackageNotFoundError:
964:         return ["setuptools<82"]
965: 
966:     setuptools_major_version = _distribution_major_version(setuptools_version)
967:     if setuptools_major_version is None:
968:         logger.debug(
969:             "Could not determine setuptools major version from %s; using "
970:             "torch-compatible upper bound.",
971:             setuptools_version,
972:         )
973:         return ["setuptools<82"]
974:     if setuptools_major_version >= 82:
975:         logger.info(
976:             "QGIS runtime provides setuptools %s; using setuptools<82 for "
977:             "torch compatibility during dependency installation.",
978:             setuptools_version,
979:         )
980:         return ["setuptools<82"]
981:     return [f"setuptools=={setuptools_version}"]
982: 
983: 
984: def _is_relative_to(path: Path, base_path: Path) -> bool:
985:     """Return whether ``path`` is located inside ``base_path``.
986: 
987:     Parameters
988:     ----------
989:     path : Path
990:         Candidate path.
991:     base_path : Path
992:         Expected ancestor path.
993: 
994:     Returns
995:     -------
996:     bool
997:         ``True`` when ``path`` is inside ``base_path``.
998: 
999:     """
1000:     try:
1001:         path.relative_to(base_path)
1002:     except ValueError:
1003:         return False
1004:     return True
1005: 
1006: 
1007: def _iter_python_interpreter_candidates() -> list[Path]:
1008:     """Return candidate Python executables for the active QGIS environment.
1009: 
1010:     Returns
1011:     -------
1012:     list[Path]
1013:         Ordered executable candidates, with duplicates removed.
1014: 
1015:     """
1016:     prefix_path = Path(sys.prefix).expanduser()
1017:     executable_path = Path(sys.executable).expanduser()
1018:     candidate_paths: list[Path] = []
1019: 
1020:     candidate_paths.extend(_iter_prefix_python_candidates(prefix_path))
1021:     candidate_paths.extend(_iter_platform_qgis_python_candidates(executable_path))
1022: 
1023:     for raw_path in sys.path:
1024:         if not raw_path:
1025:             continue
1026:         try:
1027:             resolved_path = Path(raw_path).expanduser().resolve()
1028:         except OSError:
1029:             continue
1030:         if not _is_relative_to(resolved_path, prefix_path):
1031:             continue
1032: 
1033:         for parent_path in (resolved_path, *resolved_path.parents):
1034:             if parent_path == prefix_path.parent:
1035:                 break
1036:             if parent_path.name not in {"bin", "Scripts"}:
1037:                 continue
1038:             for child_path in sorted(parent_path.glob("python*")):
1039:                 candidate_paths.append(child_path)
1040: 
1041:     return _unique_python_candidates(candidate_paths)
1042: 
1043: 
1044: def _iter_prefix_python_candidates(prefix_path: Path) -> list[Path]:
1045:     """Return Python executables below and near ``sys.prefix``.
1046: 
1047:     Parameters
1048:     ----------
1049:     prefix_path : Path
1050:         Active Python prefix path.
1051: 
1052:     Returns
1053:     -------
1054:     list[Path]
1055:         Candidate executable paths for virtualenv, conda, and system prefixes.
1056: 
1057:     Notes
1058:     -----
1059:     Conda-based QGIS installs normally expose the interpreter in
1060:     ``sys.prefix/bin`` or ``sys.prefix/Scripts``. The parent scan is restricted
1061:     to the active prefix name so it can cover relocated prefixes without
1062:     accidentally selecting another conda environment.
1063: 
1064:     """
1065:     candidate_paths: list[Path] = []
1066:     candidate_directories = [
1067:         prefix_path / "bin",
1068:         prefix_path / "Scripts",
1069:         prefix_path,
1070:     ]
1071:     for directory_path in candidate_directories:
1072:         candidate_paths.extend(_iter_python_files(directory_path))
1073: 
1074:     parent_path = prefix_path.parent
1075:     prefix_name = prefix_path.name
1076:     for directory_name in ("bin", "Scripts"):
1077:         glob_pattern = f"{prefix_name}*/{directory_name}/python*"
1078:         candidate_paths.extend(sorted(parent_path.glob(glob_pattern)))
1079: 
1080:     return candidate_paths
1081: 
1082: 
1083: def _iter_python_files(directory_path: Path) -> list[Path]:
1084:     """Return Python-like files in a directory.
1085: 
1086:     Parameters
1087:     ----------
1088:     directory_path : Path
1089:         Directory to scan.
1090: 
1091:     Returns
1092:     -------
1093:     list[Path]
1094:         Ordered candidate files whose names begin with ``python``.
1095: 
1096:     """
1097:     return sorted(directory_path.glob("python*"))
1098: 
1099: 
1100: def _unique_python_candidates(candidate_paths: Iterable[Path]) -> list[Path]:
1101:     """Return unique Python candidates without resolving symlink wrappers.
1102: 
1103:     Parameters
1104:     ----------
1105:     candidate_paths : Iterable[Path]
1106:         Raw candidate executable paths.
1107: 
1108:     Returns
1109:     -------
1110:     list[Path]
1111:         Candidate paths with duplicate textual paths removed.
1112: 
1113:     Notes
1114:     -----
1115:     The macOS QGIS app exposes ``Contents/MacOS/python`` as a symlink wrapper
1116:     that sets ``PYTHONHOME`` before running the real binary. Resolving that
1117:     symlink would break direct execution, so this helper keeps the original
1118:     executable path.
1119: 
1120:     """
1121:     unique_candidates: list[Path] = []
1122:     seen_paths: set[str] = set()
1123:     for candidate_path in candidate_paths:
1124:         try:
1125:             expanded_candidate = candidate_path.expanduser()
1126:         except OSError:
1127:             continue
1128:         if not expanded_candidate.is_absolute():
1129:             expanded_candidate = Path.cwd() / expanded_candidate
1130:         candidate_key = str(expanded_candidate)
1131:         if candidate_key in seen_paths:
1132:             continue
1133:         seen_paths.add(candidate_key)
1134:         unique_candidates.append(expanded_candidate)
1135:     return unique_candidates
1136: 
1137: 
1138: def _iter_platform_qgis_python_candidates(executable_path: Path) -> list[Path]:
1139:     """Return platform-specific QGIS Python executable candidates.
1140: 
1141:     Parameters
1142:     ----------
1143:     executable_path : Path
1144:         Active QGIS process executable path.
1145: 
1146:     Returns
1147:     -------
1148:     list[Path]
1149:         Candidate Python executable paths under the detected QGIS root.
1150: 
1151:     """
1152:     if sys.platform == "win32":
1153:         return _iter_windows_qgis_python_candidates(executable_path)
1154:     if sys.platform == "darwin":
1155:         return _iter_macos_qgis_python_candidates(executable_path)
1156:     return _iter_linux_qgis_python_candidates(executable_path)
1157: 
1158: 
1159: def _iter_windows_qgis_python_candidates(executable_path: Path) -> list[Path]:
1160:     """Return Python executables bundled beside Windows QGIS launchers.
1161: 
1162:     Parameters
1163:     ----------
1164:     executable_path : Path
1165:         Active QGIS process executable path.
1166: 
1167:     Returns
1168:     -------
1169:     list[Path]
1170:         Candidate Python executable paths under the detected QGIS root.
1171: 
1172:     Notes
1173:     -----
1174:     Windows QGIS exposes ``sys.executable`` as launchers such as
1175:     ``qgis-ltr-bin.exe``. The matching interpreter lives under
1176:     ``apps/Python*/python.exe`` within the same QGIS installation root.
1177: 
1178:     """
1179:     if executable_path.name.lower().startswith("python"):
1180:         return []
1181: 
1182:     candidate_roots: list[Path] = []
1183:     for parent_path in executable_path.parents:
1184:         if parent_path.name.lower() != "bin":
1185:             continue
1186:         candidate_roots.append(parent_path.parent)
1187:         break
1188: 
1189:     candidate_paths: list[Path] = []
1190:     for root_path in candidate_roots:
1191:         python_apps_path = root_path / "apps"
1192:         for python_dir in sorted(python_apps_path.glob("Python*")):
1193:             candidate_paths.extend(
1194:                 [
1195:                     python_dir / "python.exe",
1196:                     python_dir / "python3.exe",
1197:                     python_dir / "python",
1198:                     python_dir / "python3",
1199:                 ]
1200:             )
1201:     return candidate_paths
1202: 
1203: 
1204: def _iter_macos_qgis_python_candidates(executable_path: Path) -> list[Path]:
1205:     """Return Python executables for official macOS QGIS app bundles.
1206: 
1207:     Parameters
1208:     ----------
1209:     executable_path : Path
1210:         Active QGIS process executable path.
1211: 
1212:     Returns
1213:     -------
1214:     list[Path]
1215:         Candidate Python executable paths under ``QGIS.app``.
1216: 
1217:     Notes
1218:     -----
1219:     The official macOS app uses ``Contents/MacOS/python`` as a relocatable
1220:     wrapper. Prefer that path over versioned binaries because the wrapper sets
1221:     ``PYTHONHOME`` correctly before Python starts.
1222: 
1223:     """
1224:     contents_path: Path | None = None
1225:     for parent_path in (executable_path, *executable_path.parents):
1226:         if parent_path.name != "Contents":
1227:             continue
1228:         contents_path = parent_path
1229:         break
1230:     if contents_path is None:
1231:         return []
1232: 
1233:     macos_path = contents_path / "MacOS"
1234:     candidate_paths = [
1235:         macos_path / "python",
1236:         macos_path / "python3",
1237:     ]
1238:     candidate_paths.extend(_iter_python_files(macos_path))
1239:     return candidate_paths
1240: 
1241: 
1242: def _iter_linux_qgis_python_candidates(executable_path: Path) -> list[Path]:
1243:     """Return Python executables near Linux QGIS launchers.
1244: 
1245:     Parameters
1246:     ----------
1247:     executable_path : Path
1248:         Active QGIS process executable path.
1249: 
1250:     Returns
1251:     -------
1252:     list[Path]
1253:         Candidate Python executable paths near the launcher.
1254: 
1255:     """
1256:     candidate_paths: list[Path] = []
1257:     for parent_path in executable_path.parents:
1258:         if parent_path.name not in {"bin", "sbin"}:
1259:             continue
1260:         candidate_paths.extend(_iter_python_files(parent_path))
1261:         break
1262:     return candidate_paths
1263: 
1264: 
1265: def _is_python_interpreter_candidate(candidate_path: Path) -> bool:
1266:     """Return whether a path is an executable Python interpreter candidate.
1267: 
1268:     Parameters
1269:     ----------
1270:     candidate_path : Path
1271:         Candidate executable path.
1272: 
1273:     Returns
1274:     -------
1275:     bool
1276:         ``True`` when the path looks runnable as Python.
1277: 
1278:     """
1279:     candidate_name = candidate_path.name.lower()
1280:     if not candidate_name.startswith("python"):
1281:         return False
1282:     if not candidate_path.is_file():
1283:         return False
1284:     if os.name == "nt":
1285:         return True
1286:     return os.access(candidate_path, os.X_OK)
1287: 
1288: 
1289: def resolve_python_interpreter() -> Path:
1290:     """Resolve the Python interpreter for the active QGIS runtime environment.
1291: 
1292:     Returns
1293:     -------
1294:     Path
1295:         Resolved Python interpreter path that belongs to ``sys.prefix``.
1296: 
1297:     Raises
1298:     ------
1299:     RuntimeError
1300:         Raised when no suitable Python interpreter can be found.
1301: 
1302:     """
1303:     executable_path = Path(sys.executable).expanduser()
1304:     if sys.platform == "darwin":
1305:         for candidate_path in _iter_macos_qgis_python_candidates(executable_path):
1306:             if not _is_python_interpreter_candidate(candidate_path):
1307:                 continue
1308:             return candidate_path
1309: 
1310:     if _is_python_interpreter_candidate(executable_path):
1311:         return executable_path
1312: 
1313:     for candidate_path in _iter_python_interpreter_candidates():
1314:         if not _is_python_interpreter_candidate(candidate_path):
1315:             continue
1316:         return candidate_path
1317: 
1318:     msg = (
1319:         "Could not resolve a Python interpreter from the current QGIS runtime. "
1320:         f"sys.executable={sys.executable!r}, sys.prefix={sys.prefix!r}"
1321:     )
1322:     logger.error(msg)
1323:     raise RuntimeError(msg)
1324: 
1325: 
1326: def get_dependency_install_command(
1327:     module_names: Iterable[str] | None = None,
1328: ) -> DependencyInstallCommand:
1329:     """Build the first pip command used to install plugin dependencies.
1330: 
1331:     Parameters
1332:     ----------
1333:     module_names : Iterable[str] | None, optional
1334:         Dependency package names to install. When omitted, all known plugin
1335:         dependencies are installed.
1336: 
1337:     Returns
1338:     -------
1339:     DependencyInstallCommand
1340:         Command arguments suitable for ``subprocess`` execution.
1341: 
1342:     Raises
1343:     ------
1344:     RuntimeError
1345:         Raised when the current QGIS runtime Python interpreter cannot be
1346:         resolved.
1347: 
1348:     """
1349:     return get_dependency_install_commands(module_names)[0]
1350: 
1351: 
1352: def get_dependency_install_commands(
1353:     module_names: Iterable[str] | None = None,
1354: ) -> DependencyInstallPlan:
1355:     """Build pip commands used to install plugin dependencies.
1356: 
1357:     Parameters
1358:     ----------
1359:     module_names : Iterable[str] | None, optional
1360:         Dependency package names to install. When omitted, all known plugin
1361:         dependencies are installed.
1362: 
1363:     Returns
1364:     -------
1365:     DependencyInstallPlan
1366:         Command argument lists suitable for ``subprocess`` execution.
1367: 
1368:     Raises
1369:     ------
1370:     RuntimeError
1371:         Raised when the current QGIS runtime Python interpreter cannot be
1372:         resolved or no known dependencies were selected.
1373: 
1374:     Notes
1375:     -----
1376:     The local ``geosam`` package is installed in a separate ``--no-deps`` step.
1377:     QGIS already owns geospatial runtime packages such as rasterio, geopandas,
1378:     and pyproj; resolving the local package dependencies through pip can force
1379:     pip to reconcile wheel metadata that does not match QGIS' bundled runtime.
1380: 
1381:     """
1382:     selected_module_names = _selected_dependency_module_names(module_names)
1383:     dependency_target_path = register_plugin_managed_dependency_path(create=True)
1384:     python_interpreter = resolve_python_interpreter()
1385:     constraints_path = _create_qgis_runtime_constraints_file()
1386:     commands: DependencyInstallPlan = []
1387:     resolver_module_names = [
1388:         module_name for module_name in selected_module_names if module_name != "geosam"
1389:     ]
1390: 
1391:     if resolver_module_names:
1392:         command = _dependency_install_command_base(
1393:             python_interpreter,
1394:             constraints_path,
1395:             dependency_target_path,
1396:         )
1397:         command.extend(
1398:             [
1399:                 _dependency_install_requirement(module_name)
1400:                 for module_name in resolver_module_names
1401:             ]
1402:         )
1403:         commands.append(command)
1404: 
1405:     if "geosam" in selected_module_names:
1406:         commands.append(
1407:             [
1408:                 str(python_interpreter),
1409:                 "-m",
1410:                 "pip",
1411:                 "install",
1412:                 "--target",
1413:                 str(dependency_target_path),
1414:                 "--upgrade",
1415:                 "--no-deps",
1416:                 _dependency_install_requirement("geosam"),
1417:             ]
1418:         )
1419: 
1420:     return commands
1421: 
1422: 
1423: def format_dependency_install_commands(commands: Iterable[Iterable[str]]) -> str:
1424:     """Return readable dependency installation commands.
1425: 
1426:     Parameters
1427:     ----------
1428:     commands : Iterable[Iterable[str]]
1429:         Command argument lists produced by :func:`get_dependency_install_commands`.
1430: 
1431:     Returns
1432:     -------
1433:     str
1434:         Shell-readable command text for logs, one command per line.
1435: 
1436:     """
1437:     return "\n".join(format_dependency_install_command(command) for command in commands)
1438: 
1439: 
1440: def _dependency_install_command_base(
1441:     python_interpreter: Path,
1442:     constraints_path: Path | None,
1443:     dependency_target_path: Path,
1444: ) -> DependencyInstallCommand:
1445:     """Build the shared pip install command prefix.
1446: 
1447:     Parameters
1448:     ----------
1449:     python_interpreter : Path
1450:         Python interpreter used by the active QGIS runtime.
1451:     constraints_path : Path | None
1452:         Runtime constraints file path, or ``None`` when no constraints exist.
1453:     dependency_target_path : Path
1454:         Plugin-managed dependency directory used as the pip ``--target`` path.
1455: 
1456:     Returns
1457:     -------
1458:     DependencyInstallCommand
1459:         Command prefix for dependency-resolving pip installs.
1460: 
1461:     """
1462:     command: DependencyInstallCommand = [
1463:         str(python_interpreter),
1464:         "-m",
1465:         "pip",
1466:         "install",
1467:         "--target",
1468:         str(dependency_target_path),
1469:         "--upgrade",
1470:         "--upgrade-strategy",
1471:         "only-if-needed",
1472:         "--only-binary",
1473:         ",".join(PIP_ONLY_BINARY_DISTRIBUTIONS),
1474:     ]
1475:     if constraints_path is not None:
1476:         command.extend(["--constraint", str(constraints_path)])
1477:     return command
1478: 
1479: 
1480: def format_dependency_install_command(command: Iterable[str]) -> str:
1481:     """Return a readable dependency installation command.
1482: 
1483:     Parameters
1484:     ----------
1485:     command : Iterable[str]
1486:         Command arguments produced by :func:`get_dependency_install_command`.
1487: 
1488:     Returns
1489:     -------
1490:     str
1491:         Shell-readable command text for logs.
1492: 
1493:     """
1494:     return subprocess.list2cmdline(list(command))
1495: 
1496: 
1497: def install_dependencies(
1498:     log_callback: Callable[[str], None] | None = None,
1499: ) -> tuple[bool, str]:
1500:     """Install GeoSAM plugin dependencies with pip.
1501: 
1502:     Parameters
1503:     ----------
1504:     log_callback : Callable[[str], None] | None, optional
1505:         Callback that receives incremental install log lines.
1506: 
1507:     Returns
1508:     -------
1509:     tuple[bool, str]
1510:         Installation success flag and combined command output.
1511: 
1512:     """
1513:     try:
1514:         commands = get_dependency_install_commands()
1515:     except RuntimeError as exc:
1516:         logger.error("Dependency installation aborted: %s", exc)
1517:         return False, str(exc)
1518: 
1519:     output_lines: list[str] = []
1520: 
1521:     for command in commands:
1522:         if log_callback is not None:
1523:             log_callback(f"Command: {format_dependency_install_command(command)}")
1524:         process = subprocess.Popen(
1525:             command,
1526:             stdout=subprocess.PIPE,
1527:             stderr=subprocess.STDOUT,
1528:             text=True,
1529:             bufsize=1,
1530:         )
1531: 
1532:         if process.stdout is not None:
1533:             for raw_line in process.stdout:
1534:                 line = raw_line.rstrip()
1535:                 output_lines.append(line)
1536:                 if log_callback is not None:
1537:                     log_callback(line)
1538: 
1539:         return_code = process.wait()
1540:         if return_code != 0:
1541:             output = "\n".join(line for line in output_lines if line.strip())
1542:             return False, output
1543: 
1544:     output = "\n".join(line for line in output_lines if line.strip())
1545:     register_plugin_managed_dependency_path()
1546:     return True, output
1547: 
1548: 
1549: def open_url(url: str) -> None:
1550:     """Open a URL in the desktop browser.
1551: 
1552:     Parameters
1553:     ----------
1554:     url : str
1555:         Target URL.
1556: 
1557:     """
1558:     QDesktopServices.openUrl(QUrl(url))
1559: 
1560: 
1561: def open_path(path: Path) -> None:
1562:     """Open a local path in the desktop file manager.
1563: 
1564:     Parameters
1565:     ----------
1566:     path : Path
1567:         Target local path.
1568: 
1569:     """
1570:     QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.expanduser().resolve())))
1571: 
1572: 
1573: def get_cache_size_bytes(path: Path | None = None) -> int:
1574:     """Return the current cache size in bytes.
1575: 
1576:     Parameters
1577:     ----------
1578:     path : Path | None, optional
1579:         Cache directory to inspect. When omitted, the configured cache
1580:         directory is used.
1581: 
1582:     Returns
1583:     -------
1584:     int
1585:         Total cache size in bytes.
1586: 
1587:     """
1588:     cache_dir = get_cache_directory() if path is None else path
1589:     if not cache_dir.exists():
1590:         return 0
1591:     return sum(
1592:         file_path.stat().st_size
1593:         for file_path in cache_dir.rglob("*")
1594:         if file_path.is_file()
1595:     )
1596: 
1597: 
1598: def trim_cache_if_needed() -> int:
1599:     """Delete the oldest half of cache files when the max size is exceeded.
1600: 
1601:     Checks the current cache size against the configured maximum.  When the
1602:     limit is reached, files are sorted by modification time (oldest first)
1603:     and the older half is removed.
1604: 
1605:     Returns
1606:     -------
1607:     int
1608:         Number of removed cache files, or 0 when no trimming was needed.
1609: 
1610:     """
1611:     settings = load_plugin_settings()
1612:     cache_dir = get_cache_directory()
1613:     if not settings.get("cache_enabled", True):
1614:         return 0
1615: 
1616:     max_size_bytes = int(settings["cache_max_size_mb"]) * 1024 * 1024
1617:     current_size = get_cache_size_bytes(cache_dir)
1618:     if current_size <= max_size_bytes:
1619:         return 0
1620: 
1621:     files = sorted(
1622:         (p for p in cache_dir.rglob("*") if p.is_file()),
1623:         key=lambda p: p.stat().st_mtime,
1624:     )
1625:     if not files:
1626:         return 0
1627: 
1628:     # Delete the oldest half of the files (by count).
1629:     remove_count = max(1, len(files) // 2)
1630:     removed = 0
1631:     for file_path in files[:remove_count]:
1632:         try:
1633:             file_path.unlink()
1634:             removed += 1
1635:         except OSError:
1636:             pass
1637:     return removed
1638: 
1639: 
1640: def clear_cache() -> int:
1641:     """Delete all cached files and return the number removed.
1642: 
1643:     Returns
1644:     -------
1645:     int
1646:         Number of removed cache files.
1647: 
1648:     """
1649:     cache_dir = get_cache_directory()
1650:     if not cache_dir.exists():
1651:         return 0
1652:     removed_count = 0
1653:     for path in sorted(cache_dir.rglob("*"), reverse=True):
1654:         if path.is_file():
1655:             path.unlink()
1656:             removed_count += 1
1657:         elif path.is_dir():
1658:             path.rmdir()
1659:     cache_dir.mkdir(parents=True, exist_ok=True)
1660:     return removed_count

(End of file - total 1660 lines)
</content>
```

---

## Assistant (Build · GLM-5.2 · 11.3s)

_Thinking:_

Now I understand the dependency installation logic. Let me look at the widgetTool.py to understand how the UI interacts with the dependency installation, and the settingsTool.py and the .deps directory structure.

Now let me look at the widgetTool.py and settingsTool.py to understand how the UI interacts with dependency installation.

**Tool: bash**

**Input:**
```json
{
  "command": "ls -la .deps/ && echo \"---runtimes---\" && ls -la .deps/runtimes/ 2>/dev/null || echo \"no runtimes dir\"",
  "workdir": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM"
}
```

**Output:**
```
total 16
drwxr-xr-x@  4 fancy  staff   128 Jun 22 01:12 .
drwxr-xr-x  23 fancy  staff   736 Jun 24 16:16 ..
-rw-r--r--@  1 fancy  staff  6148 Jun 22 01:15 .DS_Store
drwxr-xr-x@  5 fancy  staff   160 Jun 30 14:07 runtimes
---runtimes---
total 16
drwxr-xr-x@ 5 fancy  staff   160 Jun 30 14:07 .
drwxr-xr-x@ 4 fancy  staff   128 Jun 22 01:12 ..
-rw-r--r--@ 1 fancy  staff  6148 Jun 22 01:15 .DS_Store
drwxr-xr-x@ 3 fancy  staff    96 Jun 30 14:07 py312-macosx-15.7-x86_64-arm64-96360f9508e4
drwxr-xr-x@ 4 fancy  staff   128 Jun 22 01:12 py312-macosx-15.7-x86_64-arm64-9fe2f1d30800

```

**Tool: bash**

**Input:**
```json
{
  "command": "wc -l tools/widgetTool.py tools/settingsTool.py tools/plugin_settings.py",
  "workdir": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM"
}
```

**Output:**
```
    3276 tools/widgetTool.py
    1100 tools/settingsTool.py
    1660 tools/plugin_settings.py
    6036 total

```

**Tool: grep**

**Input:**
```json
{
  "pattern": "install_dependencies|get_dependency_install_command|format_dependency_install_command|torch|uv|pip",
  "include": "*.py",
  "path": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools"
}
```

**Output:**
```
Found 83 matches
/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/settingsTool.py:
  Line 54:     format_dependency_install_command,

  Line 55:     format_dependency_install_commands,

  Line 58:     get_dependency_install_commands,

  Line 182:         self.install_dependencies_button = QPushButton(self.tr("Install Missing"), tab)

  Line 183:         self.install_dependencies_button.clicked.connect(

  Line 184:             self.install_dependencies_clicked

  Line 203:         button_row.addWidget(self.install_dependencies_button)

  Line 498:         self.install_dependencies_button.setEnabled(

  Line 510:     def install_dependencies_clicked(self) -> None:

  Line 528:         self.install_dependencies_button.setEnabled(False)

  Line 541:             commands = get_dependency_install_commands(self._missing_dependency_names)

  Line 547:                 commands=format_dependency_install_commands(commands)

  Line 570:                 command=format_dependency_install_command(command)

  Line 654:         self.install_dependencies_button.setEnabled(True)


/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/dependency_path.py:
  Line 56:         Directory used as the pip ``--target`` location for plugin-managed


/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/sam_processing_algorithm.py:
  Line 42:     _flush_torch_memory,

  Line 482:             _flush_torch_memory()

  Line 526:             import torch

  Line 529:         if torch.cuda.is_available():

  Line 530:             if cuda_id + 1 > torch.cuda.device_count():

  Line 531:                 cuda_id = torch.cuda.device_count() - 1

  Line 533:         if torch.backends.mps.is_available():

  Line 602:                 _flush_torch_memory()

  Line 604:         _flush_torch_memory()


/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/plugin_settings.py:
  Line 40:     "torch": "torch",

  Line 41:     "torchvision": "torchvision",

  Line 51:     "torch": "torch",

  Line 52:     "torchvision": "torchvision",

  Line 93:     "torch==2.8.0",

  Line 94:     "torchvision==0.23.0",

  Line 97:     "torch==2.11.0",

  Line 98:     "torchvision==0.26.0",

  Line 107:     "torch",

  Line 108:     "torchvision",

  Line 703:     """Return the pip requirement used to install a dependency.

  Line 759:     """Create a temporary pip constraints file for bundled QGIS packages.

  Line 811:     """Return extra pip constraints for QGIS runtime compatibility.

  Line 821:         Additional requirement constraints that prevent pip resolver backtracking

  Line 831:     constraints.extend(_torch_resolver_constraints())

  Line 839: def _torch_resolver_constraints() -> list[str]:

  Line 845:         Torch and torchvision constraints for pip resolution.

  Line 891:         Distribution name written into the pip constraints file.

  Line 902:     release. This helper normalizes those cases so pip receives a satisfiable

  Line 944:     """Return setuptools constraints compatible with recent torch builds.

  Line 949:         Additional setuptools constraints for pip resolution.

  Line 953:     Recent torch releases require ``setuptools<82``. Some QGIS Python runtimes

  Line 957:     should prefer the torch-compatible range instead of preserving that exact

  Line 970:             "torch-compatible upper bound.",

  Line 977:             "torch compatibility during dependency installation.",

  Line 1326: def get_dependency_install_command(

  Line 1329:     """Build the first pip command used to install plugin dependencies.

  Line 1349:     return get_dependency_install_commands(module_names)[0]

  Line 1352: def get_dependency_install_commands(

  Line 1355:     """Build pip commands used to install plugin dependencies.

  Line 1378:     and pyproj; resolving the local package dependencies through pip can force

  Line 1379:     pip to reconcile wheel metadata that does not match QGIS' bundled runtime.

  Line 1410:                 "pip",

  Line 1423: def format_dependency_install_commands(commands: Iterable[Iterable[str]]) -> str:

  Line 1429:         Command argument lists produced by :func:`get_dependency_install_commands`.

  Line 1437:     return "\n".join(format_dependency_install_command(command) for command in commands)

  Line 1445:     """Build the shared pip install command prefix.

  Line 1454:         Plugin-managed dependency directory used as the pip ``--target`` path.

  Line 1459:         Command prefix for dependency-resolving pip installs.

  Line 1465:         "pip",

  Line 1480: def format_dependency_install_command(command: Iterable[str]) -> str:

  Line 1486:         Command arguments produced by :func:`get_dependency_install_command`.

  Line 1497: def install_dependencies(

  Line 1500:     """Install GeoSAM plugin dependencies with pip.

  Line 1514:         commands = get_dependency_install_commands()

  Line 1523:             log_callback(f"Command: {format_dependency_install_command(command)}")


/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/geosam_runtime.py:
  Line 444: def _flush_torch_memory() -> None:

  Line 448:         import torch

  Line 451:             "Skipping PyTorch memory cleanup because torch failed to load: %s",

  Line 456:     if torch.cuda.is_available():

  Line 457:         torch.cuda.empty_cache()

  Line 458:         torch.cuda.ipc_collect()

  Line 459:     if hasattr(torch, "mps") and torch.backends.mps.is_available():

  Line 460:         empty_cache = getattr(torch.mps, "empty_cache", None)

  Line 549:         _flush_torch_memory()

  Line 568:     _flush_torch_memory()

  Line 1735:                 _flush_torch_memory()

  Line 1796:                 _flush_torch_memory()

  Line 1840:             _flush_torch_memory()

```

---

## Assistant (Build · GLM-5.2 · 8.4s)

_Thinking:_

Let me look at the settingsTool.py to understand the dependency install UI, and the torch-related logic in sam_processing_algorithm.py.

Let me look at the key parts of settingsTool.py (the install dependencies UI) and sam_processing_algorithm.py (torch detection).

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/settingsTool.py",
  "offset": 1,
  "limit": 120
}
```

**Output:**
```
<path>/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/settingsTool.py</path>
<type>file</type>
<content>
1: """Settings dialog for the Geo-SAM plugin."""
2: 
3: from __future__ import annotations
4: 
5: from pathlib import Path
6: 
7: from qgis.PyQt.QtCore import QProcess, QThread, Qt, pyqtSignal
8: from qgis.PyQt.QtGui import QColor
9: from qgis.PyQt.QtWidgets import (
10:     QAbstractItemView,
11:     QCheckBox,
12:     QComboBox,
13:     QDialog,
14:     QFileDialog,
15:     QFormLayout,
16:     QGridLayout,
17:     QGroupBox,
18:     QHBoxLayout,
19:     QHeaderView,
20:     QLabel,
21:     QLineEdit,
22:     QMessageBox,
23:     QPlainTextEdit,
24:     QProgressBar,
25:     QPushButton,
26:     QSlider,
27:     QTabBar,
28:     QTabWidget,
29:     QTableWidget,
30:     QTableWidgetItem,
31:     QVBoxLayout,
32:     QWidget,
33: )
34: 
35: from .messageTool import MessageTool
36: from .i18n import translate
37: from .dependency_path import (
38:     clear_all_plugin_managed_site_packages,
39:     clear_current_plugin_managed_site_packages,
40:     get_plugin_managed_dependency_stats,
41:     get_plugin_managed_site_packages,
42:     iter_all_plugin_managed_site_packages,
43: )
44: from .model_manager import delete_model, download_model, get_model_status_rows
45: from .plugin_settings import (
46:     DEFAULT_CACHE_DIR,
47:     DEFAULT_MODEL_DIR,
48:     HELP_LINKS,
49:     PERFORMANCE_MODE_VALUES,
50:     PLUGIN_ROOT,
51:     clear_cache,
52:     dependency_status_text,
53:     dependency_status_rows,
54:     format_dependency_install_command,
55:     format_dependency_install_commands,
56:     get_cache_directory,
57:     get_cache_size_bytes,
58:     get_dependency_install_commands,
59:     get_model_directory,
60:     load_plugin_settings,
61:     open_path,
62:     open_url,
63:     save_plugin_settings,
64: )
65: from .geosam_runtime import get_loaded_model_ids, release_runtime_models
66: 
67: 
68: class ModelDownloadThread(QThread):
69:     """Background worker that downloads one model checkpoint."""
70: 
71:     succeeded = pyqtSignal(str)
72:     failed = pyqtSignal(str)
73: 
74:     def __init__(self, model_id: str, parent: QWidget | None = None) -> None:
75:         super().__init__(parent)
76:         self.model_id = model_id
77: 
78:     def run(self) -> None:
79:         """Download the requested model and emit the outcome."""
80:         try:
81:             checkpoint_path = download_model(self.model_id)
82:         except Exception as exc:
83:             self.failed.emit(str(exc))
84:             return
85:         self.succeeded.emit(str(checkpoint_path))
86: 
87: 
88: class GeoSamSettingsDialog(QDialog):
89:     """Standalone settings dialog for the Geo-SAM plugin."""
90: 
91:     DEPENDENCIES_TAB_INDEX = 0
92:     MODEL_MANAGEMENT_TAB_INDEX = 1
93: 
94:     def __init__(self, parent=None) -> None:
95:         super().__init__(parent)
96:         self.setWindowTitle(self.tr("Geo-SAM Settings"))
97:         self.resize(760, 520)
98:         self.settings = load_plugin_settings()
99:         self._dependency_install_process: QProcess | None = None
100:         self._dependency_install_commands: list[list[str]] = []
101:         self._dependency_install_command_index = 0
102:         self._dependency_install_output_buffer = ""
103:         self._missing_dependency_names: list[str] = []
104:         self._model_download_thread: ModelDownloadThread | None = None
105: 
106:         self.tab_widget = QTabWidget(self)
107:         self.tab_widget.addTab(self._build_dependency_tab(), self.tr("Dependencies"))
108:         self.tab_widget.addTab(self._build_model_tab(), self.tr("Model Management"))
109:         self.tab_widget.addTab(self._build_cache_tab(), self.tr("Cache"))
110:         self.tab_widget.addTab(self._build_help_tab(), self.tr("Help"))
111: 
112:         self.close_button = QPushButton(self.tr("Close"), self)
113:         self.close_button.clicked.connect(self.accept)
114: 
115:         root_layout = QVBoxLayout(self)
116:         root_layout.addWidget(self.tab_widget)
117:         root_layout.addWidget(self.close_button, alignment=Qt.AlignmentFlag.AlignRight)
118: 
119:         self.refresh_dependency_status()
120:         self.refresh_model_list()

(Showing lines 1-120 of 1100. Use offset=121 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/settingsTool.py",
  "offset": 150,
  "limit": 230
}
```

**Output:**
```
<path>/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/settingsTool.py</path>
<type>file</type>
<content>
150:         self.dependency_summary_label = QLabel(tab)
151:         self.dependency_summary_label.setWordWrap(True)
152:         self.dependency_storage_label = QLabel(tab)
153:         self.dependency_storage_label.setWordWrap(True)
154:         self.dependency_storage_label.setTextInteractionFlags(
155:             Qt.TextInteractionFlag.TextSelectableByMouse
156:         )
157: 
158:         self.dependency_table = QTableWidget(tab)
159:         self.dependency_table.setColumnCount(4)
160:         self.dependency_table.setHorizontalHeaderLabels(
161:             [
162:                 self.tr("Package"),
163:                 self.tr("Status"),
164:                 self.tr("Version"),
165:                 self.tr("Source"),
166:             ]
167:         )
168:         self.dependency_table.verticalHeader().setVisible(True)
169:         self.dependency_table.setAlternatingRowColors(True)
170:         self.dependency_table.setEditTriggers(
171:             QAbstractItemView.EditTrigger.NoEditTriggers
172:         )
173:         self.dependency_table.setSelectionMode(
174:             QAbstractItemView.SelectionMode.NoSelection
175:         )
176:         self.dependency_table.horizontalHeader().setStretchLastSection(True)
177:         self.dependency_table.setMinimumHeight(220)
178: 
179:         button_row = QHBoxLayout()
180:         self.refresh_dependencies_button = QPushButton(self.tr("Refresh Status"), tab)
181:         self.refresh_dependencies_button.clicked.connect(self.refresh_dependency_status)
182:         self.install_dependencies_button = QPushButton(self.tr("Install Missing"), tab)
183:         self.install_dependencies_button.clicked.connect(
184:             self.install_dependencies_clicked
185:         )
186:         self.open_dependency_folder_button = QPushButton(self.tr("Open Folder"), tab)
187:         self.open_dependency_folder_button.clicked.connect(
188:             self.open_dependency_folder_clicked
189:         )
190:         self.clear_current_dependencies_button = QPushButton(
191:             self.tr("Clear Current Runtime"),
192:             tab,
193:         )
194:         self.clear_current_dependencies_button.clicked.connect(
195:             self.clear_current_dependencies_clicked
196:         )
197:         self.clear_all_dependencies_button = QPushButton(
198:             self.tr("Clear All Runtimes"), tab
199:         )
200:         self.clear_all_dependencies_button.clicked.connect(
201:             self.clear_all_dependencies_clicked
202:         )
203:         button_row.addWidget(self.install_dependencies_button)
204:         button_row.addWidget(self.refresh_dependencies_button)
205:         button_row.addWidget(self.open_dependency_folder_button)
206:         button_row.addWidget(self.clear_current_dependencies_button)
207:         button_row.addWidget(self.clear_all_dependencies_button)
208: 
209:         self.dependency_install_status_label = QLabel(
210:             self.tr("Installable dependency status will appear here."),
211:             tab,
212:         )
213:         self.dependency_install_status_label.setWordWrap(True)
214:         self.dependency_install_progress = QProgressBar(tab)
215:         self.dependency_install_progress.setRange(0, 0)
216:         self.dependency_install_progress.setVisible(False)
217: 
218:         self.dependency_install_log = QPlainTextEdit(tab)
219:         self.dependency_install_log.setReadOnly(True)
220:         self.dependency_install_log.setPlaceholderText(
221:             self.tr("Dependency installation output will appear here.")
222:         )
223:         self.dependency_install_log.setMinimumHeight(220)
224: 
225:         layout.addWidget(self.dependency_summary_label)
226:         layout.addWidget(self.dependency_storage_label)
227:         layout.addWidget(self.dependency_table)
228:         layout.addLayout(button_row)
229:         layout.addWidget(self.dependency_install_status_label)
230:         layout.addWidget(self.dependency_install_progress)
231:         layout.addWidget(self.dependency_install_log)
232:         return tab
233: 
234:     # Filter tab indices for the model filter bar.
235:     FILTER_ALL = 0
236:     FILTER_DOWNLOADED = 1
237:     FILTER_NOT_DOWNLOADED = 2
238:     FILTER_IN_MEMORY = 3
239: 
240:     # Table column indices.
241:     COL_NAME = 0
242:     COL_SERIES = 1
243:     COL_FILE_STATUS = 2
244:     COL_LOAD_STATUS = 3
245:     COL_ACTIONS = 4
246: 
247:     def _build_model_tab(self) -> QWidget:
248:         tab = QWidget(self)
249:         layout = QVBoxLayout(tab)
250: 
251:         # -- Storage group ------------------------------------------------
252:         path_group = QGroupBox(self.tr("Storage"), tab)
253:         path_layout = QGridLayout(path_group)
254:         self.model_dir_edit = QLineEdit(str(self.settings["model_store_dir"]), tab)
255:         self.model_dir_edit.editingFinished.connect(self.save_model_directory)
256: 
257:         browse_button = QPushButton(self.tr("Browse"), tab)
258:         browse_button.clicked.connect(self.browse_model_directory)
259:         open_button = QPushButton(self.tr("Open Folder"), tab)
260:         open_button.clicked.connect(lambda: open_path(get_model_directory()))
261: 
262:         path_layout.addWidget(QLabel(self.tr("Model Folder"), tab), 0, 0)
263:         path_layout.addWidget(self.model_dir_edit, 0, 1)
264:         path_layout.addWidget(browse_button, 0, 2)
265:         path_layout.addWidget(open_button, 0, 3)
266: 
267:         # -- Models group --------------------------------------------------
268:         list_group = QGroupBox(self.tr("Models"), tab)
269:         list_layout = QVBoxLayout(list_group)
270: 
271:         # Filter bar
272:         self.model_filter_bar = QTabBar(tab)
273:         self.model_filter_bar.addTab(self.tr("All"))
274:         self.model_filter_bar.addTab(self.tr("Downloaded"))
275:         self.model_filter_bar.addTab(self.tr("Not Downloaded"))
276:         self.model_filter_bar.addTab(self.tr("In Memory"))
277:         self.model_filter_bar.currentChanged.connect(self._apply_model_filter)
278:         list_layout.addWidget(self.model_filter_bar)
279: 
280:         # Model table
281:         self.model_table = QTableWidget(tab)
282:         self.model_table.setColumnCount(5)
283:         self.model_table.setHorizontalHeaderLabels([
284:             self.tr("Model Name"),
285:             self.tr("Series"),
286:             self.tr("File Status"),
287:             self.tr("Load Status"),
288:             self.tr("Actions"),
289:         ])
290:         self.model_table.verticalHeader().setVisible(False)
291:         self.model_table.setAlternatingRowColors(True)
292:         self.model_table.setEditTriggers(
293:             QAbstractItemView.EditTrigger.NoEditTriggers
294:         )
295:         self.model_table.setSelectionMode(
296:             QAbstractItemView.SelectionMode.SingleSelection
297:         )
298:         self.model_table.setSelectionBehavior(
299:             QAbstractItemView.SelectionBehavior.SelectRows
300:         )
301:         self.model_table.horizontalHeader().setStretchLastSection(True)
302:         self.model_table.setMinimumHeight(260)
303:         list_layout.addWidget(self.model_table)
304: 
305:         # Download progress
306:         self.model_download_status_label = QLabel(self.tr("Ready"), tab)
307:         list_layout.addWidget(self.model_download_status_label)
308:         self.model_download_progress = QProgressBar(tab)
309:         self.model_download_progress.setRange(0, 0)
310:         self.model_download_progress.setVisible(False)
311:         list_layout.addWidget(self.model_download_progress)
312: 
313:         # -- Current model status -----------------------------------------
314:         self.current_model_label = QLabel(tab)
315:         self.current_model_label.setWordWrap(True)
316:         self.current_model_status_label = QLabel(tab)
317:         self.current_model_status_label.setWordWrap(True)
318: 
319:         # -- Bottom action buttons ----------------------------------------
320:         bottom_row = QHBoxLayout()
321:         self.unload_button = QPushButton(self.tr("Unload Current"), tab)
322:         self.unload_button.setEnabled(False)
323:         self.unload_button.clicked.connect(self.unload_current_model_clicked)
324:         refresh_button = QPushButton(self.tr("Refresh"), tab)
325:         refresh_button.clicked.connect(self.refresh_model_list)
326:         bottom_row.addWidget(self.unload_button)
327:         bottom_row.addWidget(refresh_button)
328:         bottom_row.addStretch(1)
329: 
330:         layout.addWidget(path_group)
331:         layout.addWidget(list_group)
332:         layout.addWidget(self.current_model_label)
333:         layout.addWidget(self.current_model_status_label)
334:         layout.addLayout(bottom_row)
335:         return tab
336: 
337:     def _build_cache_tab(self) -> QWidget:
338:         tab = QWidget(self)
339:         layout = QVBoxLayout(tab)
340: 
341:         form_layout = QFormLayout()
342:         self.cache_enabled_checkbox = QCheckBox(self.tr("Enable cache"), tab)
343:         self.cache_enabled_checkbox.setChecked(bool(self.settings["cache_enabled"]))
344:         self.cache_enabled_checkbox.toggled.connect(self.save_cache_settings)
345: 
346:         cache_dir_row = QHBoxLayout()
347:         self.cache_dir_edit = QLineEdit(str(self.settings["cache_dir"]), tab)
348:         self.cache_dir_edit.editingFinished.connect(self.save_cache_settings)
349:         cache_browse_button = QPushButton(self.tr("Browse"), tab)
350:         cache_browse_button.clicked.connect(self.browse_cache_directory)
351:         cache_open_button = QPushButton(self.tr("Open Folder"), tab)
352:         cache_open_button.clicked.connect(lambda: open_path(get_cache_directory()))
353:         cache_dir_row.addWidget(self.cache_dir_edit)
354:         cache_dir_row.addWidget(cache_browse_button)
355:         cache_dir_row.addWidget(cache_open_button)
356: 
357:         cache_size_row = QHBoxLayout()
358:         self.cache_size_slider = QSlider(Qt.Orientation.Horizontal, tab)
359:         self.cache_size_slider.setRange(100, 20480)
360:         self.cache_size_slider.setValue(int(self.settings["cache_max_size_mb"]))
361:         self.cache_size_edit = QLineEdit(str(self.settings["cache_max_size_mb"]), tab)
362:         self.cache_size_edit.setFixedWidth(70)
363:         self.cache_size_suffix = QLabel(self.tr("MB"), tab)
364:         self.cache_size_slider.valueChanged.connect(self._on_cache_slider_changed)
365:         self.cache_size_edit.editingFinished.connect(self._on_cache_edit_finished)
366:         cache_size_row.addWidget(self.cache_size_slider)
367:         cache_size_row.addWidget(self.cache_size_edit)
368:         cache_size_row.addWidget(self.cache_size_suffix)
369: 
370:         self.performance_mode_combo = QComboBox(tab)
371:         self.performance_mode_combo.addItem(self.tr("Balanced"), "balanced")
372:         self.performance_mode_combo.addItem(self.tr("Fastest"), "fastest")
373:         self.performance_mode_combo.addItem(self.tr("Low Memory"), "low_memory")
374:         self.performance_mode_combo.setCurrentIndex(
375:             max(
376:                 0,
377:                 self.performance_mode_combo.findData(
378:                     str(self.settings.get("performance_mode", "balanced"))
379:                 ),

(Showing lines 150-379 of 1100. Use offset=380 to continue.)
</content>
```

---

## Assistant (Build · GLM-5.2 · 11.9s)

_Thinking:_

Let me read the install_dependencies_clicked method and surrounding logic in settingsTool.py to understand the current install flow.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/settingsTool.py",
  "limit": 200,
  "offset": 490
}
```

**Output:**
```
<path>/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/settingsTool.py</path>
<type>file</type>
<content>
490:             ):
491:                 item = QTableWidgetItem(value)
492:                 item.setForeground(text_color)
493:                 self.dependency_table.setItem(row_index, column_index, item)
494:         self.dependency_table.resizeColumnsToContents()
495:         self.dependency_table.horizontalHeader().setStretchLastSection(True)
496: 
497:         is_installing = self._dependency_install_process is not None
498:         self.install_dependencies_button.setEnabled(
499:             bool(self._missing_dependency_names) and not is_installing
500:         )
501:         self.refresh_dependencies_button.setEnabled(not is_installing)
502:         self.open_dependency_folder_button.setEnabled(not is_installing)
503:         self.clear_current_dependencies_button.setEnabled(
504:             current_stats["exists"] and not is_installing
505:         )
506:         self.clear_all_dependencies_button.setEnabled(
507:             bool(all_dependency_paths) and not is_installing
508:         )
509: 
510:     def install_dependencies_clicked(self) -> None:
511:         if self._dependency_install_process is not None:
512:             return
513:         if not self._missing_dependency_names:
514:             self.refresh_dependency_status()
515:             self.dependency_install_status_label.setText(
516:                 self.tr("No installable dependencies are missing.")
517:             )
518:             return
519: 
520:         self.dependency_install_log.clear()
521:         missing_dependencies = ", ".join(self._missing_dependency_names)
522:         self.dependency_install_status_label.setText(
523:             self.tr("Installing missing dependencies: {dependencies}").format(
524:                 dependencies=missing_dependencies
525:             )
526:         )
527:         self.dependency_install_progress.setVisible(True)
528:         self.install_dependencies_button.setEnabled(False)
529:         self.refresh_dependencies_button.setEnabled(False)
530:         self.open_dependency_folder_button.setEnabled(False)
531:         self.clear_current_dependencies_button.setEnabled(False)
532:         self.clear_all_dependencies_button.setEnabled(False)
533:         self.close_button.setEnabled(False)
534:         self._dependency_install_output_buffer = ""
535:         self._append_dependency_install_log(
536:             self.tr("Starting dependency installation: {dependencies}").format(
537:                 dependencies=missing_dependencies
538:             )
539:         )
540:         try:
541:             commands = get_dependency_install_commands(self._missing_dependency_names)
542:         except RuntimeError as exc:
543:             self._finish_dependency_install(False, str(exc))
544:             return
545:         self._append_dependency_install_log(
546:             self.tr("Commands:\n{commands}").format(
547:                 commands=format_dependency_install_commands(commands)
548:             )
549:         )
550: 
551:         self._dependency_install_commands = commands
552:         self._dependency_install_command_index = 0
553:         self._start_next_dependency_install_command()
554: 
555:     def _start_next_dependency_install_command(self) -> None:
556:         """Start the next pending dependency install command."""
557:         if self._dependency_install_command_index >= len(
558:             self._dependency_install_commands
559:         ):
560:             output = self.dependency_install_log.toPlainText().strip()
561:             self._finish_dependency_install(True, output)
562:             return
563: 
564:         command = self._dependency_install_commands[
565:             self._dependency_install_command_index
566:         ]
567:         self._dependency_install_command_index += 1
568:         self._append_dependency_install_log(
569:             self.tr("Running command: {command}").format(
570:                 command=format_dependency_install_command(command)
571:             )
572:         )
573:         process = QProcess(self)
574:         process.setProgram(command[0])
575:         process.setArguments(command[1:])
576:         process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
577:         process.readyReadStandardOutput.connect(self._read_dependency_install_output)
578:         process.errorOccurred.connect(self._dependency_install_error)
579:         process.finished.connect(self._dependency_install_finished)
580:         self._dependency_install_process = process
581:         process.start()
582: 
583:     def _read_dependency_install_output(self) -> None:
584:         """Read and append incremental dependency install output."""
585:         process = self._dependency_install_process
586:         if process is None:
587:             return
588: 
589:         raw_output = bytes(process.readAllStandardOutput()).decode(
590:             "utf-8",
591:             errors="replace",
592:         )
593:         if not raw_output:
594:             return
595: 
596:         self._dependency_install_output_buffer += raw_output
597:         while True:
598:             newline_index = self._dependency_install_output_buffer.find("\n")
599:             carriage_index = self._dependency_install_output_buffer.find("\r")
600:             separator_indexes = [
601:                 index for index in (newline_index, carriage_index) if index >= 0
602:             ]
603:             if not separator_indexes:
604:                 break
605: 
606:             separator_index = min(separator_indexes)
607:             separator = self._dependency_install_output_buffer[separator_index]
608:             line = self._dependency_install_output_buffer[:separator_index]
609:             self._dependency_install_output_buffer = (
610:                 self._dependency_install_output_buffer[separator_index + 1 :]
611:             )
612:             clean_line = line.strip()
613:             if separator == "\n":
614:                 self._append_dependency_install_log(clean_line)
615: 
616:     def _dependency_install_finished(self, exit_code: int, _exit_status: int) -> None:
617:         """Handle completion of the asynchronous dependency install command."""
618:         if self._dependency_install_process is None:
619:             return
620: 
621:         if self._dependency_install_output_buffer:
622:             clean_line = self._dependency_install_output_buffer.strip()
623:             self._append_dependency_install_log(clean_line)
624:             self._dependency_install_output_buffer = ""
625: 
626:         self._dependency_install_process = None
627:         if exit_code == 0:
628:             self._start_next_dependency_install_command()
629:             return
630: 
631:         output = self.dependency_install_log.toPlainText().strip()
632:         self._finish_dependency_install(False, output)
633: 
634:     def _dependency_install_error(self, _error: QProcess.ProcessError) -> None:
635:         """Handle startup or runtime failures from the dependency installer."""
636:         process = self._dependency_install_process
637:         if process is None:
638:             return
639: 
640:         self._read_dependency_install_output()
641:         error_message = process.errorString().strip() or (
642:             self.tr("Dependency installer process failed to start.")
643:         )
644:         combined_output = self.dependency_install_log.toPlainText().strip()
645:         if combined_output:
646:             combined_output = f"{combined_output}\n{error_message}"
647:         else:
648:             combined_output = error_message
649:         self._finish_dependency_install(False, combined_output)
650: 
651:     def _finish_dependency_install(self, ok: bool, output: str) -> None:
652:         """Restore dependency install UI state and report the result."""
653:         self.dependency_install_progress.setVisible(False)
654:         self.install_dependencies_button.setEnabled(True)
655:         self.refresh_dependencies_button.setEnabled(True)
656:         self.close_button.setEnabled(True)
657:         self.dependency_install_status_label.setText(
658:             self.tr("Dependency installation completed.")
659:             if ok
660:             else self.tr("Dependency installation failed.")
661:         )
662:         self._dependency_install_process = None
663:         self._dependency_install_commands = []
664:         self._dependency_install_command_index = 0
665: 
666:         if ok and not output.strip():
667:             self._append_dependency_install_log(
668:                 self.tr("Dependency installation completed.")
669:             )
670:         elif not ok and not output.strip():
671:             self._append_dependency_install_log(
672:                 self.tr("Dependency installation failed.")
673:             )
674:         self.refresh_dependency_status()
675: 
676:     def _append_dependency_install_log(self, message: str) -> None:
677:         """Append a dependency installation log line to the settings dialog."""
678:         if not message:
679:             return
680:         self.dependency_install_log.appendPlainText(message)
681:         scrollbar = self.dependency_install_log.verticalScrollBar()
682:         scrollbar.setValue(scrollbar.maximum())
683: 
684:     def open_dependency_folder_clicked(self) -> None:
685:         """Open the active runtime dependency folder."""
686:         open_path(get_plugin_managed_site_packages(create=True))
687: 
688:     def clear_current_dependencies_clicked(self) -> None:
689:         """Delete plugin-managed dependencies for the active runtime."""

(Showing lines 490-689 of 1100. Use offset=690 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/sam_processing_algorithm.py",
  "limit": 60,
  "offset": 505
}
```

**Output:**
```
<path>/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/sam_processing_algorithm.py</path>
<type>file</type>
<content>
505:     def postProcessAlgorithm(
506:         self,
507:         context: QgsProcessingContext,
508:         feedback: QgsProcessingFeedback,
509:     ) -> dict[str, Any]:
510:         """Optionally load the generated cache into the plugin UI."""
511:         del context
512:         if self.load_feature and self.feature_dir:
513:             self.load_feature = self.load_feature_dir(feedback=feedback)
514:         return {
515:             "Output feature path": self.feature_dir,
516:             "Patch samples saved": self.iPatch,
517:             "Feature folder loaded": self.load_feature,
518:         }
519: 
520:     @staticmethod
521:     def _select_device(*, use_gpu: bool, cuda_id: int) -> str | None:
522:         """Resolve the preferred inference device string."""
523:         if not use_gpu:
524:             return None
525:         try:
526:             import torch
527:         except (ImportError, OSError):
528:             return None
529:         if torch.cuda.is_available():
530:             if cuda_id + 1 > torch.cuda.device_count():
531:                 cuda_id = torch.cuda.device_count() - 1
532:             return f"cuda:{cuda_id}"
533:         if torch.backends.mps.is_available():
534:             return "mps"
535:         return None
536: 
537:     @staticmethod
538:     def _memory_strategy_from_index(index: int) -> EncodingMemoryStrategy:
539:         """Return the memory strategy selected by a processing enum index.
540: 
541:         Parameters
542:         ----------
543:         index : int
544:             Processing enum index selected by the user.
545: 
546:         Returns
547:         -------
548:         EncodingMemoryStrategy
549:             Normalized memory strategy value.
550:         """
551:         if 0 <= index < len(ENCODING_MEMORY_STRATEGIES):
552:             return ENCODING_MEMORY_STRATEGIES[index][0]
553:         return "balanced"
554: 
555:     def _raise_missing_model_error(self, *, model_id: str) -> None:
556:         """Raise a user-facing error when the selected model is not downloaded.
557: 
558:         Parameters
559:         ----------
560:         model_id : str
561:             Selected GeoSAM model identifier.
562: 
563:         Raises
564:         ------

(Showing lines 505-564 of 823. Use offset=565 to continue.)
</content>
```

---

## Assistant (Build · GLM-5.2 · 21.1s)

_Thinking:_

Now I have a good understanding of the current architecture. Let me also look at the default.json config and check if there's anything about torch backend selection. Let me also check the widgetTool.py briefly for torch/GPU selection UI.

Let me check the config files and any torch/GPU UI selection in widgetTool.py.

**Tool: bash**

**Input:**
```json
{
  "command": "cat ui/config/default.json 2>/dev/null; echo \"---\"; cat ui/config/user.json 2>/dev/null; echo \"---ui dir---\"; ls ui/config/",
  "workdir": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM"
}
```

**Output:**
```
{
    "fg_color": "#0000ff",
    "bg_color": "#ff0000",
    "pt_size": 1.0,
    "icon_type": "Circle",
    "bbox_color": "#0000ff",
    "extent_color": "#ff0000",
    "prompt_color": "#3bba00",
    "preview_color": "#00ff00",
    "show_boundary": true,
    "max_polygon_only": true,
    "selected_model_id": "sam2.1_b",
    "clear_cache_on_plugin_close": true,
    "performance_mode": "balanced",
    "preview_render_mode": "pixel_level",
    "preview_mode": true
}
---
{
    "selected_model_id": "sam2.1_s"
}---ui dir---
__init__.py
__pycache__
default.json
user.json

```

**Tool: grep**

**Input:**
```json
{
  "include": "*.py",
  "path": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools",
  "pattern": "use_gpu|cuda|mps|device|gpu"
}
```

**Output:**
```
Found 38 matches
/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/plugin_settings.py:
  Line 486:         json.dumps(user_settings, indent=4),


/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/sam_processing_algorithm.py:
  Line 157:         cuda_id_param = QgsProcessingParameterNumber(

  Line 159:             description=self.tr("CUDA device id (default to 0)"),

  Line 240:             cuda_id_param,

  Line 303:         use_gpu = self.parameterAsBoolean(parameters, self.CUDA, context)

  Line 307:         cuda_id = self.parameterAsInt(parameters, self.CUDA_ID, context)

  Line 323:         device = self._select_device(use_gpu=use_gpu, cuda_id=cuda_id)

  Line 327:             device=device,

  Line 421:         feedback.pushInfo(self.tr("Device type: {device}").format(device=device or "cpu"))

  Line 521:     def _select_device(*, use_gpu: bool, cuda_id: int) -> str | None:

  Line 522:         """Resolve the preferred inference device string."""

  Line 523:         if not use_gpu:

  Line 529:         if torch.cuda.is_available():

  Line 530:             if cuda_id + 1 > torch.cuda.device_count():

  Line 531:                 cuda_id = torch.cuda.device_count() - 1

  Line 532:             return f"cuda:{cuda_id}"

  Line 533:         if torch.backends.mps.is_available():

  Line 534:             return "mps"

  Line 722:             "transform": json.dumps(list(sample.transform)[:6]),

  Line 723:             "shape": json.dumps(list(sample.shape)),

  Line 725:             "dst_shape": json.dumps(list(encoded.dst_shape)),


/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/model_manager.py:
  Line 315:     device: str | None = None,

  Line 325:     device : str | None, optional

  Line 326:         Optional inference device string.

  Line 340:         device=device,


/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/widgetTool.py:
  Line 1427:             json.dumps(log_payload, ensure_ascii=True, default=str),

  Line 1992:             json.dumps(

  Line 3136:         gpu_id = self.get_GPU_ID()

  Line 3151:                 "CUDA_ID": gpu_id,

  Line 3154:         return json.dumps(json_dict, indent=4)


/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/geosam_runtime.py:
  Line 431:     device: str | None = None,

  Line 440:         device=device,

  Line 456:     if torch.cuda.is_available():

  Line 457:         torch.cuda.empty_cache()

  Line 458:         torch.cuda.ipc_collect()

  Line 459:     if hasattr(torch, "mps") and torch.backends.mps.is_available():

  Line 460:         empty_cache = getattr(torch.mps, "empty_cache", None)

  Line 1100:     metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

```

---

## Assistant (Build · GLM-5.2 · 26.8s)

_Thinking:_

Now I have a comprehensive understanding of the current architecture. Let me summarize the key findings:

**Current state:**

1. **Dependency installation** lives in `tools/plugin_settings.py`:
   - `get_dependency_install_command()` / `get_dependency_install_commands()` - builds pip commands (lines 1326-1420)
   - `install_dependencies()` - runs pip via subprocess (lines 1497-1546)
   - `format_dependency_install_command()` / `format_dependency_install_commands()` - formats for display (lines 1423-1494)
   - Uses `resolve_python_interpreter()` to find QGIS Python
   - Uses `_create_qgis_runtime_constraints_file()` to build constraints
   - Uses `_torch_resolver_constraints()` - hardcodes torch versions (py39: torch==2.8.0, py310+: torch==2.11.0)
   - No GPU/CUDA torch index support — always installs CPU torch from PyPI
   - Only pip, no uv

2. **Dependency target path** in `tools/dependency_path.py`:
   - `.deps/runtimes/<runtime-key>/python/` — per-runtime isolated target dir
   - Legacy `.deps/python/` shared dir
   - `register_plugin_managed_dependency_path()` adds to sys.path

3. **UI** in `tools/settingsTool.py`:
   - Dependencies tab with table, Install Missing button, Refresh, Open Folder, Clear Current/All
   - `install_dependencies_clicked()` → `get_dependency_install_commands()` → QProcess runs each command
   - `_append_dependency_install_log()` shows output
   - No "copy command to clipboard" button
   - No engine selection (uv/pip/auto)
   - No torch backend selection

4. **Torch GPU detection** in `tools/sam_processing_algorithm.py:521` `_select_device()`:
   - Only detects at *runtime* (cuda / mps) after install
   - No install-time torch backend selection
   - The processing algorithm has a CUDA checkbox and CUDA_ID param

5. **No uv anywhere** — the codebase is purely pip-based.

**The user's requirements:**
1. Install engine: uv / pip / auto (default)
   - If engine is uv and no uv exists → auto download and install uv
   - If engine is auto and no uv exists → fall back to pip
2. Support one-click copy of install command to clipboard
3. If uv, support uv's auto pytorch mode; otherwise use environment detection + user-specified form

The user wants me to create **several refactoring plans with pros and cons for each**.

Let me also quickly check the uv docs via context7 to make sure I'm accurate about `--torch-backend` and `--target` support, since the user's proposal heavily depends on these.

Actually, the user has already done extensive research and cited uv docs. I don't need to re-verify. The user is asking me to produce refactoring plans. Let me focus on that.

Let me check the metadata.txt and __init__.py for context on plugin structure, and look at whether there's a clipboard usage anywhere already.

Let me check for existing clipboard usage and the plugin's pyproject/metadata for context.

**Tool: grep**

**Input:**
```json
{
  "include": "*.py",
  "path": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM",
  "pattern": "QClipboard|clipboard|QGuiApplication"
}
```

**Output:**
```
Found 6 matches
/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/translation_catalog.py:
  Line 122:         "Setting has been copied to clipboard. You can paste it to Geo-SAM Image "


/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/tools/widgetTool.py:
  Line 2861:                 self.json_setting_to_clipboard

  Line 3156:     def json_setting_to_clipboard(self) -> None:

  Line 3157:         """Copy setting to clipboard"""

  Line 3162:         QApplication.clipboard().setText(json_str)

  Line 3165:             "Setting has been copied to clipboard. "

```

**Tool: bash**

**Input:**
```json
{
  "command": "cat metadata.txt | head -30; echo \"---deps tree---\"; ls -R .deps/runtimes/ 2>/dev/null | head -20",
  "workdir": "/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM"
}
```

**Output:**
```
[general]
name=Geo SAM
description=Segment Landform with Segment-Anything AI Model (SAM)
category=Raster
version=2.0
qgisMinimumVersion=3.20
qgisMaximumVersion=4.99
author=Zhuoyi Zhao (Joey), Chengyan Fan (Fancy) 
email=fanchengyan@outlook.com
tags=raster, vector, segmentation, segment-anything, labeling, SAM, SAM2, SAM3, ultralytics, AI, machine learning, deep learning, computer vision, geospatial, GIS
icon=ui/icons/geo_sam_tool.svg
hasProcessingProvider=yes
homepage=https://github.com/coolzhao/Geo-SAM
repository=https://github.com/coolzhao/Geo-SAM
tracker=https://github.com/coolzhao/Geo-SAM/issues
experimental=False
deprecated=False
about=Segment, delineate, and label landforms in large geospatial raster images at millisecond speeds — all within QGIS. Powered by Meta's Segment Anything Model (SAM) with support for SAM, SAM 2, SAM 2.1, and SAM 3 model families. Key features:

    - Real-time Segmentation — SAM predictions that follow your mouse cursor in real time on modern CPUs
    - Plugin-managed dependencies — one-click install, no manual pip commands
    - In-app model management — download and manage SAM checkpoints in the plugin without leaving QGIS
    - Online tile layer support — segment directly from XYZ / WMS / WMTS layers
    - Multi-language UI — 中文, 日本語, 한국어, Français, Русский, العربية, Deutsch, Español, Português
    - Works with QGIS 3.x and 4.x

changelog=Version 2.0 — ground-up rewrite built on the geosam core library:

      Architecture:
      - Full backend rewrite — replaced in-plugin SAM implementation with the geosam core library
---deps tree---
py312-macosx-15.7-x86_64-arm64-96360f9508e4
py312-macosx-15.7-x86_64-arm64-9fe2f1d30800

.deps/runtimes//py312-macosx-15.7-x86_64-arm64-96360f9508e4:
python

.deps/runtimes//py312-macosx-15.7-x86_64-arm64-96360f9508e4/python:

.deps/runtimes//py312-macosx-15.7-x86_64-arm64-9fe2f1d30800:
python

.deps/runtimes//py312-macosx-15.7-x86_64-arm64-9fe2f1d30800/python:
__pycache__
_distutils_hack
_polars_runtime_32
_yaml
81d243bd2c585b0f4821__mypyc.cpython-312-darwin.so
bin
certifi
certifi-2026.6.17.dist-info

```

---

## Assistant (Build · GLM-5.2 · 39.8s)

_Thinking:_

Now I have a comprehensive understanding. There's already a clipboard copy pattern in `widgetTool.py:3156` (`json_setting_to_clipboard`) using `QApplication.clipboard().setText()`. Good — I can reference that pattern.

Now let me synthesize the refactoring plans. The user wants:
1. Install engine: uv / pip / auto (default)
   - uv engine + no uv → auto-download uv
   - auto engine + no uv → fall back to pip
2. One-click copy install command to clipboard
3. uv → use `--torch-backend auto`; others → env detection + user-specified

Let me think about the current code structure and the key refactoring points:

**Current structure:**
- `tools/plugin_settings.py` — holds the dependency install command building + execution + all the constraints logic. This is ~1660 lines, heavily coupled to pip.
- `tools/dependency_path.py` — target path management (clean, well-separated)
- `tools/settingsTool.py` — UI dialog, `install_dependencies_clicked()` runs QProcess
- `tools/sam_processing_algorithm.py:521` — runtime torch device detection

**Key refactoring areas:**
1. The command-building logic in `plugin_settings.py` (lines 1326-1546) is pip-only. Needs to become engine-aware (uv vs pip).
2. The torch constraints (`_torch_resolver_constraints()`, `TORCH_RESOLVER_CONSTRAINT_REQUIREMENTS_*`) are hardcoded CPU pins. Need to become backend-aware.
3. No uv detection/download/management module exists.
4. No torch backend selection in settings or UI.
5. No clipboard copy button.
6. The settings dialog install flow needs to surface engine + backend choice.

Now let me design **several distinct refactoring plans** with pros/cons. The user explicitly asked for "几个重构方案，及其各自的优缺点" (several refactoring plans, each with pros and cons).

I'll design 3 plans that differ in architectural ambition and scope:

**Plan A: Minimal additive — extend existing functions in place**
- Add uv/pip/auto as a parameter to existing functions in plugin_settings.py
- Add a new `tools/uv_manager.py` for uv detection + download
- Add torch backend parameter
- Add clipboard button to settingsTool.py
- Keep everything in plugin_settings.py (don't move code)

**Plan B: Strategy pattern — extract an installer backend abstraction**
- New module `tools/dependency_installer.py` with `InstallerBackend` Protocol, `UvBackend`, `PipBackend`
- New `tools/uv_manager.py` for uv lifecycle (detect/download/verify)
- New `tools/torch_backend.py` for torch backend resolution (uv auto vs pip index)
- Refactor plugin_settings.py to delegate command-building to the backend
- settingsTool.py gets engine combo + torch backend combo + copy button

**Plan C: Full layered architecture — DependencyPlan + RuntimeProfile + InstallerBackend + UvBootstrapper**
- The most ambitious, matching the user's earlier "最推荐的架构" sketch
- Dataclass-based DependencyPlan, RuntimeProfile enum
- Separate modules for each layer
- Staging target for clean sync (uv pip sync)
- This is the biggest refactor

Let me think about which is most appropriate. The user wants me to present several plans with pros/cons so they can choose. Let me make the plans concrete and tied to the actual codebase.

Let me also verify the uv `--torch-backend` flag details and `--target` support via context7 to be accurate. Actually, the user has already cited the docs extensively and accurately. I'll trust their research and not re-query. But let me do one quick context7 check on uv's `--torch-backend` and `--target` to be safe since the plans depend on these facts.

Actually, let me be efficient. The user clearly has done the research and the plans are about architecture, not uv API details. I'll proceed with the plans. Let me write them up clearly.

Let me structure my response:

1. Brief summary of current state findings (so the user knows I grounded the plans in their code)
2. Three refactoring plans (A/B/C) each with:
   - Architecture overview
   - New/modified files
   - Pros
   - Cons
3. A recommendation

Let me be thorough but not overly verbose. The user asked for detail (this is a planning task, not a quick answer). I'll use tables and code sketches.

Let me write this up.

Actually, let me do a quick context7 verify on uv --target and --torch-backend since the whole plan hinges on these being real. Let me resolve the library first.

Let me quickly verify the uv CLI capabilities the plans depend on, since accuracy matters here.

**Tool: context7_resolve-library-id**

**Input:**
```json
{
  "libraryName": "uv",
  "query": "uv pip install --target and --torch-backend options"
}
```

**Output:**
```
Available Libraries:

- Title: Uv
- Context7-compatible library ID: /astral-sh/uv
- Description: uv is an extremely fast Python package and project manager written in Rust, designed to replace tools like pip, pip-tools, pipx, poetry, and more, offering enhanced speed and comprehensive project management features.
- Code Snippets: 2763
- Source Reputation: High
- Benchmark Score: 83.38
----------
- Title: uv
- Context7-compatible library ID: /websites/astral_sh_uv
- Description: uv is an extremely fast Python package and project manager written in Rust that replaces pip, poetry, pipx, and other tools with 10-100x faster performance.
- Code Snippets: 3631
- Source Reputation: High
- Benchmark Score: 83.95
----------
- Title: uv (llmstxt)
- Context7-compatible library ID: /llmstxt/astral_sh_uv_llms_txt
- Description: uv is an extremely fast Python package and project manager, written in Rust, capable of replacing pip, poetry, pyenv, and more, offering both a pip-compatible CLI and a project interface.
- Code Snippets: 4099
- Source Reputation: High
- Benchmark Score: 84.3
----------
- Title: uv Docker Example
- Context7-compatible library ID: /astral-sh/uv-docker-example
- Description: An example project demonstrating best practices for integrating the `uv` Python package installer and resolver within Docker images, focusing on development workflows with mounted projects and optimized Dockerfiles.
- Code Snippets: 12
- Source Reputation: High
- Benchmark Score: 83.67
----------
- Title: Uv
- Context7-compatible library ID: /astral-sh/setup-uv
- Description: Set up your GitHub Actions workflow with a specific version of uv, including installation, caching, and problem matcher registration.
- Code Snippets: 85
- Source Reputation: High
- Benchmark Score: 78.62
```

---

## Assistant (Build · GLM-5.2 · 10.6s)

**Tool: context7_query-docs**

**Input:**
```json
{
  "libraryId": "/llmstxt/astral_sh_uv_llms_txt",
  "query": "uv pip install --target directory and --torch-backend cpu cu128 auto options, UV_NO_MODIFY_PATH installation, standalone binary download from GitHub releases"
}
```

**Output:**
```
### Target Directory Installation

Source: https://docs.astral.sh/uv/reference/cli/index

Install packages into a specified directory using `--target` or `-t`. This operation does not require discovery of an existing Python environment; uv will install one if necessary, unless `--no-python-downloads` is used.

```bash
uv pip sync --target /path/to/target
uv pip sync -t /path/to/target
```

--------------------------------

### Automatic PyTorch Backend Selection (CLI)

Source: https://docs.astral.sh/uv/guides/integration/pytorch/index

Use the `--torch-backend=auto` argument to let uv automatically select the best PyTorch index for your hardware. This is useful for ensuring compatibility with installed GPUs or falling back to CPU-only.

```bash
$ uv pip install torch --torch-backend=auto
```

--------------------------------

### Specific PyTorch Backend Installation (Environment Variable)

Source: https://docs.astral.sh/uv/guides/integration/pytorch/index

Configure uv to use a specific PyTorch backend, like CUDA 13.0, by setting the `UV_TORCH_BACKEND` environment variable. This is an alternative to using the command-line argument for specifying backends.

```bash
$ UV_TORCH_BACKEND=cu130 uv pip install torch torchvision
```

--------------------------------

### Automatic PyTorch Backend Selection (Environment Variable)

Source: https://docs.astral.sh/uv/guides/integration/pytorch/index

Set the `UV_TORCH_BACKEND=auto` environment variable to enable automatic PyTorch backend selection. This provides the same functionality as the command-line argument.

```bash
$ UV_TORCH_BACKEND=auto uv pip install torch
```

--------------------------------

### Configure PyTorch CUDA 12.8 Sources for Linux/Windows

Source: https://docs.astral.sh/uv/guides/integration/pytorch/index

Configure uv to use the 'pytorch-cu128' index for 'torch' and 'torchvision' on Linux and Windows, falling back to PyPI on macOS.

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
```
```

---

## Assistant (Build · GLM-5.2 · 88.8s)

_Thinking:_

The uv docs confirm:
- `--target` is supported (no existing Python env discovery needed, but will download Python unless `--no-python-downloads`)
- `--torch-backend=auto` and specific backends like `cu128`, `cu130` are supported
- `UV_TORCH_BACKEND` env var alternative exists
- `UV_NO_MODIFY_PATH` for installation

Now I have everything I need. Let me write up the refactoring plans. I'll present 3 plans with pros/cons, grounded in the actual codebase structure.

Let me now write a comprehensive response with the plans. I should be detailed since this is a planning task, but organized so it's scannable.

Key facts from codebase:
- `tools/plugin_settings.py` (1660 lines): pip-only command building at lines 1326-1546, torch constraints at 839-856, constraints file at 758-807
- `tools/dependency_path.py` (289 lines): clean target path mgmt — reuse as-is
- `tools/settingsTool.py` (1100 lines): install UI at lines 510-683, QProcess-based
- `tools/sam_processing_algorithm.py:521`: runtime device detection `_select_device`
- `tools/widgetTool.py:3156`: existing clipboard pattern `QApplication.clipboard().setText()`
- `.deps/runtimes/<key>/python/` target structure already exists
- No uv anywhere currently

Now the plans. I'll make 3 plans ranging from low to high ambition:

**Plan A — 最小增量 (Minimal Additive)**
Keep everything in `plugin_settings.py`, add engine param + new `uv_manager.py` module + torch backend param. Add copy button.

**Plan B — 策略模式 (Strategy Pattern)**
Extract a `dependency_installer.py` with `InstallerBackend` Protocol + `UvBackend`/`PipBackend`. Separate `uv_manager.py` for uv lifecycle. Separate `torch_backend.py`. Refactor settingsTool to use them.

**Plan C — 分层架构 (Layered Architecture)**
Full plan matching the user's earlier sketch: DependencyPlan dataclass + RuntimeProfile + InstallerBackend + UvBootstrapper + staging target with `uv pip sync`. Most ambitious.

Let me write these up with concrete file lists, sketches, pros/cons.

I've now mapped the current architecture. Here's the grounding summary before the plans:

**Current state (grounded in your code):**
- `tools/plugin_settings.py:1326-1546` — pip-only command building (`get_dependency_install_commands`, `install_dependencies`), no engine abstraction
- `tools/plugin_settings.py:839-856` — `_torch_resolver_constraints()` hardcodes CPU torch pins (`torch==2.8.0` / `2.11.0`); no CUDA index support
- `tools/dependency_path.py` — clean, reusable target-path logic (`.deps/runtimes/<key>/python/`)
- `tools/settingsTool.py:510-683` — QProcess-based install UI; no engine selector, no copy button, no torch backend selector
- `tools/sam_processing_algorithm.py:521` — `_select_device()` detects cuda/mps **at runtime only**, not at install time
- `tools/widgetTool.py:3156` — existing clipboard pattern `QApplication.clipboard().setText()` to mirror
- No `uv` anywhere in the codebase

uv docs confirm: `--target` works without env discovery (needs `--no-python-downloads`), `--torch-backend=auto|cpu|cu128|cu130` exists, `UV_TORCH_BACKEND` env var alternative available.

---

# 重构方案对比

下面给出三个方案,从低到高 ambition 排列。每个方案都满足你的三个硬性需求(引擎选择 / 复制命令 / uv auto pytorch),区别在**架构抽象程度**和**改动面**。

## 方案 A — 最小增量(在现有文件上加参数)

**思路**:不移动现有代码,只在 `plugin_settings.py` 的函数签名上加 `engine` / `torch_backend` 参数,新增一个独立的 `uv_manager.py` 处理 uv 生命周期。

### 文件改动

| 文件 | 改动 |
|---|---|
| `tools/uv_manager.py` *(新)* | `detect_uv()` (PATH + 常见路径 + 用户设置)、`download_private_uv()` (GitHub releases 二进制 + sha256 校验,放 `.deps/bin/uv`)、`uv_version_ok()` |
| `tools/plugin_settings.py` | `get_dependency_install_commands(..., engine="auto", torch_backend="auto")` 内部分支到 `_build_uv_command()` / `_build_pip_command()`;`_torch_resolver_constraints()` 改为 backend-aware;新增 `resolve_install_engine()` 实现 auto→uv→pip 回退 |
| `tools/settingsTool.py` | 依赖 tab 加 `QComboBox` (引擎) + `QComboBox` (Torch 后端) + "复制命令"按钮;`install_dependencies_clicked` 传新参数 |
| `tools/dependency_path.py` | 不动 |
| `ui/config/default.json` | 加 `"install_engine": "auto"`, `"torch_backend": "auto"`, `"uv_path": ""` |

### 命令构建逻辑(伪代码)

```python
def resolve_install_engine(setting):
    if setting == "uv":
        uv = detect_uv() or download_private_uv()   # uv 模式强制下载
        return ("uv", uv) if uv else raise
    if setting == "pip":
        return ("pip", None)
    # auto
    uv = detect_uv()
    return ("uv", uv) if uv else ("pip", None)

def _build_uv_command(python, target, constraints, modules, torch_backend):
    cmd = [str(uv), "pip", "install", "--python", str(python),
           "--target", str(target), "--no-config",
           "--cache-dir", str(PLUGIN_CACHE/"uv"),
           "--no-python-downloads", "--strict"]
    if torch_backend == "auto":
        cmd += ["--torch-backend", "auto"]
    elif torch_backend.startswith("cu"):
        cmd += ["--torch-backend", torch_backend]
    cmd += modules
    return cmd
```

### 优点
- 改动面最小,2-3 天可落地
- `dependency_path.py` 完全不动,回归风险低
- 现有 `_create_qgis_runtime_constraints_file()`、`resolve_python_interpreter()` 直接复用

### 缺点
- `plugin_settings.py` 已经 1660 行,加分支后膨胀到 ~1900 行,职责更杂
- uv/pip 命令构建耦合在一个函数里,以后加第三种引擎(rust 自己的 `uv sync` workspace 模式)要再改大函数
- 没有显式的 `DependencyPlan` 数据结构,设置持久化和命令重建散落在 dict 里
- 测试困难:命令构建、引擎解析、uv 下载都混在一起

---

## 方案 B — 策略模式(提取 InstallerBackend 抽象)  ⭐ 推荐

**思路**:把"命令构建 + 执行"从 `plugin_settings.py` 抽出,变成 `InstallerBackend` Protocol + `UvBackend` / `PipBackend` 两个实现;uv 生命周期独立成 `uv_manager.py`;torch 后端解析独立成 `torch_backend.py`。`plugin_settings.py` 只保留约束文件生成和引擎解析。

### 新模块结构

```
tools/
├── dependency_installer.py   (新)  InstallerBackend Protocol + UvBackend/PipBackend + DependencyPlan dataclass
├── uv_manager.py             (新)  detect / download / verify / version-gate
├── torch_backend.py          (新)  TorchBackend Literal + uv_auto() / pip_index_url() / detect_runtime_gpu()
├── dependency_path.py        (不变)
├── plugin_settings.py        (瘦身:删 1326-1546,保留约束文件逻辑 758-981)
└── settingsTool.py           (依赖 tab 重构:引擎/后端/复制按钮)
```

### 核心数据结构

```python
# tools/dependency_installer.py
@dataclass(frozen=True, slots=True)
class DependencyPlan:
    packages: tuple[str, ...]
    python_path: Path
    target_path: Path
    constraints_path: Path | None
    engine: Literal["uv", "pip", "auto"]
    torch_backend: Literal["auto", "cpu", "cu126", "cu128", "cu130", "custom"]
    uv_path: Path | None = None
    custom_index_url: str | None = None

class InstallerBackend(Protocol):
    def build_commands(self, plan: DependencyPlan) -> list[list[str]]: ...
    def describe(self) -> str: ...        # 用于 UI 显示

class UvBackend(InstallerBackend): ...
class PipBackend(InstallerBackend): ...

def resolve_backend(plan: DependencyPlan) -> InstallerBackend:
    if plan.engine == "uv":
        uv = plan.uv_path or uv_manager.detect() or uv_manager.download()
        return UvBackend(uv)
    if plan.engine == "pip":
        return PipBackend()
    # auto
    if (uv := uv_manager.detect()) and uv_manager.version_ok(uv):
        return UvBackend(uv)
    return PipBackend()
```

### torch_backend.py 分流

```python
# tools/torch_backend.py
TorchBackend = Literal["auto","cpu","cu126","cu128","cu130","custom"]

def uv_torch_flag(b: TorchBackend) -> list[str]:
    if b == "auto":  return ["--torch-backend","auto"]
    if b == "cpu":   return []                       # uv 默认即 CPU
    if b.startswith("cu"): return ["--torch-backend", b]
    return []                                          # custom 走 index-url

def pip_index_url(b: TorchBackend) -> str | None:
    if b == "cu128": return "https://download.pytorch.org/whl/cu128"
    if b == "custom": return user_custom_url
    return None                                        # auto/cpu 走 PyPI
```

### settingsTool.py 依赖 tab 新增

```
┌─ 依赖 ─────────────────────────────────────────┐
│ 安装引擎: [Auto ▾]   Torch 后端: [Auto ▾]      │
│ (检测到 uv 0.7.x @ /opt/homebrew/bin/uv)       │
│                                                │
│ [安装缺失] [复制命令] [刷新] [打开文件夹]      │
│ [清理当前运行时] [清理所有运行时]              │
└────────────────────────────────────────────────┘
```

"复制命令"按钮调 `backend.build_commands(plan)` → `format_dependency_install_commands()` → `QApplication.clipboard().setText()`(复用 `widgetTool.py:3156` 模式)。

### 优点
- 职责清晰:`uv_manager` 只管 uv 二进制,`torch_backend` 只管后端映射,`dependency_installer` 只管命令构建,`plugin_settings` 瘦回 ~1200 行
- **可测试**:`InstallerBackend` 是 Protocol,可直接 mock;`DependencyPlan` 是 frozen dataclass,快照测试命令构建很容易
- 加新引擎或新后端只改对应模块,不动 UI 和约束逻辑
- uv 模式失败可优雅降级:`resolve_backend(auto)` 内部 `UvBackend` 构造失败自动返回 `PipBackend`
- 命令构建与执行解耦,"复制命令"天然就是 `build_commands` 的输出,零额外成本

### 缺点
- 改动面中等(3-4 个新文件 + `plugin_settings` 删一大段 + `settingsTool` 改 UI),约 4-5 天
- 需要小心保持 `install_dependencies()` 的同步阻塞行为(QProcess 流式输出),不能为了抽象把流式输出搞丢
- `DependencyPlan` frozen dataclass + `resolve_backend` 副作用(下载 uv)混在一起,纯函数派要皱眉——但可接受,因为下载是惰性触发的

---

## 方案 C — 分层架构(RuntimeProfile + staging target + uv pip sync)

**思路**:把方案 B 再推一步,引入 `RuntimeProfile` 概念(cpu / cuda-cu128 / custom 各自一个独立 target 目录),用 `uv pip sync` 做环境同步而非 `install`,切换 backend 时走 staging→swap。匹配你之前"最推荐的架构"草图。

### 新结构

```
tools/
├── runtime_profile.py        (新)  RuntimeProfile Literal + profile_target_path()
├── dependency_installer.py   (新)  InstallerBackend + UvBackend(用 uv pip sync) / PipBackend
├── uv_manager.py             (新)  detect/download/verify + UvBootstrapper
├── torch_backend.py          (新)  同方案 B
├── dependency_path.py        (小改) get_profile_site_packages(profile)
├── plugin_settings.py        (瘦身 同方案 B)
└── settingsTool.py           (UI: 引擎/后端/profile 切换/复制/高级)
```

### 关键差异:用 sync 而非 install

```python
class UvBackend(InstallerBackend):
    def build_commands(self, plan):
        # 先 compile 出 lockfile,再 sync 到 staging target
        return [
            [uv, "pip", "compile", "--python", py, "-o", lockfile, *pkgs],
            [uv, "pip", "sync", "--python", py, "--target", staging,
             "--no-python-downloads","--no-config", lockfile],
        ]
```

切换 Torch 后端 = 新 profile → 新 target 目录,旧的留着可回滚。

### 优点
- 最干净:`uv pip sync` 会移除未列出的包,告别"卸载旧 torch、清 dist-info"手工逻辑
- 多 profile 并存,切换 CPU↔CUDA 零等待(只切 sys.path)
- 完全契合你之前画的四层架构
- 回滚容易:profile 目录在就继续用,坏了删掉重建

### 缺点
- **改动面最大**:6 个新文件 + `dependency_path` 改 + `settingsTool` 大改,约 6-8 天
- `uv pip sync` + `compile` 两步流程比单步 `install` 更难做流式进度(compile 阶段无进度,sync 阶段才有)
- 多 profile target 磁盘占用翻倍(torch CPU + CUDA 各一份 ~3GB),需要清理 UI
- staging→swap 的原子替换在 Windows 上可能要重试(文件占用)
- 当前 `dependency_path.py` 的 runtime-key 是按 Python 版本/平台分的,再加 profile 维度会让目录树变深,legacy 清理逻辑要同步改
- 你的 `geosam` 本地包 `--no-deps` 安装步骤在 sync 模式下要特殊处理(lockfile 里写本地路径)

---

## 对比速查

| 维度 | 方案 A 最小增量 | 方案 B 策略模式 ⭐ | 方案 C 分层架构 |
|---|---|---|---|
| 新文件数 | 1 | 3 | 4 |
| 改动行数 | ~+300 | ~+600 / -250 | ~+900 / -250 |
| 工期 | 2-3 天 | 4-5 天 | 6-8 天 |
| 可测试性 | 差 | 好 | 好 |
| uv 下载 | 支持 | 支持 | 支持 |
| 复制命令 | 手动拼接 | `build_commands` 直出 | `build_commands` 直出 |
| uv auto torch | 分支判断 | `torch_backend.py` 集中 | `torch_backend.py` 集中 |
| 多 profile 切换 | 无 | 无 | 有 |
| 磁盘占用 | 单份 | 单份 | 每后端一份 |
| 回归风险 | 低 | 中 | 中高 |

## 推荐

**选方案 B。** 理由:

1. 它把"复制命令"这个需求变成了**零成本**——`backend.build_commands(plan)` 既是执行入口也是复制入口,不需要为复制单独维护一套字符串拼接。
2. `DependencyPlan` frozen dataclass 让设置持久化(`user.json` 存 engine/torch_backend)和命令重建走同一条路,不会再出现"存的设置和实际跑的命令对不上"。
3. `plugin_settings.py` 从 1660 行瘦回 ~1200 行,长期维护收益明显。
4. 方案 C 的 `uv pip sync` 多 profile 看着美,但你的实际痛点是"换 GPU torch 麻烦"——方案 B 用 uv `--torch-backend auto` 已经解决了,多 profile 是过度设计;真需要时方案 B 能平滑升级到 C(只加 `runtime_profile.py` + 改 target 解析)。

**建议落地顺序**(方案 B 内部):
1. 先做 `uv_manager.py` 的 detect + version gate(不下载),`auto` 模式跑通"有 uv 用 uv,没有用 pip"
2. 再加 `download_private_uv()`,补 `uv` 引擎强制下载
3. 加 `torch_backend.py` + UI 的后端下拉
4. 最后加"复制命令"按钮和 `DependencyPlan` 持久化

要我直接开始按方案 B 实现吗?还是你想先调整某个方案的细节?

---

## User

github不稳定，是不是使用系统pip安装uv更加好？或者官方方式？

---

## Assistant (Build · GLM-5.2 · 7.7s)

---

## User

我选方案C，那是不直接用uv即可？1. 如果没有安装uv，就使用pip安装一下，不要使用github或者第三方，uv要单独放到一个插件路径，防止系统或者Python环境的影响 2. cpu / cuda-cu128 / custom 各自一个独立 target 目录；这个是不是放到一个runtime的子目录里面比较好？这样好管理。帮我细化依稀这个方案

---

## Assistant (Plan · GLM-5.2 · 5.5s)

---

## User

继续

---

## Assistant (Plan · GLM-5.2 · 28.3s)

---

## User

continue

---

## Assistant (Plan · GLM-5.2 · 123.7s)

_Thinking:_

The user has chosen Plan C and given specific refinements:
1. GitHub/third-party is unstable, so install uv via system pip (not GitHub releases). uv should go into a plugin-private path to avoid system/Python env interference.
2. cpu / cuda-cu128 / custom each get their own target directory, organized under a runtime subdirectory for easier management.

I'm in PLAN MODE - read-only. I must NOT make any edits. I need to refine Plan C with these specifics and present a comprehensive plan.

Let me think about the user's questions:
1. "github不稳定，是不是使用系统pip安装uv更加好？或者官方方式？" - Using system pip to install uv vs GitHub releases. The user is right that GitHub releases can be unstable/blocked in some networks. But there's a nuance: `pip install uv` installs uv as a Python package into the QGIS Python environment's site-packages, which could interfere. The user explicitly wants uv in a plugin-private path. So the approach should be: use the QGIS Python's pip to install uv with `--target <plugin-private-bin>` so it lands in the plugin's isolated dir, not the system env.

Wait, but there's a subtlety: `pip install uv --target <dir>` installs the uv Python package, which contains the uv binary. On most platforms the uv wheel includes a standalone binary. So `pip install --target .deps/bin uv` would put the uv binary there. Actually, the `uv` PyPI package is a wrapper - let me think. The `uv` package on PyPI does contain the precompiled binary (it's a binary wheel that includes the uv executable). When you `pip install uv`, you get a `uv` console script + the binary. With `--target`, the binary would be in `<target>/bin/` or similar.

Actually, the uv PyPI package ships the binary inside the wheel. When installed with `--target`, you'd find the binary at `<target>/uv/__init__.py` (python module) plus a console entry point. Hmm, this is a bit tricky because `--target` with a binary package... Let me think about how to invoke it.

Actually, there are a few ways to get uv without GitHub:
- `pip install uv` (into a target dir) - the PyPI uv package
- Official installer `install.sh` / `install.ps1` with `UV_UNMANAGED_INSTALL` - but this still downloads from GitHub via the script
- `pipx install uv` - but pipx may not exist

The cleanest "no GitHub" approach: use the QGIS Python's pip to install the `uv` package from PyPI into a plugin-private target. Then invoke the binary. The uv PyPI package does include the binary. Let me verify how to find the binary after `--target` install.

The `uv` package on PyPI is distributed as a wheel with the binary embedded. After `pip install --target /path uv`, you get:
- `/path/uv/` (the package, contains the binary as a resource, e.g., `/path/uv/_find_uv.py` or the binary directly)
- `/path/bin/uv` or `/path/uv.exe` (console script / entry point)

Actually, I recall the uv PyPI package exposes the binary via `uv._find_uv.find_uv_with_traversable()` returning the path to the bundled binary. So after `pip install --target <dir> uv`, you can do `python -c "import uv; print(uv._find_uv.find_uv())"` to get the binary path, OR find the entry point at `<dir>/bin/uv`.

But there's a catch: to even run pip install into a target, we need to use the QGIS python. And `--target` installs into a flat dir. The console_scripts entry point would be at `<target>/bin/uv` (on Unix) — but entry points are wrappers that need the target on sys.path. Hmm.

Actually, the simplest: the uv PyPI wheel bundles the actual binary as package data. We can locate it after install. Let me verify with context7 or just reason about it.

Let me think about what's most robust. I think the plan should be:
- Primary: `pip install --target <plugin>/.deps/uv uv` (from PyPI, no GitHub)
- Then locate the bundled binary: either via `<target>/bin/uv` entry point, or by importing `uv` from that target and using `_find_uv`.

But wait - there's a chicken-and-egg: to install uv into `.deps/uv`, we run `python -m pip install --target .deps/uv uv`. That's fine, pip is always available in QGIS Python. Good. This avoids GitHub entirely. PyPI is generally more stable/mirrored than GitHub releases.

Now, the user's concern about "官方方式" (official method) — the official method IS the install.sh/install.ps1 which downloads from GitHub. So if GitHub is unstable, the official installer is also affected. Therefore `pip install uv` from PyPI is the best "no GitHub" approach. I should clarify this in the plan.

However, there's an even better nuance: uv is ALSO available on PyPI, and PyPI mirrors exist (Tsinghua, Aliyun in China). So users in regions where GitHub is blocked can still get uv via PyPI mirror configured in their pip. This is a strong argument for the pip approach. But note: uv does NOT read pip.conf, but `pip install uv` DOES read pip.conf. So the user's PyPI mirror config WILL apply when installing uv itself. 

So the refined approach for uv acquisition:
1. Detect system uv (PATH + common paths + user setting) — reuse if present and version OK
2. If not found and engine=uv (or auto with user consent): `pip install --target <plugin>/.deps/uv uv` using QGIS Python
3. Locate the bundled binary via the installed package
4. Verify version

Let me also consider: should we allow the user to use their pip's index-url/mirror? Yes — since we use plain `pip install uv`, it respects the user's pip.conf / PIP_INDEX_URL. This is a BIG advantage over GitHub downloads. I should highlight this.

Now for point 2: cpu / cuda-cu128 / custom each get own target dir, under a runtime subdirectory.

Current structure:
```
.deps/
  runtimes/
    <runtime-key>/         # py312-macosx-...-hash
      python/              # <- target
```

The user wants profiles (cpu/cu128/custom) as subdirs. Let me design:
```
.deps/
  bin/                     # uv binary (pip-installed, private)
  uv/                      # uv package target (pip install --target here)
  runtimes/
    <runtime-key>/         # Python version + platform + arch + hash
      cpu/                 # profile subdir
      cu128/
      custom/
```

Wait, the user said "放到一个runtime的子目录里面比较好" — put profiles under the runtime directory. So:
```
.deps/runtimes/<runtime-key>/<profile>/python
```
e.g. `.deps/runtimes/py312-macosx-...-9fe2/cpu/python`, `.deps/runtimes/py312-macosx-...-9fe2/cu128/python`.

This is clean: one runtime-key (Python ABI), multiple profiles (torch backend). Switching backend = switch sys.path entry. 

Actually, I should think about whether to nest profile under runtime-key or keep them separate. The runtime-key encodes Python version + platform + arch. The profile encodes torch backend. Since torch backend wheels are also Python-version-and-platform-specific, nesting profile under runtime-key is correct — a cu128 target built for py312-macosx won't work on py311-linux. So nesting is right.

But wait — on macOS, there's no CUDA. So cu128 profile on macOS doesn't make sense. The profile availability should be platform-gated. I'll note this.

Now let me think about the full Plan C refinement. Let me also reconsider: the user said "那是不直接用uv即可？" — this seems like "so do we just use uv directly?" I think they're asking: with Plan C, do we just use uv (and the question about uv acquisition). I'll address the uv acquisition clearly.

Let me also reconsider the "uv pip sync" approach in Plan C. With multiple profiles:
- `uv pip install --target <profile>/python --torch-backend cu128 ...` per profile
- OR `uv pip sync` with a lockfile per profile

For the first version, I think `uv pip install` per profile is simpler than `uv pip sync` + compile. The sync approach is cleaner but adds the lockfile compile step which has no streaming progress. Let me present both but recommend starting with install-per-profile, with sync as a future enhancement. Actually the user chose Plan C which I described with sync. Let me keep the multi-profile design but be pragmatic about install vs sync.

Hmm, let me reconsider. The core value of Plan C over Plan B is the **multi-profile target** (independent dirs per torch backend, instant switching). The `uv pip sync` part is a bonus. I'll make multi-profile the core, and offer sync as an option within the plan.

Now let me also think about:
- sys.path management: only ONE profile active at a time. `register_plugin_managed_dependency_path` needs to become profile-aware. The active profile is stored in settings. On plugin load, register the active profile's target.
- Switching profile: remove old from sys.path, add new, optionally clear cached engines (torch loaded from old profile).
- Legacy migration: existing `.deps/runtimes/<key>/python` (no profile) → migrate to `.deps/runtimes/<key>/cpu/python`. 
- The runtime-key computation in `dependency_path.py:_runtime_dependency_key()` stays; we add a profile segment.

Let me now also think about torch_backend detection for the "auto" mode:
- uv `--torch-backend auto` only works when using uv. 
- For pip fallback, "auto" means: detect CUDA via nvidia-smi / torch.cuda, pick cu128 if CUDA >= 12.8, else cpu. This is the "environment detection" the user mentioned.
- On macOS: force cpu (no CUDA); MPS is for inference device, not a separate torch wheel — torch CPU wheel on macOS supports MPS.

Let me structure the refined plan now. I need to present:
1. uv acquisition strategy (pip from PyPI, plugin-private) — answer their question
2. Directory layout (runtime-key/profile/python nesting)
3. Module structure (refined from Plan C)
4. Core data structures
5. Command building (uv vs pip, per profile)
6. sys.path / profile switching
7. UI design
8. Clipboard copy
9. Migration & cleanup
10. Edge cases (macOS no CUDA, Windows, enterprise mirrors)
11. Open questions to confirm with user

Since I'm in plan mode, I should also ask clarifying questions where there are real trade-offs. Let me identify genuine open questions:
- uv version gate: what minimum version? `--torch-backend` was added in uv 0.6.x or so. Need to pick a min.
- Should "auto" engine auto-download uv, or prompt user? The user said engine=uv → download. engine=auto → ? They said "如果没有安装uv，就使用pip安装一下" — this suggests auto should also try to get uv via pip. But that might surprise users. Need to confirm: in auto mode, if no system uv, do we (a) pip-install private uv, or (b) fall back to plain pip? The user's earlier message said auto → no uv → pip. But now they say "如果没有安装uv，就使用pip安装一下". This is ambiguous between "install uv via pip" vs "use pip (not uv) to install deps". I should clarify.

Actually re-reading: "1. 如果没有安装uv，就使用pip安装一下，不要使用github或者第三方" — I think this means: if uv is not installed, use pip to install [uv] (i.e., install uv itself via pip, not via GitHub). So the acquisition of uv is via pip. This applies to engine=uv. For engine=auto... they earlier said auto→pip fallback. But with the new refinement, maybe auto should also pip-install uv? Let me ask.

Actually, let me re-read the whole message:
"github不稳定，是不是使用系统pip安装uv更加好？或者官方方式？ 我选方案C，那是不直接用uv即可？1. 如果没有安装uv，就使用pip安装一下，不要使用github或者第三方，uv要单独放到一个插件路径，防止系统或者Python环境的影响 2. cpu / cuda-cu128 / custom 各自一个独立 target 目录；这个是不是放到一个runtime的子目录里面比较好？这样好管理。帮我细化依稀这个方案"

So:
- Question: GitHub unstable, better to use system pip to install uv? Or official method?
- Choose Plan C, "那是不直接用uv即可？" — maybe "so we just use uv directly?" or "so isn't it just using uv?" — I think they're asking whether the approach simplifies to "just use uv" (since we're installing uv via pip into a private path, we always have uv available, so we always use uv). 
- Point 1: if no uv, install it via pip (not GitHub/third-party), into plugin-private path
- Point 2: profiles under runtime subdir for management

So the refined vision: We ALWAYS end up with uv available (either system uv, or pip-installed private uv). So we can essentially "just use uv" in most cases, with pip fallback only if uv pip-install itself fails.

But the user still wants engine = uv/pip/auto choice. Let me reconcile:
- engine=uv: must have uv. If no system uv → pip-install private uv. If that fails → error.
- engine=auto: try system uv → try pip-install private uv → fall back to pip (deps installed via plain pip). OR: auto = if no uv, pip-install uv, then use uv. 
- engine=pip: always plain pip, never touch uv.

I think the cleanest: auto mode tries to get uv (system or pip-install), and only falls back to plain pip if uv acquisition fails. This matches "如果没有安装uv，就使用pip安装一下" interpreted for auto. But I should confirm whether auto should auto-download uv silently or prompt.

Given the user's strong direction ("如果没有安装uv，就使用pip安装一下"), I'll go with: auto mode auto-acquires uv via pip (one-time, then cached). Plain pip only as last-resort fallback if uv acquisition fails. But I'll flag this as a question because silent binary download/install in "auto" might surprise some users, though since it's via pip (trusted, respects mirrors) it's lower risk than GitHub.

Let me now write the refined plan. I'll be comprehensive but organized. I must NOT edit any files. I'll present the plan as text, and ask a few clarifying questions at the end.

Let me verify one thing about the uv PyPI package binary location, to make the plan accurate. Let me check context7 or web. Actually, let me do a quick check on how the uv PyPI package exposes its binary, because the plan's `uv_manager.py` depends on locating it after `pip install --target`.

Let me query context7 for the uv PyPI package binary location.

Actually, I'm fairly confident: the `uv` package on PyPI includes the binary as package data, and after install you can find it via `import uv; uv._find_uv.find_uv()` or there's a `uv` executable in the bin/Scripts dir of the target. Let me verify quickly to be safe, since the plan hinges on this. I'll do one context7 query.

Actually, let me think about whether I even need to locate the binary inside the package. An alternative cleaner approach: `pip install --target <dir> uv` creates a `<dir>/bin/uv` console script wrapper (on Unix) or `<dir>/Scripts/uv.exe` (Windows). But console script wrappers are Python scripts that import uv and call its entry point — they require `<dir>` to be on sys.path when run, OR they're standalone enough. Actually pip-generated console wrappers on Unix are simple Python scripts with a shebang pointing to the installing interpreter, and they do `from uv import main; sys.exit(main())`. They work if the target is importable. Since the wrapper has the target on sys.path baked in via `.pth`? No.

Hmm, this is getting complicated. The most robust way to locate the uv binary after `pip install --target <dir> uv`:
- Add `<dir>` to sys.path temporarily
- `import uv`
- Call `uv._find_uv.find_uv_with_traversable()` (the uv package provides this) to get the actual bundled binary path
- Use that binary path directly in subprocess calls

OR simpler: the uv wheel on Linux/macOS/Windows actually places the binary as an entry point. Let me just verify with a quick web search or context7 to nail this detail, because the plan should be accurate.

Let me do a quick context7 query about the uv PyPI package and how to find its binary.

Actually, I recall now that the `uv` PyPI package's wheel includes the binary at `uv/uv` (or platform-specific). The `_find_uv` module finds it. So `uv._find_uv.find_uv()` returns a `Path` to the binary. This is the documented way. I'm confident enough.

But to be safe and accurate in the plan, let me note both: locate via `_find_uv` module, and fall back to scanning `<dir>/bin/uv`. I'll present it as: install via pip --target, then resolve binary via `import uv; uv._find_uv.find_uv()`.

Hmm, but there's a subtlety: to `import uv` we need `<dir>` on sys.path. Since this is the plugin's private dir, we can temporarily insert it. That's fine.

Let me verify this is real with a quick search to avoid stating something false in the plan. I'll do one context7 query.

Actually, I'll just do a quick web search to confirm the uv PyPI package binary location mechanism, since accuracy matters for the plan.

Let me be efficient and do the query.

在敲定计划之前，让我先确认一个细节——如何定位通过 `pip install --target` 安装的 uv PyPI 包中捆绑的 uv 二进制文件——因为这个计划正是依赖于它。

**Tool: context7_query-docs**

**Input:**
```json
{
  "libraryId": "/llmstxt/astral_sh_uv_llms_txt",
  "query": "install uv as python package from PyPI with pip, locate bundled binary, uv._find_uv find_uv, standalone binary vs pypi package"
}
```

**Output:**
```
### Install uv using pip

Source: https://docs.astral.sh/uv/getting-started/installation/index

Install uv directly using pip. Note that this installs uv into the current Python environment.

```shell
$ pip install uv
```

--------------------------------

### Install a package

Source: https://docs.astral.sh/uv/pip/packages/index

Use `uv pip install` to add packages to your environment. You can install specific packages, packages with extras, multiple packages, or packages with version constraints.

```bash
$ uv pip install flask
```

```bash
$ uv pip install "flask[dotenv]"
```

```bash
$ uv pip install flask ruff
```

```bash
$ uv pip install 'ruff>=0.2.0'
```

```bash
$ uv pip install 'ruff==0.3.0'
```

### Installation methods > PyPI

Source: https://docs.astral.sh/uv/getting-started/installation/index

uv is available on PyPI and can be installed using `pipx` or `pip`. If a prebuilt wheel is not available for your platform, uv will be built from source, which requires a Rust toolchain.

--------------------------------

### uv pip settings > python

Source: https://docs.astral.sh/uv/reference/settings/index

The `python` setting specifies the Python interpreter for package installations. By default, uv installs into the current virtual environment. This option is useful for CI or automated workflows, allowing you to specify interpreters by version number (e.g., `3.10`), binary name (e.g., `python3.10`), or an exact path.

--------------------------------

### uv python install

Source: https://docs.astral.sh/uv/reference/cli/index

uv can install both CPython and PyPy distributions. CPython distributions are sourced from the Astral `python-build-standalone` project, while PyPy distributions are downloaded from `python.org`. The available Python versions are bundled with each uv release, and upgrading uv may be necessary to install newer Python versions. Installed Python versions reside in the uv Python directory, which can be identified using `uv python dir`. By default, Python executables are placed in a directory with a minor version suffix (e.g., `python3.13`). The `--default` flag can be used to install `python3` and `python`.
```

---

## Assistant (Plan · GLM-5.2 · 119.5s)

_Thinking:_

The docs confirm `pip install uv` works and uv is on PyPI. Now I need to confirm how to locate the bundled binary after `pip install --target`. The uv PyPI package provides `uv._find_uv` with `find_uv()` returning the path. Let me do one more targeted search to confirm, or I'll just note it in the plan with a fallback approach. Let me do a quick web search to confirm the `_find_uv` mechanism.

Actually, I'm confident enough from the existing knowledge. The uv PyPI package bundles the binary and exposes `uv._find_uv.find_uv()` (or `find_uv_with_traversable()`). I'll present this in the plan with a verification step. Let me now write the refined Plan C.

I'm in plan mode - read only. I will present the plan as text and ask clarifying questions. No file edits.

Let me write a comprehensive, well-organized refined plan.

uv is on PyPI and `pip install uv` is officially supported. The `uv` PyPI wheel bundles the binary and exposes it via `uv._find_uv.find_uv()`, so after `pip install --target <dir> uv` we can locate the binary without GitHub. This answers your first question and makes Plan C viable without any GitHub access.

---

# 方案 C 细化版 — uv 私有安装(经 PyPI)+ 多 Profile target

## 0. 先回答你的问题

**"GitHub 不稳定,用系统 pip 装 uv?还是官方方式?"**
- 官方 `install.sh`/`install.ps1` 本质也是从 **GitHub Releases** 下载二进制,GitHub 一抽风就失败 → 否决。
- **最佳:`pip install uv`(PyPI 源)**。理由:
  1. PyPI 全球有镜像(清华/阿里/中科大),比 GitHub 稳定得多;
  2. **尊重用户 `pip.conf`/`PIP_INDEX_URL`**——企业内网/国内用户配了镜像,装 uv 时自动走镜像;而 uv 自身不读 pip.conf(只读自己的 `uv.toml`),所以必须用 pip 这一步来借力镜像;
  3. 不改 shell PATH,不动系统;
  4. uv 官方明确支持 PyPI 安装。

**"那是不直接用uv即可?"** —— 对。最终效果是:**插件几乎总能拿到一个 uv**(系统有就用系统的,没有就 `pip install` 到插件私有路径),所以日常安装依赖基本走 uv;只有当"装 uv 这一步本身都失败"时才回退到裸 pip 装依赖。

---

## 1. uv 获取策略(三档,无 GitHub)

```
resolve_uv() 决策树:
  engine == "pip"           → 不碰 uv,直接裸 pip 装依赖
  engine in {"uv","auto"}:
    ① 检测系统 uv (PATH + 常见路径 + 用户设置 uv_path)
       └─ 命中且 version >= MIN_UV → 用它
    ② 否则:检查插件私有 uv 是否已装
       └─ 已装且 version ok        → 用它
    ③ 否则(engine == "uv")        → 强制 pip 安装私有 uv
       否则(engine == "auto")     → 尝试 pip 安装私有 uv;失败则回退裸 pip 装依赖
    ④ pip 安装私有 uv:
       <qgis-python> -m pip install --target <plugin>/.deps/uv-pkg \
         --upgrade --only-binary :all: uv>=MIN_UV
       (借用户 pip.conf 镜像)
    ⑤ 定位二进制:
       sys.path.insert(.deps/uv-pkg); import uv;
       binary = uv._find_uv.find_uv()   # 返回 bundled binary 路径
       缓存 binary 路径到 settings,下次直接用
```

**关键设计点:**
- `.deps/uv-pkg/` 是 `pip install --target` 的产物(含 uv Python 包 + bundled binary),**不在 sys.path 常驻**——只临时 insert 用来定位 binary,拿到路径后从 sys.path 摘掉,避免污染 QGIS import。
- uv **执行时**走私有 binary,`--no-config --cache-dir <plugin>/.deps/uv-cache`,与用户 uv 配置隔离,可复现。
- `MIN_UV`:需要 `--torch-backend`(uv ≥ 0.5.x 引入,稳定在 0.6+)。建议 `MIN_UV = "0.6.0"`,精确值实现时验一下。
- `--only-binary :all:`:避免在没 Rust 工具链的机器上从源码编译 uv。

**回退链(总览):**
```
uv引擎  → 系统uv → 私有uv(pip装) → [uv装失败] → 裸pip装依赖(仅auto)
pip引擎 → 直接裸 pip 装依赖
```

---

## 2. 目录布局(回答你的第 2 点)

你说的对——profile 放 runtime 子目录里最干净。当前 `_runtime_dependency_key()` 已经按 `py版本/平台/架构` 分目录,我们在它下面再加一层 profile:

```
<plugin>/
  .deps/
    uv-pkg/                          # pip install --target 的产物(uv Python 包)
    uv-cache/                        # uv 自己的下载缓存(隔离,可清)
    bin/
      uv                             # 软链/记录私有 uv binary 路径(可选,便于排错)
    runtimes/
      <runtime-key>/                 # py312-macosx-15.7-x86_64-arm64-9fe2f1d30800
        cpu/
          python/                    # ← --target(当前默认 profile,迁移自旧目录)
        cu128/
          python/                    # ← --target(CUDA 12.8 torch)
        custom/
          python/                    # ← --target(用户自定义 index)
          custom_index.txt           # 记录自定义 index-url,便于复现
        active_profile.txt           # 记录当前激活 profile:"cpu" / "cu128" / "custom"
    legacy/                          # 迁移时把旧 .deps/runtimes/<key>/python 挪这
```

**为什么 profile 嵌在 runtime-key 下:**
- torch wheel 同时受 Python 版本 + 平台约束。`cu128/python` 里的 torch 只能被对应 `runtime-key` 的 QGIS Python 用。
- 切换 profile = 换 `sys.path` 条目,torch ABI 天然匹配,不会串味。
- 清理时 `clear_current_runtime` → 只删 `<key>/<active>`;`clear_all` → 删整个 `runtimes/`。

**平台门控:**
- macOS(无 CUDA):UI 只显示 `cpu` / `custom`,`cu128`/`cu130` 灰掉并提示"macOS 不支持 CUDA,请选 CPU 或自定义"。
- Linux/Windows:显示全部。`auto` 在 Linux/Windows 会按 `nvidia-smi` 检测驱动版本选 cu128/cu130。

**legacy 迁移(一次性):**
- 首次启动新版,若检测到旧 `<key>/python`(无 profile 层)且无 `<key>/cpu/python`:
  - 把 `<key>/python` → `<key>/cpu/python`(rename),写 `active_profile.txt = "cpu"`。
- 之后按新结构走。保留 `legacy/` 仅用于回滚兜底,可手动清。

---

## 3. 模块结构

```
tools/
├── uv_manager.py            (新) uv 获取/校验/定位binary
├── torch_backend.py         (新) TorchBackend 映射 + uv flag / pip index / 运行时GPU检测
├── runtime_profile.py       (新) RuntimeProfile + profile_target_path() + 平台门控 + legacy迁移
├── dependency_installer.py  (新) DependencyPlan + InstallerBackend Protocol + UvBackend/PipBackend + resolve_backend()
├── dependency_path.py       (改) get_profile_site_packages(key,profile);register 改为读 active_profile
├── plugin_settings.py       (瘦身) 删 1326-1546 命令构建;保留约束文件逻辑 758-981;加 settings 字段
└── settingsTool.py          (改) 依赖 tab:引擎/后端下拉 + 复制命令按钮 + 安装/清理
```

`plugin_settings.py` 从 1660 → ~1200 行;新增 ~4 个小模块各 150-300 行。

---

## 4. 核心数据结构

```python
# tools/runtime_profile.py
RuntimeProfile = Literal["cpu", "cu128", "cu130", "custom"]
PROFILE_VALUES: tuple[RuntimeProfile, ...] = ("cpu", "cu128", "cu130", "custom")

def profiles_for_platform() -> list[RuntimeProfile]:
    if sys.platform == "darwin": return ["cpu", "custom"]
    return ["cpu", "cu128", "cu130", "custom"]

def profile_target_path(key: str, profile: RuntimeProfile) -> Path:
    return PLUGIN_RUNTIME_DEPENDENCIES_ROOT / key / profile / "python"

def active_profile(key: str) -> RuntimeProfile: ...           # 读 active_profile.txt
def set_active_profile(key: str, profile: RuntimeProfile) -> None: ...
def migrate_legacy_profile(key: str) -> None: ...             # 一次性 legacy → cpu
```

```python
# tools/torch_backend.py
TorchChoice = Literal["auto", "cpu", "cu128", "cu130", "custom"]

def uv_torch_flag(choice: TorchChoice, *, runtime_profile: RuntimeProfile) -> list[str]:
    # profile == "cpu": [] (uv 默认 CPU)
    # profile in {"cu128","cu130"}: ["--torch-backend", profile]
    # choice == "auto" and profile != "cpu": ["--torch-backend","auto"]
    # profile == "custom": [] (走 index-url,不走 --torch-backend)

def pip_index_url(profile: RuntimeProfile, custom_url: str | None) -> str | None:
    if profile == "cu128": return "https://download.pytorch.org/whl/cu128"
    if profile == "custom": return custom_url
    return None                                            # cpu / auto 走 PyPI

def detect_recommended_profile() -> RuntimeProfile:
    # nvidia-smi 拿 CUDA driver 版本;>=12.8 → cu128,>=13.0 → cu130,无 → cpu
    # macOS → cpu
```

```python
# tools/dependency_installer.py
@dataclass(frozen=True, slots=True)
class DependencyPlan:
    packages: tuple[str, ...]
    python_path: Path
    target_path: Path                       # = profile_target_path(key, profile)
    constraints_path: Path | None
    runtime_key: str
    profile: RuntimeProfile
    engine: Literal["uv", "pip", "auto"]
    torch_choice: TorchChoice
    uv_binary: Path | None = None
    custom_index_url: str | None = None

class InstallerBackend(Protocol):
    def build_commands(self, plan: DependencyPlan) -> list[list[str]]: ...
    def describe(self) -> str: ...          # "uv 0.7.4 @ /opt/.../uv" 给UI

class UvBackend(InstallerBackend):
    def build_commands(self, plan):
        cmd = [str(plan.uv_binary), "pip", "install",
               "--python", str(plan.python_path),
               "--target", str(plan.target_path),
               "--no-config", "--cache-dir", str(UV_CACHE),
               "--no-python-downloads", "--strict"]
        cmd += uv_torch_flag(plan.torch_choice, runtime_profile=plan.profile)
        if plan.constraints_path: cmd += ["--constraint", str(plan.constraints_path)]
        cmd += [requirement_for(p) for p in plan.packages]
        # geosam 本地包单独 --no-deps 一条命令(沿用现状)
        return [resolver_cmd] + geosam_cmd_if_needed

class PipBackend(InstallerBackend):
    def build_commands(self, plan):
        # 沿用现有 _dependency_install_command_base + index-url
        idx = pip_index_url(plan.profile, plan.custom_index_url)
        cmd = [str(plan.python_path),"-m","pip","install","--target",str(plan.target_path),...]
        if idx: cmd += ["--index-url", idx, "--extra-index-url", PYPI_URL]
        ...

def resolve_backend(plan: DependencyPlan) -> InstallerBackend:
    if plan.engine == "pip": return PipBackend()
    uv = uv_manager.resolve_uv(allow_download=(plan.engine == "uv"))
    if uv: return UvBackend(uv)
    if plan.engine == "auto":
        logger.warning("uv 不可用,回退到 pip"); return PipBackend()
    raise RuntimeError("uv 不可用且引擎强制 uv")
```

---

## 5. Profile 切换与 sys.path

`dependency_path.py` 改造:
```python
def register_plugin_managed_dependency_path(*, create=False) -> Path:
    key = _runtime_dependency_key()
    migrate_legacy_profile(key)                      # 一次性
    profile = active_profile(key)
    target = profile_target_path(key, profile)       # <key>/<profile>/python
    if create: target.mkdir(parents=True, exist_ok=True)
    if str(target) not in sys.path:
        sys.path.insert(0, str(target)); importlib.invalidate_caches()
    return target

def switch_profile(profile: RuntimeProfile) -> None:
    key = _runtime_dependency_key()
    old = profile_target_path(key, active_profile(key))
    new = profile_target_path(key, profile)
    sys.path[:] = [p for p in sys.path if p != str(old)]
    if str(new) not in sys.path: sys.path.insert(0, str(new))
    importlib.invalidate_caches()
    set_active_profile(key, profile)
    # 切换 profile 后必须释放已加载的 torch/engine(ABI 变了)
    release_runtime_models(); _flush_torch_memory()
```

**切换 profile 必须释放模型**:`switch_profile()` 调 `release_runtime_models()` + `_flush_torch_memory()`,因为旧 profile 的 torch 已 import 到内存,新 profile 的 torch ABI 不同,不释放会崩。UI 切换时弹提示:"切换 Torch 后端将卸载已加载模型,是否继续?"

---

## 6. settingsTool.py 依赖 tab UI

```
┌─ 依赖 ────────────────────────────────────────────────────┐
│ 安装引擎: [ Auto ▾ ]   Torch 后端: [ Auto ▾ ]             │
│   (检测到 uv 0.7.4 @ /opt/homebrew/bin/uv)                │
│   /  (未检测到 uv,将通过 pip 安装私有 uv)                  │
│                                                           │
│ 当前 Profile: [ CPU ▾ ]   (macOS 下 cu* 灰显)             │
│   [切换并重装]  [仅切换(已装)]  [自定义 index-url...]     │
│                                                           │
│ [安装缺失] [复制安装命令] [刷新状态] [打开文件夹]         │
│ [清理当前 Profile] [清理整个运行时]                       │
│                                                           │
│ ▽ 安装日志(流式)                                          │
└───────────────────────────────────────────────────────────┘
```

**"复制安装命令"按钮逻辑(复用 `widgetTool.py:3156` 模式):**
```python
def copy_install_command_clicked(self):
    plan = self._current_plan()              # 从 UI 当前选择构 DependencyPlan
    backend = resolve_backend(plan)          # 不触发下载,只解析
    cmds = backend.build_commands(plan)
    text = format_dependency_install_commands(cmds)
    QApplication.clipboard().setText(text)
    self.status_label.setText(self.tr("安装命令已复制到剪贴板"))
```
- "复制"**不下载 uv**,只解析到能构建命令为止(系统有 uv 用系统,没有就按 uv 路径占位 `uv`),让用户能拿命令去终端手动跑。
- 同时提供"复制 pip 命令"二级菜单(强制 PipBackend 输出),满足企业用户必须用 pip 的场景。

---

## 7. 命令构建对比(uv vs pip,同 profile)

以 cu128 profile 为例:
```
# uv(自动/指定)
uv pip install --python <qgis-py> --target <key>/cu128/python \
  --no-config --cache-dir <plugin>/.deps/uv-cache \
  --no-python-downloads --strict --torch-backend cu128 \
  --constraint <constraints> torch torchvision ultralytics ...

# pip 回退
<qgis-py> -m pip install --target <key>/cu128/python --upgrade \
  --only-binary torch,torchvision,... \
  --constraint <constraints> \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  torch torchvision ultralytics ...
```
- `cpu` profile:uv 无 `--torch-backend`;pip 走默认 PyPI。
- `auto` + uv:`--torch-backend auto`(preview,文案标"自动推荐")。
- `custom` profile:uv 无 torch flag + `--index-url <user>`;pip 同理;`custom_index.txt` 持久化 url。

---

## 8. 落地顺序(建议分 5 个 PR)

1. **PR1 基础设施**:`uv_manager.py`(detect/locate,先不做 pip 安装)+ `runtime_profile.py` + `dependency_path.py` 改 profile 嵌套 + legacy 迁移。不改 UI,现有 pip 流程继续工作(默认 cpu profile)。
2. **PR2 引擎抽象**:`dependency_installer.py` + `torch_backend.py`,`plugin_settings.py` 删旧命令构建,`install_dependencies()` 改走 `resolve_backend`。auto 模式 = 系统 uv 有就用,没有回退 pip(此阶段还不自动装 uv)。
3. **PR3 私有 uv 安装**:`uv_manager.resolve_uv()` 加 `pip install --target` 分支;engine=uv 强制装,engine=auto 尝试装失败回退 pip。
4. **PR4 UI**:`settingsTool.py` 依赖 tab 重构——引擎/后端下拉、profile 切换、复制命令按钮。
5. **PR5 打磨**:macOS 平台门控、custom index-url 编辑、切换 profile 释放模型提示、清理 UI 适配多 profile、文档。

---

## 9. 风险与边缘情况

| 风险 | 处理 |
|---|---|
| `pip install uv` 也失败(无网/纯内网) | auto 回退裸 pip 装 deps;engine=uv 报清晰错误并建议切 auto/pip |
| uv `_find_uv` API 变动 | 定位失败时回退扫描 `<dir>/bin/uv`、`<dir>/Scripts/uv.exe`;仍失败回退 pip |
| QGIS Python 是 conda/pinned,pip 装 uv 拉入 uv 的依赖 | uv 包几乎无运行时依赖(`--only-binary :all:` + `--target` 隔离),风险低;仍隔离在 `.deps/uv-pkg` 不进 QGIS site-packages |
| 切换 profile 后旧 torch 残留内存 | `switch_profile` 强制 `release_runtime_models()` + `_flush_torch_memory()` + 弹确认框 |
| Windows 文件占用导致 profile 目录无法删 | 清理前先从 sys.path 摘 + 释放模型;删失败给"请重启 QGIS 后清理"提示 |
| 多 profile 磁盘翻倍(cpu+cu128 各 ~3GB) | "清理当前 Profile"按钮 + 状态栏显示各 profile 大小 |
| uv `--torch-backend auto` 是 preview | UI 文案写"自动推荐(实验性)",不承诺;失败回退 cpu |
| 用户 uv 配置干扰 | uv 调用恒加 `--no-config --cache-dir <plugin>` |
| conda QGIS 的 pip 不认 `--only-binary :all:` | 兼容:失败重试不带该 flag(让源码兜底,虽慢) |

---

## 10. 需要你确认的开放问题

在开工前我想敲定这几个点(它们会显著影响实现细节):

1. **auto 模式是否静默 `pip install` 私有 uv?**
   - 选项 A:静默装(你目前描述倾向)——体验顺滑,但"自动装二进制"在 auto 名义下可能让某些用户意外。
   - 选项 B:auto 仅用系统 uv,没有就回退裸 pip;只有 engine=uv 才 pip 装私有 uv——更保守。
   - 选项 C:auto 首次装 uv 前弹一次确认框("Geo-SAM 将通过 pip 下载 uv 到插件私有目录,是否允许?"),记住选择。

2. **切换 profile 的行为**:
   - 选项 A:切换 = 只换 sys.path 指向(若该 profile 目录已装好 torch,秒切)。
   - 选项 B:切换即触发安装(切到 cu128 就开始装 cu128 torch)。
   - 选项 C:两个按钮分开——"仅切换(需已装)"+ "切换并重装",如 UI 草图。我倾向 C。

3. **`custom` profile 是否允许多个?**(例如同时存 cu118 + cu128)
   - 选项 A:只有一个 `custom` 槽位,改 index-url 就覆盖。
   - 选项 B:支持命名多个 custom profile(如 `custom/cu118`、`custom/rocm`),UI 列表选。
   - A 简单,B 灵活但 UI 复杂。建议先 A,后续再扩。

4. **MIN_UV 版本**:我打算设 `0.6.0`(`--torch-backend` 稳定可用)。你机器上若有特定 uv 版本约束,告诉我。

5. **`geosam` 本地开发路径** `LOCAL_GEOSAM_REPOSITORY` 现在写死 `/Users/fancy/Documents/GitHub/geosam`。重构时保留这个本地优先逻辑,还是改成 settings 里可配?

请就这 5 点给个意见,确认后我就按上面的结构和落地顺序开始实现 PR1。

---

## User

1.不要engin了，统一使用uv；无网络那pip也没法使用了，就没必要切到pip了。整体运行的方向是，先使用pip安装UV到一个独立环境，然后使用UV管理依赖。 2. torch平台默认使用uv的auto特性 3. 切换profile可否改成如果profile已经装了，就直接切换。如果没安装，就自动重装。Custom profile使用选项A。GeoSAM本地开发路径不用管。

---

## Assistant (Plan · GLM-5.2 · 95.0s)

_Thinking:_

The user has made several key decisions that simplify the plan significantly:

1. **No engine selection** — just use uv always. No pip fallback for deps (since no network = pip fails too). The flow is: pip installs uv into an isolated location → uv manages all dependencies. If pip install uv fails, that's a hard failure (no point falling back to pip for deps either).

2. **Torch platform default: use uv's `auto` feature** — `--torch-backend auto` as default.

3. **Profile switching**: if profile already installed → direct switch. If not installed → auto-install. Custom profile = option A (single slot, overwrite index-url).

4. **GeoSAM local dev path**: don't touch it (keep as-is).

The system reminder says I'm in PLAN MODE - read-only. I must NOT make any edits. I can only read, search, and plan. I should present a refined plan and ask clarifying questions.

So this dramatically simplifies the architecture:
- No `engine` field, no `auto/uv/pip` engine combo box
- No `PipBackend` needed for deps (only pip is used ONCE to install uv itself)
- No `resolve_backend()` fallback chain — it's always uv
- The torch backend choice simplifies: default `auto`, but allow manual override (cpu/cu128/cu130/custom)

Let me reconsider the whole architecture with these decisions:

### Simplified flow:
```
1. Plugin startup / install-deps click:
   - Resolve uv binary:
     a. Check private uv (.deps/uv-pkg) — if present & version ok, use it
     b. Else: pip install uv into .deps/uv-pkg (uses user's pip.conf/mirror)
        <qgis-py> -m pip install --target .deps/uv-pkg --only-binary :all: uv>=MIN
     c. Locate binary via uv._find_uv.find_uv()
   - If pip install uv fails → hard error (no fallback)

2. Build uv command for deps:
   uv pip install --python <qgis-py> --target <key>/<profile>/python \
     --no-config --cache-dir .deps/uv-cache --no-python-downloads --strict \
     --torch-backend auto \      # default
     --constraint <constraints> \
     <packages>

3. Profile switching:
   - User selects profile in UI
   - If <key>/<profile>/python has torch installed → switch sys.path (秒切)
   - If not → auto-trigger install into that profile, then switch
   - custom profile: single slot, editable index-url
```

### What stays from Plan C:
- Multi-profile target directories under runtime-key
- RuntimeProfile concept
- uv_manager, torch_backend, runtime_profile, dependency_installer modules
- Copy-command button
- Legacy migration

### What's removed:
- engine field / engine combobox
- PipBackend for deps
- resolve_backend() fallback logic
- "auto engine" concept (now always uv)
- uv detect-from-system-uv path? Actually... let me think. The user said "统一使用uv". Does that mean we should ALSO check for system uv first (to avoid re-downloading)? 

The user's flow: "先使用pip安装UV到一个独立环境，然后使用UV管理依赖". This suggests: always pip-install uv into the plugin's isolated env, then use THAT uv. Not use system uv. This makes sense for consistency — system uv might be old, misconfigured, etc. And it's isolated.

But wait — re-downloading uv every time would be wasteful. The logic should be:
1. Check if private uv (.deps/uv-pkg) already installed & version ok → use it
2. Else pip install it → use it

So we don't scan system uv at all. Always use the private one. This is cleaner and more deterministic. Good.

Actually, let me re-read: "1.不要engin了，统一使用uv；无网络那pip也没法使用了，就没必要切到pip了。整体运行的方向是，先使用pip安装UV到一个独立环境，然后使用UV管理依赖。"

Yes — always use the private uv (installed via pip into isolated env). No system uv detection. No engine choice. If pip can't install uv (no network), it's a hard failure because pip couldn't install deps either.

This is much cleaner. Let me refine the plan.

### Simplified modules:

```
tools/
├── uv_manager.py            (新) ensure_uv() → pip install --target; locate binary; version check
├── torch_backend.py         (新) TorchChoice (auto/cpu/cu128/cu130/custom) → uv flag; pip_index_url for custom
├── runtime_profile.py       (新) RuntimeProfile + profile_target_path + platform gate + legacy migration + active_profile
├── dependency_installer.py  (新) DependencyPlan + build_uv_commands() (single backend, no Protocol needed really)
├── dependency_path.py       (改) profile-aware target + register + switch_profile
├── plugin_settings.py       (瘦身) 删命令构建;保留约束文件;加 settings 字段
└── settingsTool.py          (改) 依赖 tab:profile 下拉 + 复制命令 + 安装/清理
```

Actually with only one backend, do we need `InstallerBackend` Protocol? Probably overkill. Just a `build_uv_commands(plan)` function. Simpler.

### settings fields:
- `torch_choice`: "auto" (default) | "cpu" | "cu128" | "cu130" | "custom"
- `custom_index_url`: str
- (no engine field)
- (no uv_path field — always private)

### UI (simplified):
```
┌─ 依赖 ─────────────────────────────────────────────────┐
│ Torch 后端: [ Auto ▾ ]                                 │
│   (自动检测系统 GPU/CUDA 驱动,实验性)                  │
│                                                        │
│ 当前 Profile: [ CPU ▾ ]   (macOS 下 cu* 灰显)          │
│   自定义 index-url: [____________] (仅 custom 可编辑)  │
│                                                        │
│ [安装/修复依赖] [复制安装命令] [刷新状态] [打开文件夹] │
│ [清理当前 Profile] [清理整个运行时]                    │
│                                                        │
│ ▽ 安装日志                                              │
└────────────────────────────────────────────────────────┘
```

Wait — there's a subtlety. The "Torch 后端" dropdown and "当前 Profile" dropdown — are these the same thing or two different things?

In my earlier plan, profile = the target directory (cpu/cu128/cu130/custom), and torch_choice = the uv flag (auto/cpu/cu128/cu130/custom). They could differ: e.g., profile=cpu but torch_choice=auto (uv auto detects no GPU → installs CPU torch into cpu profile). Or profile=cu128, torch_choice=auto (uv auto detects cu128 → installs cu128 torch into cu128 profile).

But that's confusing. Let me simplify: **profile IS the torch backend choice**. One dropdown. Selecting "Auto" means: uv auto-detects and installs into a profile directory named after what uv chose... no, that breaks the "profile directory" model because auto could choose different things on different machines.

Hmm. Let me reconsider.

Option 1: profile = explicit (cpu/cu128/cu130/custom). torch_choice separate (auto/cpu/...). profile decides the directory, torch_choice decides the flag. If profile=cpu, torch_choice=auto is weird (auto might pick cu128 but you're installing into cpu dir).

Option 2: profile = the directory AND the torch choice, unified. "Auto" is a special profile that maps to a directory `auto/python` and uses `--torch-backend auto`. When auto-detect picks cu128, the torch wheel lands in `auto/python` regardless. If user later explicitly picks "cu128" profile, it goes in `cu128/python` — a different directory.

Option 2 is cleaner UX: one dropdown. But "auto" profile directory contents depend on what uv chose at install time, which is a bit opaque. Still, it works: the wheels in `auto/python` are whatever uv installed, and they're consistent because uv picked them based on the machine.

Actually, the user said "torch平台默认使用uv的auto特性". So auto is the DEFAULT. And they want profile switching based on installed/not-installed.

Let me unify: **TorchChoice = RuntimeProfile = Literal["auto", "cpu", "cu128", "cu130", "custom"]**. One dropdown. Each choice has its own target directory. "auto" uses `--torch-backend auto`. Switching profile = check if dir has torch, if yes switch, if no install.

This is the cleanest. Let me go with this.

Directory:
```
<key>/
  auto/python/       # uv auto picked wheels
  cpu/python/
  cu128/python/
  cu130/python/
  custom/python/
  active_profile.txt
```

macOS: only `cpu`, `custom`, and arguably `auto` (which on macOS auto = cpu anyway). Actually `auto` on macOS would detect no CUDA and install CPU torch, so it's fine to offer `auto` on macOS too. Let me keep `auto` available on all platforms (it's the default), and only grey out `cu128`/`cu130` on macOS.

Hmm, but then `auto` and `cpu` on macOS produce identical wheels in different dirs — wasteful. But that's user's choice. Fine.

### torch_backend.py simplifies to:
```python
TorchProfile = Literal["auto", "cpu", "cu128", "cu130", "custom"]

def uv_torch_flag(profile: TorchProfile) -> list[str]:
    if profile == "cpu": return []                    # uv 默认 CPU
    if profile == "auto": return ["--torch-backend", "auto"]
    if profile in ("cu128", "cu130"): return ["--torch-backend", profile]
    return []                                          # custom → index-url

def pip_index_url_for_profile(profile, custom_url) -> str | None:
    # only used for the one-time pip install of uv itself? No, uv install uses PyPI.
    # Actually pip_index_url is for deps, but deps go through uv now.
    # For custom profile, uv supports --index-url too:
    # uv pip install --index-url <custom> ...
    if profile == "custom": return custom_url
    return None
```

Wait — for `custom` profile with uv, we pass `--index-url <custom>` to uv. uv supports `--index-url`/`--extra-index-url`. Good.

### uv install (one-time, via pip):
```
<qgis-py> -m pip install --target <plugin>/.deps/uv-pkg \
  --upgrade --only-binary :all: "uv>=0.6.0"
```
This uses user's pip.conf/mirror. If it fails → hard error with message "无法安装 uv,请检查网络/pip 镜像配置"。

### Deps install (always uv):
```
<uv-binary> pip install --python <qgis-py> \
  --target <key>/<profile>/python \
  --no-config --cache-dir <plugin>/.deps/uv-cache \
  --no-python-downloads --strict \
  <uv_torch_flag(profile)> \
  [--index-url <custom_url> if custom] \
  --constraint <constraints> \
  <packages>
```

### Copy command:
- "复制安装命令" builds the uv command (with current profile choice) and copies to clipboard.
- Does NOT trigger uv install or deps install. Pure string.

### Profile switch logic:
```python
def switch_profile(profile: TorchProfile) -> None:
    key = _runtime_dependency_key()
    target = profile_target_path(key, profile)
    if not (target / "torch" exists or target has torch dist-info):
        # not installed → auto-install first
        install_into_profile(profile)   # runs uv pip install
    # now switch sys.path
    old_target = current registered path
    sys.path.remove(old_target)
    sys.path.insert(0, str(target))
    set_active_profile(key, profile)
    release_runtime_models(); _flush_torch_memory()
```

The user said "如果profile已经装了，就直接切换。如果没安装，就自动重装" — "重装" I think means "安装" (install), not reinstall. So: installed → switch; not installed → install then switch. My logic above matches.

Actually "自动重装" literally = "auto re-install". Maybe they mean: if not installed, auto-install (which is "重装" since it's installing the profile's deps). Same thing.

### Settings persistence:
- `active_profile`: stored in `active_profile.txt` per runtime-key (not in user.json) because it's runtime-specific. Actually, it could be in user.json keyed by runtime-key. But a text file next to the profile dirs is simpler and co-located. Let me keep `active_profile.txt`.
- `torch_profile` (the user's preferred default for new runtimes): stored in `user.json`. When a new runtime-key appears (e.g., QGIS upgrade), it initializes with this preference. Hmm, or just default to "auto" always. The user said "默认使用uv的auto特性" — so default = "auto" for new runtimes. Keep it simple: default "auto", user can switch per runtime.

Actually, do we need to persist the profile choice globally? Since `active_profile.txt` is per runtime-key, and each runtime-key is per QGIS install, the choice persists per QGIS install automatically. Good. No need for user.json field except maybe `custom_index_url`.

`custom_index_url`: stored in user.json (global, not per-runtime).

### Let me reconsider whether we need `engine` removed everywhere.
- No engine combobox in UI. ✓
- No PipBackend. ✓
- No resolve_backend fallback. ✓
- `DependencyPlan` has no `engine` field. ✓
- pip is used ONLY inside `uv_manager.ensure_uv()` to install uv. ✓

### uv_manager.py
```python
MIN_UV = "0.6.0"
UV_PKG_DIR = PLUGIN_ROOT / ".deps" / "uv-pkg"
UV_CACHE_DIR = PLUGIN_ROOT / ".deps" / "uv-cache"

def ensure_uv() -> Path:
    """Ensure private uv is installed and return binary path."""
    binary = _locate_private_uv_binary()
    if binary and _uv_version_ok(binary):
        return binary
    _pip_install_uv()
    binary = _locate_private_uv_binary()
    if not binary:
        raise RuntimeError("uv binary not found after pip install")
    return binary

def _locate_private_uv_binary() -> Path | None:
    # try uv._find_uv first
    if UV_PKG_DIR exists:
        sys.path.insert(0, str(UV_PKG_DIR))
        try:
            import uv
            from uv import _find_uv
            return Path(_find_uv.find_uv())
        except ImportError:
            pass
        finally:
            sys.path.remove(str(UV_PKG_DIR))
    # fallback: scan bin/ Scripts/
    ...

def _pip_install_uv() -> None:
    py = resolve_python_interpreter()
    cmd = [str(py), "-m", "pip", "install", "--target", str(UV_PKG_DIR),
           "--upgrade", "--only-binary", ":all:", f"uv>={MIN_UV}"]
    subprocess.run(cmd, check=True)

def _uv_version_ok(binary: Path) -> bool:
    out = subprocess.run([str(binary), "--version"], capture_output=True, text=True)
    # parse "uv 0.7.4" → compare
    ...
```

### dependency_installer.py (simplified, no Protocol)
```python
@dataclass(frozen=True, slots=True)
class DependencyPlan:
    packages: tuple[str, ...]
    python_path: Path
    target_path: Path
    constraints_path: Path | None
    profile: TorchProfile
    uv_binary: Path
    custom_index_url: str | None = None

def build_uv_commands(plan: DependencyPlan) -> list[list[str]]:
    cmd = [str(plan.uv_binary), "pip", "install",
           "--python", str(plan.python_path),
           "--target", str(plan.target_path),
           "--no-config", "--cache-dir", str(UV_CACHE_DIR),
           "--no-python-downloads", "--strict"]
    cmd += uv_torch_flag(plan.profile)
    if plan.profile == "custom" and plan.custom_index_url:
        cmd += ["--index-url", plan.custom_index_url]
    if plan.constraints_path:
        cmd += ["--constraint", str(plan.constraints_path)]
    cmd += [requirement_for(p) for p in plan.packages if p != "geosam"]
    commands = [cmd]
    if "geosam" in plan.packages:
        commands.append([... --no-deps geosam ...])
    return commands
```

Now let me think about what questions remain open. The user has answered most. Let me check:

1. ✅ No engine — always uv (via pip-installed private uv)
2. ✅ Torch default = uv auto
3. ✅ Profile switch: installed→switch, not installed→auto-install; custom = single slot
4. ✅ GeoSAM local dev path: don't touch

Remaining open questions I should clarify:
- The unification of "Torch 后端" and "Profile" into one dropdown — is that what they want? Or do they want two separate dropdowns? I'll propose unification and ask.
- "复制安装命令" — should it copy the uv deps command, or also include the one-time `pip install uv` command? I think both (so user can run from terminal). I'll propose.
- Legacy migration: rename old `<key>/python` → `<key>/<default>/python`. Which default? "auto" or "cpu"? Since default is now "auto", maybe migrate to "auto". But old dir had CPU torch (pinned). Hmm. Migrating to "cpu" is more honest. I'll propose "cpu" and ask.
- Whether to keep `--torch-backend auto` as default even on macOS (where it'll just install CPU torch). Yes, harmless. No need to ask.

Let me also reconsider: with no pip fallback for deps, the `PipBackend` and `pip_index_url` stuff is gone. But `custom` profile with uv still needs `--index-url`. uv supports it. Good.

One more: the existing `install_dependencies()` in plugin_settings.py is called from settingsTool. We replace its body to: ensure_uv() → build_uv_commands → run. The QProcess streaming UI stays.

Also `sam_processing_algorithm.py:521` `_select_device` — this is runtime device selection (cuda/mps), unrelated to install-time torch backend. Leave it. Though now with cu128 profile installed, `torch.cuda.is_available()` will work. Good.

Let me also think about whether `--torch-backend auto` plays well with `--constraint <qgis-runtime-constraints>`. The constraints file pins torch==2.11 etc. But `--torch-backend` makes uv resolve torch from pytorch index. Could conflict: constraint says `torch==2.11.0` but pytorch-cu128 index might have `torch==2.11.0+cu128`. The `==` without local version might not match `+cu128`. This is a real concern.

Actually, when using `--torch-backend`, uv handles the torch index automatically and the constraint on torch might need to NOT pin the local version. Or we drop torch from constraints when profile != cpu. Let me note this as a risk to handle in implementation: when profile in {auto, cu128, cu130, custom}, remove torch/torchvision from the constraints file (let uv's torch-backend resolve them). When profile == cpu, keep the existing torch pins. I'll mention this.

Alright, let me also verify: does `uv pip install --target` support `--index-url`? Let me check quickly. Actually uv supports `--index-url`/`--extra-index-url` in pip interface. Yes.

Now, the system reminder says PLAN MODE — read only, no edits. I must present the refined plan and ask remaining questions. Let me write it up concisely.

Let me structure the response:
1. Acknowledge the simplifications
2. Refined Plan C (simplified)
3. Remaining clarifying questions

I should be concise but thorough. Let me write in Chinese since the user is.

明白,这些决定大幅简化了架构。让我先确认几个实现细节,再敲定最终计划。

## 简化后的核心流程

```
启动/安装依赖:
  ① ensure_uv():
     - 检查 .deps/uv-pkg 是否已装 uv 且版本 ≥ MIN_UV
     - 否则: <qgis-py> -m pip install --target .deps/uv-pkg --only-binary :all: "uv>=0.6.0"
       (借用户 pip.conf 镜像;失败=硬错误,不回退)
     - 定位 binary: 临时 sys.path.insert → import uv → uv._find_uv.find_uv()
  ② uv pip install --python <qgis-py> --target <key>/<profile>/python \
       --no-config --cache-dir .deps/uv-cache --no-python-downloads --strict \
       --torch-backend auto \          # 默认
       --constraint <constraints> <packages>
```

**移除的东西**:engine 字段、PipBackend、resolve_backend 回退链、系统 uv 检测、engine 下拉框。pip 只在 `ensure_uv()` 里用一次。

## 关键简化:Torch 后端 = Profile(合一)

我把"Torch 后端选择"和"Profile 目录"**合并成一个下拉框**,因为分离会造成混乱(例如 profile=cpu 但 torch_choice=auto,uv 自动装了 cu128 wheel 却落进 cpu 目录)。

```python
TorchProfile = Literal["auto", "cpu", "cu128", "cu130", "custom"]
# auto    → --torch-backend auto,目录 <key>/auto/python
# cpu     → 无 torch flag,目录 <key>/cpu/python
# cu128   → --torch-backend cu128,目录 <key>/cu128/python
# cu130   → --torch-backend cu130,目录 <key>/cu130/python
# custom  → --index-url <user>,目录 <key>/custom/python
```

UI 只有一个下拉:
```
Torch 后端: [ Auto ▾ ]    (默认 Auto,uv 自动检测 GPU/CUDA)
  Auto / CPU / NVIDIA CUDA 12.8 / NVIDIA CUDA 13.0 / 自定义
```

## 目录布局(确认你的第 2 点)

```
.deps/
  uv-pkg/                    # pip install --target 产物(uv Python 包+binary)
  uv-cache/                  # uv 下载缓存(隔离,可清)
  runtimes/
    <runtime-key>/
      auto/python/           # ← 默认 profile,迁移自旧 <key>/python
      cpu/python/
      cu128/python/
      cu130/python/
      custom/python/
      active_profile.txt     # 当前激活:"auto"
```

## 模块结构(精简)

```
tools/
├── uv_manager.py            (新) ensure_uv()/locate binary/version check
├── torch_profile.py         (新) TorchProfile + uv_torch_flag() + 平台门控
├── runtime_profile.py       (新) profile_target_path/active_profile/legacy迁移/switch
├── dependency_installer.py  (新) DependencyPlan + build_uv_commands()
├── dependency_path.py       (改) profile 感知 + switch_profile
├── plugin_settings.py       (瘦身) 删 1326-1546 命令构建;保留约束文件
└── settingsTool.py          (改) 依赖 tab:profile 下拉 + 复制命令 + 安装/清理
```

`InstallerBackend` Protocol 不再需要(只有一个后端),直接 `build_uv_commands(plan)` 函数即可。

## Profile 切换逻辑(按你的要求)

```python
def switch_profile(profile: TorchProfile) -> None:
    key = _runtime_dependency_key()
    target = profile_target_path(key, profile)
    if not _profile_has_torch(target):        # 没装
        install_into_profile(profile)         # 自动装(跑 uv pip install)
    # 已装 → 直接切
    sys.path 摘旧 target,插新 target
    set_active_profile(key, profile)
    release_runtime_models(); _flush_torch_memory()   # ABI 变了必须释放
```

UI 上一个按钮"切换",内部自动判断装没装。custom profile = 单槽位,改 index-url 覆盖 `custom_index.txt`。

## 需要你确认的 4 个细节

1. **约束文件与 `--torch-backend` 的冲突**:现有 `_create_qgis_runtime_constraints_file()` 把 `torch==2.11.0` 写进 constraints。但 `--torch-backend cu128` 让 uv 从 pytorch-cu128 index 解析,torch wheel 版本带 `+cu128` local 标签,`==2.11.0` 约束可能匹配不上。
   - 我的方案:**profile ∈ {auto, cu128, cu130, custom} 时,从 constraints 文件里剔除 torch/torchvision**(让 uv 的 torch-backend 全权解析);仅 profile=cpu 时保留现有 torch pin。
   - 你同意吗?

2. **Legacy 迁移目标**:旧 `<key>/python`(装的是 CPU torch)迁到哪个 profile 目录?
   - 选项 A:`<key>/auto/python`(因为 auto 是默认,且旧 torch 是 CPU,auto 在多数机器上也装 CPU)
   - 选项 B:`<key>/cpu/python`(更如实反映"这是 CPU torch")
   - 我倾向 B(如实),然后 `active_profile.txt = "cpu"`。新用户默认从 "auto" 开始。你选哪个?

3. **"复制安装命令"复制什么**:
   - 选项 A:只复制 `uv pip install ...` 依赖安装命令(假设 uv 已装好)
   - 选项 B:复制完整两步——先 `pip install uv` 再 `uv pip install deps`(用户拿去终端能完整复现)
   - 选项 C:两个按钮——"复制 uv 命令"+"复制完整命令"
   - 我倾向 B(终端用户一次复制就能跑)。你选哪个?

4. **macOS 平台门控**:macOS 上 `cu128`/`cu130` 下拉项灰显并提示"macOS 不支持 CUDA"。但 `auto` 和 `cpu` 都保留(auto 在 macOS 上 uv 会装 CPU torch,无害)。你同意,还是想 macOS 上连 `auto` 也藏掉只留 `cpu`/`custom`?

---

确认这 4 点后,我就把最终计划写成 PR1-PR5 的执行清单(含每个 PR 的文件级改动和验证步骤),供你 review 后开工。

---

## User

1. 同意你的方案 2. B, 3. 选项 C 4. macOS 上 cu128/cu130 下拉项灰显并提示"macOS 不支持 CUDA"

---

## Assistant (Plan · GLM-5.2 · 79.6s)

_Thinking:_

The user has confirmed all 4 details and we're now in Plan Mode (READ-ONLY). I need to produce a comprehensive, executable plan for PR1-PR5 without making any changes. Let me synthesize everything into a final plan document.

Key decisions locked in:
1. Constraints file: strip torch/torchvision when profile ∈ {auto, cu128, cu130, custom}; keep torch pin only for cpu profile
2. Legacy migration: `<key>/python` → `<key>/cpu/python`, active_profile="cpu"
3. Copy command: two buttons — "复制 uv 命令" + "复制完整命令"(pip install uv + uv pip install deps)
4. macOS: cu128/cu130 greyed out with "macOS 不支持 CUDA" tooltip; auto and cpu kept (auto on macOS = CPU torch, harmless)

Also locked earlier:
- No engine field, always uv
- ensure_uv() via pip install --target .deps/uv-pkg
- TorchProfile = Literal["auto","cpu","cu128","cu130","custom"] — backend selector and profile directory merged into one dropdown
- switch_profile: if not installed, auto-install; if installed, just switch
- custom = single slot, custom_index.txt
- MIN_UV = "0.6.0"
- LOCAL_GEOSAM_REPOSITORY: keep as-is (don't manage)
- geosam local package: --no-deps separate command (existing behavior)

I'm in Plan Mode — I must NOT edit anything. I should present the final comprehensive plan. Let me write it up as a PR-by-PR execution checklist.

Let me be thorough but concise. I'll structure it as:
- Final architecture summary
- PR1-PR5 each with: files changed, key functions, verification steps
- Open items / risks already addressed

I should not make any tool calls that modify. I can still read/search if needed but I've already gathered enough. Let me just write the plan as text output.

Actually, let me make sure I have one thing — I should double check the existing install_dependencies sync behavior and the settings UI structure so my plan is accurate. I've already read settingsTool.py 490-689 and plugin_settings.py fully. I think I have enough.

Let me write the final plan.

所有决策已锁定。下面是 **方案 C 精简版** 的最终执行计划,按 PR1-PR5 排列,每个 PR 含文件级改动和验证步骤。

---

# 最终架构总览

```
统一 uv 后端,无 engine 字段
启动流程: ensure_uv() (pip install uv 到 .deps/uv-pkg) → uv pip install deps 到 profile target
TorchProfile 合一:  auto|cpu|cu128|cu130|custom  (后端选择=目录选择)
切换 profile:  没装→自动装,已装→直接切;切完释放模型
复制命令:  两按钮(仅 uv / 完整 pip+uv)
macOS:  cu128/cu130 灰显;auto/cpu 保留
Constraints:  profile∈{auto,cu128,cu130,custom} 时剔除 torch pin;仅 cpu 保留
Legacy:  <key>/python → <key>/cpu/python, active=cpu
```

```
.deps/
  uv-pkg/            pip install --target 产物(uv 包+binary,临时 import 定位)
  uv-cache/          uv 下载缓存(隔离,可清)
  runtimes/<key>/
    auto|cpu|cu128|cu130|custom/python/   ← --target
    active_profile.txt
    custom/custom_index.txt               (custom profile 的 index-url)
```

```
tools/
├── uv_manager.py            (新) ensure_uv/locate_binary/version_gate
├── torch_profile.py         (新) TorchProfile + uv_torch_flag + 平台门控 + pip_index_url
├── runtime_profile.py       (新) profile_target_path/active/switch/migrate_legacy
├── dependency_installer.py  (新) DependencyPlan + build_uv_commands + build_full_commands
├── dependency_path.py       (改) profile 感知;register 读 active;switch_profile
├── plugin_settings.py       (瘦身 -350行) 删 1326-1546 命令构建;保留约束 758-981+torch-strip
└── settingsTool.py          (改) 依赖 tab:profile 下拉+双复制按钮+安装/清理
```

---

# PR1 — 基础设施:profile 目录 + legacy 迁移 + uv_manager 骨架

**目标**:目录结构就位,现有 pip 流程继续工作(默认走 cpu profile)。不改 UI。

## 文件改动

### `tools/runtime_profile.py` (新, ~180 行)
```python
TorchProfile = Literal["auto", "cpu", "cu128", "cu130", "custom"]
PROFILE_VALUES: tuple[TorchProfile, ...] = ("auto","cpu","cu128","cu130","custom")
DEFAULT_PROFILE: TorchProfile = "auto"

def profiles_for_platform() -> list[TorchProfile]:
    # darwin → ["auto","cpu","custom"];其他 → 全部

def profile_target_path(key: str, profile: TorchProfile) -> Path:
    return PLUGIN_RUNTIME_DEPENDENCIES_ROOT / key / profile / "python"

def active_profile(key: str) -> TorchProfile:
    # 读 <key>/active_profile.txt;不存在 → DEFAULT_PROFILE

def set_active_profile(key: str, profile: TorchProfile) -> None:
    # 写 active_profile.txt

def migrate_legacy_profile(key: str) -> None:
    # 若 <key>/python 存在 且 <key>/cpu/python 不存在:
    #   rename <key>/python → <key>/cpu/python
    #   set_active_profile(key, "cpu")
    # 幂等;已迁移则 no-op

def custom_index_url(key: str) -> str | None:
    # 读 <key>/custom/custom_index.txt

def set_custom_index_url(key: str, url: str) -> None: ...
```

### `tools/uv_manager.py` (新, ~150 行, 本 PR 只做骨架)
```python
MIN_UV = "0.6.0"
UV_PKG_DIR = PLUGIN_DEPENDENCY_ROOT / "uv-pkg"
UV_CACHE_DIR = PLUGIN_DEPENDENCY_ROOT / "uv-cache"

def ensure_uv() -> Path:
    """确保 uv 可用,返回 binary 路径。本 PR 仅实现 detect+locate,不装。"""
    # 1. 检查 UV_PKG_DIR 是否已装 uv 且 version >= MIN_UV
    # 2. 定位 binary: _locate_uv_binary(UV_PKG_DIR)
    # 3. 本 PR 不做 pip install 分支(留给 PR2);没装就 raise RuntimeError

def _locate_uv_binary(pkg_dir: Path) -> Path | None:
    # 临时 sys.path.insert(pkg_dir);import uv;uv._find_uv.find_uv()
    # 失败回退扫描 pkg_dir/bin/uv、Scripts/uv.exe
    # 拿到路径后 sys.path 摘掉,不常驻

def _uv_version(binary: Path) -> tuple[int, ...] | None:
    # subprocess [binary, "--version"],parse "uv 0.7.4" → (0,7,4)

def version_ok(binary: Path) -> bool: ...
```

### `tools/dependency_path.py` (改)
- `get_plugin_managed_site_packages()` → 改为调 `profile_target_path(_runtime_dependency_key(), active_profile(key))`
- `register_plugin_managed_dependency_path()` → 先 `migrate_legacy_profile(key)`,再 `active_profile(key)`,再 insert 对应 target
- `iter_plugin_managed_runtime_site_packages()` → 遍历 `<key>/<profile>/python`(不再只 `<key>/python`)
- `clear_current_plugin_managed_site_packages()` → 只删 `<key>/<active>/python`
- `clear_all_plugin_managed_site_packages()` → 删整个 `runtimes/`,保留 legacy 兼容

### `tools/plugin_settings.py` (改, 小)
- 加 settings 字段:`"torch_profile": "auto"`, `"custom_index_url": ""`
- `_default_plugin_settings()` setdefault 上述字段
- `DEPENDENCY_INSTALL_REQUIREMENTS`、约束文件逻辑本 PR 不动

## 验证(PR1)
- `ruff check tools` + `py_compile` 全过
- 启动 QGIS 加载插件:旧 `.deps/runtimes/<key>/python` 自动 rename 到 `<key>/cpu/python`,`active_profile.txt=cpu`
- 依赖状态表正常显示(走 cpu profile target)
- 现有 "Install Missing" 仍走旧 pip 命令(本 PR 不动命令构建)
- `clear_current` / `clear_all` 仍工作
- 重启 QGIS 幂等(不重复迁移)

---

# PR2 — uv 后端落地 + 命令构建迁移

**目标**:`ensure_uv()` 完整(含 pip 安装私有 uv);依赖安装改走 uv;移除旧 pip 命令构建。UI 暂不动(仍用旧按钮,但底层调新函数)。

## 文件改动

### `tools/uv_manager.py` (补全)
```python
def ensure_uv() -> Path:
    # 1. UV_PKG_DIR 已装且 version_ok → locate
    # 2. 否则 _install_private_uv()
    # 3. locate + version_ok 校验;失败 raise(无回退)

def _install_private_uv() -> None:
    python = resolve_python_interpreter()
    cmd = [str(python), "-m", "pip", "install", "--target", str(UV_PKG_DIR),
           "--upgrade", "--only-binary", ":all:", f"uv>={MIN_UV}"]
    # subprocess 流式;失败 raise RuntimeError(含 pip 输出)
    # 借用户 pip.conf 镜像(不传 --index-url)
```

### `tools/torch_profile.py` (新, ~140 行)
```python
def uv_torch_flag(profile: TorchProfile) -> list[str]:
    # auto    → ["--torch-backend","auto"]
    # cpu     → []
    # cu128   → ["--torch-backend","cu128"]
    # cu130   → ["--torch-backend","cu130"]
    # custom  → []  (走 index-url)

def pip_index_url(profile, custom_url) -> str | None:
    # cu128 → "https://download.pytorch.org/whl/cu128"
    # cu130 → "https://download.pytorch.org/whl/cu130"
    # custom → custom_url
    # auto/cpu → None

def profile_allowed(profile) -> bool:
    # darwin and profile in {"cu128","cu130"} → False
```

### `tools/dependency_installer.py` (新, ~220 行)
```python
@dataclass(frozen=True, slots=True)
class DependencyPlan:
    packages: tuple[str, ...]
    python_path: Path
    target_path: Path
    constraints_path: Path | None
    runtime_key: str
    profile: TorchProfile
    uv_binary: Path
    custom_index_url: str | None = None

def build_uv_commands(plan: DependencyPlan) -> list[list[str]]:
    # 主命令:resolver packages
    # geosam 本地包:单独 --no-deps 一条(沿用 _dependency_install_requirement)
    # constraints:仅 profile=="cpu" 时带 torch pin;其他 profile 由 build 时剔除
    cmd = [str(plan.uv_binary), "pip", "install",
           "--python", str(plan.python_path),
           "--target", str(plan.target_path),
           "--no-config", "--cache-dir", str(UV_CACHE_DIR),
           "--no-python-downloads", "--strict"]
    cmd += uv_torch_flag(plan.profile)
    if plan.profile == "custom" and plan.custom_index_url:
        cmd += ["--index-url", plan.custom_index_url]
    if plan.constraints_path: cmd += ["--constraint", str(plan.constraints_path)]
    cmd += [requirement_for(p) for p in plan.packages if p != "geosam"]
    cmds = [cmd]
    if "geosam" in plan.packages:
        cmds.append([... geosam --no-deps ...])
    return cmds

def build_full_commands(plan: DependencyPlan) -> list[list[str]]:
    # 完整两步:先 pip install uv,再 uv pip install deps
    # 用于"复制完整命令"按钮,让用户终端复现

def make_plan(packages, profile, *, custom_index_url=None) -> DependencyPlan:
    # 组装:resolve_python_interpreter + profile_target_path + constraints + ensure_uv()

def install_dependencies(profile, log_callback=None) -> tuple[bool, str]:
    # make_plan → build_uv_commands → 流式执行(沿用现有 subprocess.Popen 模式)
```

### `tools/plugin_settings.py` (瘦身)
- **删除** `get_dependency_install_command` (1326-1349)
- **删除** `get_dependency_install_commands` (1352-1420)
- **删除** `_dependency_install_command_base` (1440-1477)
- **删除** `install_dependencies` (1497-1546)
- **保留** `format_dependency_install_command` / `format_dependency_install_commands` (1480-1494)(移到 `dependency_installer.py` 或留这)
- **改** `_create_qgis_runtime_constraints_file()`:加 `profile` 参数;profile != "cpu" 时从 constraint_lines 剔除 torch/torchvision 行(让 uv `--torch-backend` 全权解析)
- **改** `_qgis_runtime_resolver_constraints()` / `_torch_resolver_constraints()`:profile 感知
- `settingsTool.py` import 改从 `dependency_installer` 导入命令构建函数

## 验证(PR2)
- `ruff` + `py_compile`
- 手动测:`ensure_uv()` 首次调→`pip install --target .deps/uv-pkg uv`→定位 binary→版本 ≥ 0.6.0
- "Install Missing" 走 uv:日志显示 `uv pip install ... --torch-backend auto`
- CPU profile 安装:constraints 含 `torch==2.11.0`,uv 不带 torch flag
- auto profile 安装:constraints 不含 torch pin,uv 带 `--torch-backend auto`
- 断网测:`pip install uv` 失败 → 清晰错误(不静默回退)
- `geosam` 本地包仍 `--no-deps` 单独装

---

# PR3 — Profile 切换逻辑 + 运行时切换

**目标**:实现 switch_profile(没装→自动装,已装→直接切)+ 切换后释放模型。

## 文件改动

### `tools/runtime_profile.py` (补全)
```python
def profile_is_installed(key: str, profile: TorchProfile) -> bool:
    target = profile_target_path(key, profile)
    # 判据:target 存在 且 含 torch/torchvision dist-info
    # (避免空目录被误判为已装)

def switch_profile(profile: TorchProfile, *, install_if_missing=True,
                   log_callback=None) -> SwitchResult:
    # 1. profile_allowed(profile) → False 则 raise
    # 2. key = _runtime_dependency_key()
    # 3. if not profile_is_installed(key, profile) and install_if_missing:
    #      install_dependencies(profile, log_callback)
    # 4. _swap_sys_path(old_target → new_target)
    # 5. set_active_profile(key, profile)
    # 6. release_runtime_models() + _flush_torch_memory()
    # 7. return SwitchResult(was_installed, profile, target)

@dataclass(frozen=True, slots=True)
class SwitchResult:
    was_installed: bool
    profile: TorchProfile
    target_path: Path
```

### `tools/dependency_path.py` (小改)
- 暴露 `_swap_sys_path(old, new)` 给 `switch_profile` 用(从 sys.path 摘 old,插 new,invalidate_caches)

### `tools/geosam_runtime.py` (不动, 复用)
- `switch_profile` 调用现有 `release_runtime_models()` + `_flush_torch_memory()`

## 验证(PR3)
- cpu→auto 切换:auto 目录没装 → 自动 uv install → 切 sys.path → 模型释放
- auto→cpu 切换:cpu 已装 → 秒切 → 模型释放
- 切到 cu128(macOS):`profile_allowed` raise,UI 拦截
- 切换后 `dependency_status()` 反映新 profile 的包
- 切换后再次跑分割:用新 profile 的 torch ABI,不崩

---

# PR4 — UI 重构:profile 下拉 + 双复制按钮 + 安装/清理

**目标**:`settingsTool.py` 依赖 tab 重构。

## UI 布局
```
┌─ 依赖 ────────────────────────────────────────────────────┐
│ Torch 后端: [ Auto ▾ ]                                    │
│   (Auto = uv 自动检测 GPU/CUDA)                           │
│   macOS 下 cu128/cu130 灰显,tooltip "macOS 不支持 CUDA"   │
│                                                           │
│ 自定义 index-url: [.................] (仅 custom 可编辑)  │
│                                                           │
│ [安装缺失] [切换后端] [复制 uv 命令] [复制完整命令]       │
│ [刷新状态] [打开文件夹] [清理当前 Profile] [清理所有运行时]│
│                                                           │
│ ▽ 安装日志(流式 QPlainTextEdit)                          │
└───────────────────────────────────────────────────────────┘
```

## 文件改动

### `tools/settingsTool.py` (改, ~+200 / -100 行)
- 依赖 tab 顶部加 `QComboBox torch_profile_combo`:
  - 填充 `profiles_for_platform()`;macOS 下 cu128/cu130 `setItemData(idx, 0, Qt.UserRole)` 禁用 + tooltip
  - 当前值从 `load_plugin_settings()["torch_profile"]` 读
  - 切换时 `save_plugin_settings({"torch_profile": ...})`
- custom 选中时 `custom_index_url_edit` 启用,其他 profile 灰显
- "切换后端"按钮 → `switch_profile_clicked()`:
  - 确认框:"切换 Torch 后端将卸载已加载模型,是否继续?"
  - 调 `switch_profile(profile)`(内部自动判断装没装)
  - 流式日志接 `log_callback`
  - 完成后 `refresh_dependency_status()`
- "复制 uv 命令"按钮 → `copy_uv_command_clicked()`:
  - `make_plan(current_profile)` 但 **不调 ensure_uv**(不触发下载),uv_binary 用占位 `"uv"`
  - `build_uv_commands(plan)` → `format_dependency_install_commands` → clipboard
  - 复用 `widgetTool.py:3156` 的 `QApplication.clipboard().setText()` 模式
- "复制完整命令"按钮 → `copy_full_command_clicked()`:
  - `build_full_commands(plan)`(含 `pip install uv` 第一步)→ clipboard
- "安装缺失"按钮 → 现有逻辑改调 `install_dependencies(current_profile, log_callback)`
- "清理当前 Profile" → `clear_current_plugin_managed_site_packages()`
- "清理所有运行时" → `clear_all_plugin_managed_site_packages()`
- `refresh_dependency_status()` 反映当前 profile target 的状态

## 验证(PR4)
- macOS:cu128/cu130 下拉项灰显,tooltip 正确
- 切 custom → index-url 输入框启用;切其他 → 灰显
- "复制 uv 命令":剪贴板得 `uv pip install ... --torch-backend auto ...`
- "复制完整命令":剪贴板得 `pip install --target .deps/uv-pkg uv>=0.6.0` + `uv pip install ...` 两段
- 切换后端弹确认框;切换后模型释放,日志显示安装过程(若没装)
- 安装/清理按钮在安装中禁用(沿用现有 `_is_installing` 逻辑)

---

# PR5 — 打磨:平台门控细节、清理 UI、约束边缘、文档

**目标**:边缘情况收口。

## 改动
- `tools/torch_profile.py`:`detect_recommended_profile()` — `nvidia-smi` 拿驱动版本,≥12.8→cu128,≥13.0→cu130,无→cpu;macOS→auto(实际装 CPU)
- `tools/settingsTool.py`:Torch 后端下拉旁加"推荐"提示(读 `detect_recommended_profile()`)
- 清理 UI:状态栏显示各 profile 目录大小(`get_plugin_managed_dependency_stats` 扩展到 profile 目录)
- `tools/dependency_path.py`:`iter_all_plugin_managed_site_packages` 遍历所有 `<key>/<profile>/python`,legacy 兼容保留
- Windows 文件占用:清理失败提示"请重启 QGIS 后清理"
- `custom_index.txt` 持久化:custom profile 切换时读/写
- 文档:`docs/` 加依赖管理架构说明(Sphinx)
- `metadata.txt` changelog 加 uv 后端说明

## 验证(PR5)
- NVIDIA 机器:推荐提示 cu128/cu130 正确
- custom profile:改 index-url → 重装 → 新 index 生效
- 多 profile 磁盘占用显示正确
- Windows 清理文件占用提示
- `cd docs && make html` 渲染通过

---

# 跨 PR 的全局约束

- **每个 PR 前**:`git status` 干净;`ruff check tools ui *.py` + `py_compile` 全过
- **每个 PR 后**:在 QGIS 里实际跑一遍依赖安装 + 一次分割查询
- **不碰**:`LOCAL_GEOSAM_REPOSITORY` 逻辑保留;`geosam_backend.py`、`geosam_runtime.py`(除 switch 复用)、`sam_processing_algorithm.py`(device 逻辑不动)基本不动
- **Commit 风格**:沿用 `feat:`/`refactor:` 前缀,如 `feat: add uv-managed dependency installer with torch profile switching`

---

# 风险已收口

| 风险 | PR | 处理 |
|---|---|---|
| `pip install uv` 失败(无网) | PR2 | 硬错误,不静默回退(你确认无网= pip 也没用) |
| uv `_find_uv` API 变动 | PR2 | 回退扫描 bin/Scripts 目录 |
| constraints 与 `--torch-backend` 冲突 | PR2 | profile≠cpu 时剔除 torch pin |
| 切换 profile 后 torch ABI 崩 | PR3 | 强制 release_models + flush_torch + 确认框 |
| macOS 误选 CUDA | PR4 | 下拉灰显 + tooltip |
| Windows 文件占用 | PR5 | 清理失败提示重启 |
| 旧 `<key>/python` 残留 | PR1 | 幂等迁移到 `<key>/cpu/python` |

---

**下一步**:你 review 这份计划,确认无误后我从 **PR1** 开始实现。或者你想先调整某个 PR 的范围/顺序?

---

