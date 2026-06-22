# Repository Guidelines

## Project Structure & Module Organization

This repository is a QGIS plugin, aiming to using geosam as the core library to do image segmentation, and use QGIS as the interface to interact with users. The plugin is located at `/Users/fancy/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM`, and the geosam library is located at `/Users/fancy/Documents/GitHub/geosam`.

Root entry points such as `__init__.py`, [`geo_sam_tool.py`](/Users/fancy/Library/Application%20Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/geo_sam_tool.py), and [`geo_sam_provider.py`](/Users/fancy/Library/Application%20Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM/geo_sam_provider.py) wire the plugin into QGIS.

- `tools/`: runtime, canvas, settings, processing, and widget logic.
- `ui/`: Qt `.ui` files, config JSON, cursors, and icons.
- `docs/`: Sphinx documentation source and build helpers.
- `models/`: local model checkpoints used by the plugin at runtime.

Keep new plugin behavior in `tools/`, keep static assets in `ui/`, and avoid committing generated caches such as `.cache/`, `__pycache__/`, or machine-local settings.

## Code Style Guidelines

### Language and Naming

- Write all code and comments in English
- Use descriptive English names for variables, functions, and classes
- Follow PEP 8 and PEP 257 standards

## Code Conventions

- **Python 3.9+ and 3.11+ typing** required
- **`from __future__ import annotations`** in every file
- **Type hints**: Use Python 3.11+ syntax (`str | None`, `dict[str, int]`). Use `Literal` for fixed value sets. Do not using Union，Optional.
- **Docstrings**: Use NumPy-style docstrings for all public modules, classes, functions, and methods. Include Parameters, Returns, Raises, and Examples where applicable, and use Sphinx reStructuredText markup such as :func:..., :class:..., and directives like .. note::, .. tip::, and .. warning:: when needed.
- **Logging**: Use `from geosam.logging import setup_logger; logger = setup_logger(__name__)` — log before raising exceptions errors
- **Paths**: Use `pathlib.Path` internally; convert to `str` only when passing to ISCE2/ISCE3 APIs
- **Type-checking imports**: Put heavy/circular imports inside `if TYPE_CHECKING:` blocks
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- **Linter**: ruff only (no black, no flake8). Line length 88. Ruff excludes `tests/`, `docs/`, `examples/` directories.

## Build, Test, and Development Commands

- `uv run ruff check tools ui *.py`: lint Python sources.
- `uv run python -m py_compile tools/*.py ui/*.py geo_sam_tool.py geo_sam_provider.py`: quick syntax validation.
- `cd docs && uv run make html`: build the Sphinx documentation into `docs/build/html`. Run this after every doc edit session to verify rendering.
- `python -m pip install geosam rasterio geopandas pyarrow ultralytics`: install runtime dependencies into the QGIS Python environment when needed.

Run commands from the plugin root unless noted otherwise.

## Coding Style & Naming Conventions

Use 4-space indentation, Python 3.11+ type hints, and NumPy-style docstrings. Write code, comments, and log messages in English. Prefer descriptive `snake_case` for functions and variables, `PascalCase` for Qt/QGIS widget classes, and `UPPER_CASE` for constants. Use `Literal`, `TypedDict`, `Protocol`, and `Self` where they clarify the API. Format and lint with `ruff`; use `uv` for Python tooling.

## Testing Guidelines

There is no dedicated automated test suite in this repository today. At minimum:

- run `ruff` and `py_compile`;
- exercise the changed workflow in QGIS;
- verify both live encoding raster and feature-folder paths when touching `tools/geosam_runtime.py` or `tools/widgetTool.py`.

When adding tests later, place them in a top-level `tests/` package and name files `test_<module>.py`.

## Commit & Pull Request Guidelines

Recent commits use short, imperative prefixes such as `fix:`, `refactor:`, `update`, and `remove:`. Follow that style and keep subjects concise, for example `fix: preserve feature-cache model fallback`.

PRs should include a clear summary, affected QGIS workflows, manual test steps, and screenshots or GIFs for UI changes. Link related issues when relevant.

## Configuration & Safety Notes

Do not commit machine-specific paths in `ui/config/user.json`. Keep defaults portable, and log context before raising errors in runtime code.

## Release Branch Sync Strategy

When syncing changes from `main` to release branches (e.g. `qgis-release/2.0`), **only propagate modified files; do not re-introduce files that have been deleted on the release branch.**

The following paths are intentionally excluded from release branches and must never be synced back:

- `AGENTS.md`
- `docs/` (entire directory — documentation is hosted separately, not bundled in the plugin package)
- `README.md`
- `CITATION.cff`
- `clean_pycache.sh` (exists only on release branches, do not sync back to main)

Use `git checkout main -- <file>` or selective cherry-pick to bring over only the files that actually changed. Avoid `git merge main` into release branches, as it will resurrect the deleted files.
