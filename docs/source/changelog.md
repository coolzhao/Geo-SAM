# Changelog

## GeoSAM v2.0

GeoSAM v2.0 is a ground-up rewrite of the plugin. Compared with the last
release on the `main` branch (v1.3.2), the entire backend has been rebuilt
around the [`geosam`](https://github.com/Fanchengyan/geosam) Python library.

:::{note}
GeoSAM v2.0 is now available on the
[QGIS Official Plugin Repository](https://plugins.qgis.org/). Install and
update Geo-SAM directly from the QGIS Plugin Manager -- no manual download
needed.
:::

### Architecture Rewrite

- **Full backend rewrite** -- the segmentation engine now runs on the
  [`geosam`](https://github.com/Fanchengyan/geosam) core library, replacing
  the previous in-plugin implementation.
- **SAM backend switched to Ultralytics** -- model loading and inference now
  rely on the `ultralytics` SAM stack (Ultra-Lite SAM / SAM3) instead of the
  original `segment_anything` package.
- **Removed `torchgeo` dependency** -- the custom `torchgeo_sam.py` and
  `sam_ext.py` modules have been deleted; geospatial data handling is now
  provided by `geosam` and `rasterio` directly.
- **QGIS Processing algorithm** -- the Image Encoder is registered as
  `geo_sam:geo_sam_encoder` and is available from the Processing toolbox.
- **Runtime backend abstraction** -- `geosam_backend.py` and
  `geosam_runtime.py` isolate QGIS-specific runtime configuration from the
  core library.
- **Plugin-managed dependencies** -- PyTorch, geosam, ultralytics and other
  dependencies are installed into a plugin-private directory per QGIS
  runtime, so no manual `pip` commands are required.

### New Features

- **Live Encoding mode** -- segment directly from a raster layer without
  pre-encoding; features are encoded on the fly via QGIS background tasks and
  cached for fast re-queries.
- **Dual source mode selector** -- switch between **Live Encoding** and
  **Pre-encoded** workflows in the Segmentation tool.
- **Online tile layer support** -- XYZ and WMS tile layers can be used in
  live-encoding mode; tiles are exported to a local raster before encoding.
- **SAM2/SAM3 model support** -- added the SAM2 and SAM3 model family.
- **Vectorization mode** -- choose between **Pixel-Level** and **Simplified**
  polygon output.
- **Max Polygon Only mode** -- keep only the largest polygon from the current
  mask to filter out fragments.
- **Unified Settings dialog** -- four tabs (Dependencies, Model Management,
  Cache, Help) for managing all plugin resources.
- **Model Management** -- download, delete, and unload SAM checkpoints with
  filtering (All / Downloaded / Not Downloaded / In Memory) and
  Ultralytics + ModelScope fallback download.
- **Feature cache management** -- configurable cache directory, maximum size,
  performance mode (Balanced / Fastest / Low Memory), and clear-on-close.
- **Split-panel UI** -- the Segmentation dock widget uses a split-panel
  layout (Input/Output + Prompts on the left, Styles + Options on the right).
- **Multi-language UI** -- the plugin interface is translated into 中文,
  日本語, 한국어, Français, Русский, العربية, Deutsch, Español, and
  Português, following the QGIS locale setting automatically.

### Improvements

- PyQt6 / QGIS 4 compatibility.
- Faster prompt response after initial encoding via feature cache reuse.
- Better memory management during encoding (Balanced vs Low Memory
  strategies).
- Preserved polygon holes when saving SAM results.
- Tab key cycles prompt types (BBox -> FG -> BG) with cursor changes;
  application-level keyboard shortcuts work even when the widget is not
  focused.

### Removed

- `torchgeo` and `segment_anything` as runtime dependencies.
- `tools/torchgeo_sam.py`, `tools/sam_ext.py`, and `tools/SAMTool.py`
  (superseded by `geosam`).
- **Minimum Pixels** parameter from the Segmentation tool UI.
- **Load Demo** option (the demo image is no longer auto-loaded).
- Manual `pip` dependency installation (replaced by the Settings dialog).
