# Dependencies

The Geo-SAM plugin requires several Python packages to run. The **Dependencies** tab in Settings manages their installation inside an isolated, plugin-private directory — no manual `pip` commands needed.

## Installing Dependencies

1. Open **Geo-SAM Settings** and go to the **Dependencies** tab.
2. Review the dependency table — missing packages are highlighted in red.
3. Click **Install Missing** to begin installation.
4. Monitor progress in the log area at the bottom of the dialog.
5. After installation completes, **restart QGIS** for the changes to take effect.

```{note}
The first installation may take several minutes because **PyTorch** is a large package (several hundred MB). Subsequent launches use the cached installation and load much faster.
```

## Required Packages

| Package | Purpose |
|---|---|
| `torch` | PyTorch deep-learning framework |
| `torchvision` | Image transforms and utilities for PyTorch |
| `ultralytics` | SAM model loading and inference |
| `rasterio` | Reading and writing geospatial raster data |
| `geopandas` | Vector data handling (Shapefiles) |
| `shapely` | Geometric operations |
| `pyproj` | Coordinate reference system transformations |
| `geosam` | Geo-SAM core library (image encoding and segmentation) |

## Dependency Status

Each package shows one of the following statuses in the table:

- **Installed (runtime)** -- provided by the QGIS Python environment itself (e.g., `rasterio`, `geopandas`).
- **Installed (plugin-managed)** -- installed by Geo-SAM into its private dependency directory under `.deps/`.
- **Missing (installable)** -- not yet installed, but the plugin knows how to install it.

## Managing Installed Dependencies

- **Refresh Status** -- re-check all dependency availability without restarting.
- **Open Folder** -- open the current runtime's plugin-managed dependency directory.
- **Clear Current Runtime** -- remove all plugin-managed packages for the active QGIS runtime.
- **Clear All Runtimes** -- remove plugin-managed packages across all QGIS runtimes.

```{warning}
Clearing dependencies removes all plugin-installed packages. You will need to reinstall them before using Geo-SAM again.
```
