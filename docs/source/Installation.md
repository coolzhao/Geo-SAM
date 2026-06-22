# Installation

Total steps: **5**

- [Installation](#installation)
  - [Step 1: Install QGIS](#step-1-install-qgis)
  - [Step 2: Install the Geo-SAM Plugin](#step-2-install-the-geo-sam-plugin)
  - [Step 3: Install Dependencies](#step-3-install-dependencies)
  - [Step 4: Download a SAM Model](#step-4-download-a-sam-model)
  - [Step 5: Start Using Geo-SAM](#step-5-start-using-geo-sam)

---

## Step 1: Install QGIS

Install the latest version of [QGIS](https://www.qgis.org/en/site/forusers/download.html).
Geo-SAM has been tested on QGIS 3.x/4.x.

:::{admonition} Alternative: install QGIS via conda
:class: tip
:collapsible:

If you encounter issues with the official installer, you can install QGIS via
conda:

```bash
conda create -n qgis python qgis -c conda-forge -y
conda activate qgis
qgis 
# or 
# conda run qgis (if multiple qgis installed)
```
:::

## Step 2: Install the Geo-SAM Plugin

In QGIS, go to the menu **Plugins** > **Manage and Install Plugins**. Search
for **Geo SAM** in the search bar, select it, and click **Install**. After
installation, check the checkbox to activate the plugin.

```{image} img/Active_geo_sam.png
:alt: Plugin menu
:width: 90%
:align: center
```

After activating the Geo-SAM plugin, you will find the Geo-SAM tools under the
**Plugins** menu,

```{image} img/Plugin_menu_geo_sam.png
:alt: Plugin menu
:width: 60%
:align: center
```

You will also see a new toolbar with four icons:

```{image} img/Toolbar_geo_sam.png
:alt: Plugin toolbar
:width: 33%
:align: center
```

| Name | Role | Description |
|---|---|---|
| **Geo-SAM Segmentation** | **Main Tool** | Interactive segmentation with points and bounding boxes |
| **Geo-SAM Image Encoder** | **Auxiliary Tool** | Pre-encode raster images into reusable feature files |
| **Geo-SAM Encoder Copilot** | **Auxiliary Tool** | Preview patch coverage before encoding |
| **Geo-SAM Settings** | **Settings** | Manage dependencies, models, cache, and help |

## Step 3: Install Dependencies

Geo-SAM requires several Python packages (PyTorch, geosam, rasterio, etc.) to
function. These are managed **directly inside the plugin** -- no manual `pip`
commands are needed.

1. Click the **Geo-SAM Settings** icon in the toolbar.
2. Go to the **Dependencies** tab.
3. Click **Install Missing** and wait for the installation to finish.
4. **Restart QGIS** when prompted.

```{admonition} First-time setup
:class: tip

Dependencies are installed into an isolated, plugin-private directory so they
do not interfere with your QGIS Python environment.
```

See {doc}`Settings/dependencies` for details.

## Step 4: Download a SAM Model

A SAM model checkpoint is required for both image encoding and segmentation.

1. Open **Geo-SAM Settings** and go to the **Model Management** tab.
2. Select a model (e.g., **SAM2.1 Base** for a good balance of speed and
   accuracy).
3. Click **Download** and wait for the download to complete.

See {doc}`Settings/models` for the full list of available models.

## Step 5: Start Using Geo-SAM

- **5-minute quickstart**: {doc}`Usage/quickstart`
- **Full user guide**: {doc}`Usage/index`
