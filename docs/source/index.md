# Welcome to Geo-SAM's documentation

:::{note}
🎉 **GeoSAM is now on the [QGIS Official Plugin Repository](https://plugins.qgis.org/)!**
Install and update Geo-SAM directly from the QGIS Plugin Manager --
**Plugins > Manage and Install Plugins**, search for "Geo SAM", and click
Install. No manual download needed.
:::

By [Zhuoyi Zhao](https://github.com/coolzhao/) (Joey) and [Chengyan Fan](https://github.com/Fanchengyan) (Fancy) from [Cryosphere Lab](https://cryocuhk.github.io/), ESSC, CUHK.

## Introduction

Geo-SAM is a QGIS plugin that helps you segment, delineate, and label
landforms in large geospatial raster images. It is built on the
[Segment Anything Model](https://segment-anything.com/) (SAM) and uses the
strategies of encoding image features in advance and stripping the heavy image
encoder from inference. **The interactive segmentation runs at `millisecond`
speeds on a laptop CPU**, making it a convenient and efficient tool for remote
sensing image analysis.

Since **v2.0**, Geo-SAM provides two segmentation workflows:

- **Live Encoding mode** -- Select a raster layer and a SAM model,
  then start segmenting. The plugin encodes features on the fly using QGIS
  background tasks and caches them for fast re-queries. This is the
  recommended workflow for most users.

- **Pre-encoded mode** -- Use the Image Encoder to generate
  reusable feature files from a raster, then load them in the Segmentation
  tool. Ideal when segmenting the same image repeatedly or when encoding on
  a remote server.

```{figure} img/Geo_SAM.png
:width: 100%
:alt: Geo-SAM workflow comparison

Comparison of the workflow between Geo-SAM and the original SAM. The
original SAM encodes prompts and image simultaneously, while Geo-SAM
encodes image features in advance and queries prompts at `millisecond`
speeds by loading those saved features.
```

## Reasons for choosing Geo-SAM

- **QGIS-based** -- cross-platform GUI with no programming skills required.
- **Fast feedback** -- segmentation appears `instantly after giving
  prompts`, and can even follow the mouse cursor in `Preview mode` for a
  smooth, interactive labelling experience.
- **Dual workflow** -- segment directly from an image layer or from
  pre-encoded feature files.
- **Multi-band support** -- adapted to handle one or two-band images
  (grayscale, NDVI, NDWI, SAR) in addition to standard three-band RGB.
- **SAM model family support** -- SAM, SAM 2, SAM 2.1, and SAM 3.
- **Multi-language UI** -- the interface is available in 中文, 日本語, 한국어,
  Français, Русский, العربية, Deutsch, Español, and Português, following the
  QGIS locale setting automatically.

```{note}
- SAM is designed to **segment one object at a time** with a series of
  prompts. Save the current results before moving to the next object.
- The Geo-SAM plugin is in active development. For questions or
  suggestions, please open an issue on
  [GitHub](https://github.com/coolzhao/Geo-SAM/issues).
```

```{toctree}
:maxdepth: 2
:caption: Contents:
:hidden:

Installation
Usage/index
Settings/index
faq
more
```
