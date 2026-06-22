# Usage of Geo-SAM Tools

If you are new to Geo-SAM, start with the {doc}`Quick Start </Usage/quickstart>` guide
for a 5-minute walkthrough of the basic segmentation workflow.

## Choosing a Workflow Mode

Since v2.0, Geo-SAM provides two modes for segmentation workflows. This
section describes how to use each of them.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card}
:link: live_encoding_segmentation
:link-type: doc
:class-header: sd-bg-primary sd-text-center sd-text-white
:class-footer: sd-text-center

**Live Encoding mode (new)**
^^^

Select a raster layer and a SAM model, then start segmenting immediately.
Features are encoded on the fly using QGIS background tasks and cached for
fast re-queries.

- No pre-encoding step needed
- First prompt triggers background encoding
- Subsequent prompts load from cache at millisecond speed
- Supports online tile layers (XYZ, WMS)

+++

**Recommended for most users.**
:::

:::{grid-item-card}
:link: pre_encoded_segmentation
:link-type: doc
:class-header: sd-bg-primary sd-text-center sd-text-white
:class-footer: sd-text-center

**Pre-Encoded mode**
^^^

Use the Image Encoder to generate reusable feature files, then load them in
the Segmentation tool. Ideal when segmenting on a remote server.

- Encode once, segment many times
- Features saved as `.pt` files with a `manifest.parquet`
- Can be generated on Colab / AWS and loaded locally
- Fastest prompt response after encoding

+++

**Best for repeated segmentation of the same image.**
:::

::::

### Differences between Two Modes

| Aspect | Live Encoding mode | Pre-encoded mode |
|---|---|---|
| First prompt | Triggers background encoding | Instant (features pre-encoded) |
| Model selection | User selects any model | Locked to encoding model |
| Feature cache | Built on the fly | Pre-built on disk |
| Online tile layers | Supported | Not applicable |
| Best for | Quick exploration | Repeated segmentation |

## Tools Overview

::::{tab-set}

:::{tab-item} Segmentation
:sync: seg

The **Geo-SAM Segmentation** tool is the primary interactive labelling
interface. It supports both the **Live Encoding** and **Pre-encoded** modes via
a source selector dropdown.

- {doc}`quickstart` -- 5-minute guide (Live Encoding mode)
- {doc}`live_encoding_segmentation` -- full Live Encoding mode reference
- {doc}`pre_encoded_segmentation` -- full Pre-encoded mode reference
:::

:::{tab-item} Image Encoder
:sync: encode

The **Geo-SAM Image Encoder** is a QGIS Processing algorithm that generates
reusable feature files from a raster layer. It supports band selection,
sliding-window stride, memory strategy, GPU acceleration, and CRS/resolution
resampling.

- {doc}`encoding` -- Image Encoder reference
:::

:::{tab-item} Encoder Copilot
:sync: copilot

The **Encoder Copilot** helps you find the right encoding parameters by
previewing patch coverage in real-time before running the full encoder.

- {doc}`copilot` -- Encoder Copilot reference
:::

:::{tab-item} Encoder Package (Legacy)
:sync: package

The standalone `GeoSAM-Image-Encoder` Python package is superseded by the
`geosam` core library. See {doc}`encoder_package` for legacy usage notes.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

quickstart
live_encoding_segmentation
pre_encoded_segmentation
encoding
copilot
encoder_package
```
