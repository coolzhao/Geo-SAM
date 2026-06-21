# GeoSAM Image Encoder Package

```{admonition} Superseded by the geosam library
:class: warning

The standalone `GeoSAM-Image-Encoder` Python package has been **superseded**
by the [`geosam`](https://github.com/Fanchengyan/geosam) core library since
Geo-SAM v2.0.

**Use the `geosam` library directly** (see the migration section below). The
legacy package is no longer supported by the Geo-SAM QGIS plugin v2.0+; it
is only compatible with plugins older than v2.0.
```

---

## Migration to `geosam` (recommended)

Install the `geosam` library with all dependencies (geospatial stack + SAM
models):

```bash
pip install "geosam[all]"
```

Then use it directly in your own scripts. The recommended starting point is
the [`FeatureCacheBuilder`](https://github.com/Fanchengyan/geosam) helper,
which handles tiling, encoding, and manifest writing for you:

```python
from geosam import RasterDataset, FeatureCacheBuilder
from geosam.models import ModelSpec

dataset = RasterDataset("image.tif", crs="EPSG:32650")

model_spec = ModelSpec(
    model_type="sam2",   # "sam" | "sam2" | "sam3"
    checkpoint_path="/path/to/model.pth",
)

builder = FeatureCacheBuilder(
    dataset,
    model_spec,
    output_dir="features",
    chip_size=1024,      # default: 1024 SAM/SAM2, 1008 SAM3
    stride=512,          # 50% overlap (recommended)
)
manifest_path = builder.build()
```

`builder.build()` slices the raster into overlapping chips, runs the SAM
image encoder on each chip, saves the features under `features/features/`,
and writes `features/manifest.parquet`. The resulting folder can be loaded
directly by the Geo-SAM plugin (see the
[Pre-encoded segmentation](pre_encoded_segmentation.md) page) or by
`FeatureQueryEngine` in your own scripts.

### What each part does — and how to customize it

**`RasterDataset` — open the raster**

`RasterDataset` accepts any path readable by rasterio/GDAL: GeoTIFF, COG,
JP2, or virtual filesystem paths like `/vsicurl/https://...`. If the path
is wrong or the file cannot be opened, it raises `FileNotFoundError`.

- **`crs`** is optional. Set it only when you want the raster reprojected
  on the fly (for example, reproject a lat/lon image to UTM so you can work
  in meters). Leave it out to keep the source CRS.
- **`indexes`** selects which bands SAM will see. Band numbers start at 1.
  If you don't set it, every band in the file is used. Common cases:

  ```python
  RasterDataset("image.tif")                      # all bands
  RasterDataset("image.tif", indexes=[1])         # single-band (SAR, DEM, pan)
  RasterDataset("image.tif", indexes=[1, 2, 3])   # RGB
  RasterDataset("image.tif", indexes=[4, 1, 2])   # custom order, e.g. NIR-R-G
  ```

  Whatever you pick, the encoder reshapes it to exactly 3 channels for SAM:
  1-band is copied to 3, more than 3 bands take the first 3. So for a
  multispectral image, set `indexes` explicitly to feed SAM the bands you
  want it to see.

**`ModelSpec` — choose the model**

The `model_type` → checkpoint mapping:

| `model_type` | Checkpoints |
| --- | --- |
| `"sam"` | SAM v1 (`sam_vit_b_01ec64.pth`, `sam_vit_l_0b3195.pth`) |
| `"sam2"` | SAM2 / SAM2.1 (`sam2_hiera_*.pt`, `sam2.1_hiera_*.pt`) |
| `"sam3"` | SAM3 (`sam3.pt`) |

If you don't want to remember the mapping, let geosam infer it from the
checkpoint filename:

```python
from geosam.runtime import create_model_spec_from_checkpoint
model_spec = create_model_spec_from_checkpoint("/path/to/model.pth")
```

**`FeatureCacheBuilder` — tile and encode**

- **`chip_size`** — omit it to use the model's native image size (the
  default). You'd only set it explicitly to force a different window; for
  example, encoding very large rasters on a GPU with limited memory.
- **`stride`** is the step between windows in pixels:
  - `stride = chip_size` → no overlap. Smaller cache, but objects sitting
    on a tile border may be missed.
  - `stride = chip_size // 2` → 50% overlap. This is what the Geo-SAM plugin
    uses by default and is recommended.
  - Smaller stride = more chips = more disk space and longer encode time,
    but better border coverage.
- **`output_dir`** is where the feature folder is written. After
  `builder.build()` finishes it contains:

  ```
  features/
  ├── features/
  │   ├── chip_000000.pt
  │   ├── chip_000001.pt
  │   └── ...
  └── manifest.parquet
  ```

  The `manifest.parquet` is required — without it the plugin cannot open
  the feature folder. `FeatureCacheBuilder` writes it for you; if you ever
  build a feature folder by hand, remember to write one too.

See the [`geosam` documentation](https://github.com/Fanchengyan/geosam) for
the full API reference, including the
[`OnlineQueryEngine`](https://github.com/Fanchengyan/geosam) for single-chip
interactive queries and the
[`FeatureCacheBuilder`](https://github.com/Fanchengyan/geosam) for batched
pre-encoding.

---

The rest of this page documents the **legacy** `GeoSAM-Image-Encoder` package
for reference. These workflows are only compatible with Geo-SAM plugin versions
older than v2.0.

## Legacy Overview

The `GeoSAM-Image-Encoder` package was a standalone Python package that did
not depend on QGIS. It allowed you to encode remote sensing images into
features on a remote server (e.g., Colab or AWS) and then load them in the
Geo-SAM QGIS plugin.

## Legacy Installation

```{admonition} Install PyTorch first
:class: note

Installing `GeoSAM-Image-Encoder` directly installs the CPU version of
PyTorch. Install the appropriate PyTorch version first from
<https://pytorch.org/get-started/locally/>.
```

```bash
pip install GeoSAM-Image-Encoder
# or from source
pip install git+https://github.com/Fanchengyan/GeoSAM-Image-Encoder.git
```

## Legacy Python Usage

```python
import geosam
from geosam import ImageEncoder

# check GPU availability
geosam.gpu_available()

# encode by direct parameters
checkpoint_path = "/content/sam_vit_l_0b3195.pth"
image_path = "/content/beiluhe_google_img_201211_clip.tif"
feature_dir = "./"

img_encoder = ImageEncoder(checkpoint_path)
img_encoder.encode_image(image_path, feature_dir)
```

### Using a Settings JSON File

```python
import geosam
from geosam import ImageEncoder

setting_file = "/content/setting.json"
feature_dir = "./"

settings = geosam.parse_settings_file(setting_file)
settings.update({"feature_dir": feature_dir})
init_settings, encode_settings = geosam.split_settings(settings)

img_encoder = ImageEncoder(**init_settings)
img_encoder.encode_image(**encode_settings)
```

## Legacy Terminal Usage

```bash
image_encoder.py -i /content/image.tif -c /content/checkpoint.pth -f ./
# override settings from a file
image_encoder.py -s /content/setting.json -f ./ --stride 256 --value_range "10,255"
# see all options
image_encoder.py -h
```

## Legacy Colab Example

The original Colab notebook is still available:

<https://colab.research.google.com/github/Fanchengyan/GeoSAM-Image-Encoder/blob/main/examples/geosam-image-encoder.ipynb>
