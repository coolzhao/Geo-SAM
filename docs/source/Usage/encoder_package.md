# GeoSAM Image Encoder Package (Legacy)

```{admonition} Superseded by the `geosam` library
:class: warning

The standalone `GeoSAM-Image-Encoder` Python package has been **superseded**
by the [`geosam`](https://github.com/Fanchengyan/geosam) core library as of
GeoSAM v2.0. The information below is retained for legacy reference only.

For new projects, use the `geosam` library directly. The QGIS plugin now uses
`geosam` as its backbone for both encoding and segmentation.
```

## Overview

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

## Migration to `geosam`

If you are starting a new project, use the `geosam` library directly:

```python
from geosam import RasterDataset, build_model_adapter

dataset = RasterDataset("image.tif", indexes=[1, 2, 3], crs="EPSG:32650")
adapter = build_model_adapter(model_spec)

for chip_bounds in chip_rectangles:
    sample = dataset[chip_bounds]
    model_image = sample.to_model_image(value_range=(0, 255))
    encoded = adapter.encode_image(model_image)
    encoded.save("features/chip_000000.pt")
```

See the [`geosam` documentation](https://github.com/Fanchengyan/geosam) for
the full API reference.

## Legacy Colab Example

The original Colab notebook is still available:

<https://colab.research.google.com/github/Fanchengyan/GeoSAM-Image-Encoder/blob/main/examples/geosam-image-encoder.ipynb>
