# Geo-SAM Image Encoder

```{admonition} Prerequisites
:class: note

Before using the Image Encoder, make sure you have installed the required
dependencies and downloaded a SAM model in {doc}`/Settings/index`.
```

The **Geo-SAM Image Encoder** is a QGIS Processing algorithm that
pre-processes geospatial images and generates reusable feature files using
the SAM image encoder. The generated features can then be loaded in the
{doc}`pre_encoded_segmentation` mode to label landforms interactively.

## Opening the Encoder

Click the **Geo-SAM Image Encoder** icon in the toolbar. The QGIS Processing
dialog opens with the algorithm parameters.

```{image} ../img/encoder_demo.gif
:alt: Image Encoder demo
:width: 600px
:align: center
```

## Parameters

### Required Parameters

| Parameter | Description |
|---|---|
| **Input raster layer** | The raster layer or image file to encode. |
| **Bands** | Select up to 3 bands (preferably in RGB order). If omitted, the first 1-3 bands are used. |
| **Sliding-window stride** | Determines patch overlap. Overlap = patch_size - stride. Default: 512. Range: 1-1024. |
| **GeoSAM model** | Select a model from the dropdown. The model must be downloaded first. |
| **Output feature-cache directory** | The folder where features and manifest will be saved. |
| **Use GPU if CUDA is available** | Enable GPU acceleration (default: on). |
| **Load output features in Geo-SAM tool** | Automatically load the features in the Segmentation tool after encoding (default: on). |

### Advanced Parameters

| Parameter | Description |
|---|---|
| **Target CRS** | Resample the image to a different CRS. Defaults to the original CRS. |
| **Target resolution** | Resample to a specific resolution in meters. When the input CRS is geographic (degrees), a UTM CRS is estimated automatically. Defaults to native resolution. |
| **Data value range** | Fixed `[min, max]` range to rescale to `[0, 255]`. If omitted, the tool computes min/max from raster statistics. |
| **CUDA device id** | GPU device index when multiple GPUs are available (default: 0). |
| **Encoding memory strategy** | **Balanced** (default) or **Low Memory**. See below. |

## Band Selection

SAM natively supports only three-band RGB images. Geo-SAM has been adapted to
support one or two-band images, so you can use:

- Grayscale images (1 band)
- Spectral index images like NDVI, NDWI (1 band)
- SAR images (1-2 bands)
- Standard RGB images (3 bands)

If you select no bands, the tool defaults to the first 1-3 bands.

## Value Range Rescaling

SAM expects input values in the range `[0, 255]`. Remote sensing images often
have values outside this range. The tool handles this in two ways:

- **Automatic** (default): computes the min and max values from the raster
  statistics within the processing extent and rescales `[min, max]` to
  `[0, 255]`.
- **Manual**: specify a fixed `[min, max]` range in the advanced *Data value
  range* parameter.

## Patch Sampling

SAM supports input images of size 1024 x 1024 (SAM3 uses 1008 x 1008). The
encoder handles images of any size by sampling them into overlapping patches:

- **Small images** are resized to match the input size.
- **Large images** are split into a grid of overlapped patches.

The **stride** parameter controls the overlap: `overlap = patch_size - stride`.
A smaller stride produces more overlap (more patches, slower encoding, better
edge continuity). A larger stride produces less overlap (fewer patches, faster
encoding).

## Memory Strategy

| Strategy | Description |
|---|---|
| **Balanced** | Flushes GPU/Python memory every 16 chips. Good balance of speed and memory. |
| **Low Memory** | Flushes memory after every chip. Slower but uses less peak memory. Use this when you encounter out-of-memory errors. |

## Output Structure

The encoder creates the following structure in the output directory:

```
output_directory/
├── manifest.parquet           # chip metadata (GeoDataFrame)
└── <layer_name>/
    └── features/
        ├── chip_000000.pt     # encoded feature for chip 0
        ├── chip_000001.pt     # encoded feature for chip 1
        └── ...
```

The `manifest.parquet` contains per-chip metadata including bounds, CRS,
transform, shape, model type, and checkpoint path. This allows the
Segmentation tool to load features efficiently.

## After Encoding

If **Load output features in Geo-SAM tool** is enabled (default), the
Segmentation tool opens automatically and loads the generated feature folder.
You can start labelling immediately.

If you disabled auto-loading, open the Segmentation tool, switch to
**Pre-encoded** mode, and load the output folder manually.

## Tips for Faster Encoding

- Choose a **smaller processing extent** instead of encoding the full image.
- **Reduce the target resolution** (in advanced parameters) to decrease the
  number of patches.
- **Increase the stride** to minimize overlap and reduce patch count.
- Choose a **smaller model** (e.g., SAM2.1 Tiny instead of SAM2.1 Large).
- **Use a GPU** when enough accelerator memory is available.
- Use the **Low Memory** strategy when peak memory matters more than speed.

## Model Support for Feature Reuse

Most SAM models support reusable features (encoding once, querying many
times). If a model does not support feature reuse, the encoder will report an
error. SAM3 models use a different image size (1008 x 1008) which is handled
automatically.
