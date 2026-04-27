
# Geo-SAM Image Encoder (QGIS plugin)

If you want to try your own images, you can use the `Geo-SAM Image Encoder` tool. This tool helps to preprocess geospatial images and generate image features using the SAM image encoder. The generated image features can then be used in our `Geo-SAM Segmentation` tool to label the landforms by adding points and bounding box prompts.

## Download GeoSAM Models

GeoSAM model checkpoints should be downloaded in advance from Geo-SAM Settings. Select the GeoSAM model in the encoder; the tool resolves the checkpoint from the configured GeoSAM model directory. If the selected model has not been downloaded, choose another downloaded model or download it in Settings and retry.

## Select Bands and Value Range for Processing

After selecting the raster layer or image file you want to process, you should also choose the proper bands. The SAM natively supports only three-band RGB images, but we have adapted the tool to support one or two-band images so that you can try grayscale images or NDVI spectral index images.

The values of the image input to the SAM should range from 0 to 255, and you may need to specify the value range (in `Advanced Parameters`) to be rescaled to [0, 255]. By default, the tool will help you to find the min and max values of the image data and rescale the value range of [min, max] to [0, 255].

## Patch Sampling

Since SAM only supports input images with sizes of (1024, 1204), small images will be resized to match the input size, while large images will be sampled into overlapped patches (patch_size=1024) in a grid-like fashion. The stride parameter will determine the overlap behavior, overlap = patch_size - stride.

## Demo Animation

The following animation shows how to use the encoding tool.

```{image} ../img/encoder_demo.gif
:alt: Try Geo SAM
:width: 600px
:align: center
```

After processing the image, by default, the generated features will automatically be loaded in the segmentation tool for you to start labeling. Or you can choose to load the image features manually afterward.

## Tips for Making the Encoding Process Faster

- Choose a smaller processing extent
- Reduce target resolution (in `Advanced Parameters`)
- Increase stride to minimize overlap
- Choose a smaller GeoSAM model
- Use GPU when enough accelerator memory is available
- Use the Low memory strategy when peak memory matters more than speed
