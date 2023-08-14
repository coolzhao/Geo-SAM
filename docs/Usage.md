# Usage

## Use the Geo SAM Segmentation Tool

Click the `Segmentation Tool` icon to open the interactive segmentation widget. You will be shown a demo raster image with thaw slump and small pond landforms for you to try the tool. With a single click on the map, a segmentation result will be generated.


```{image} img/try_geo_sam.gif
:alt: try_geo_sam
:width: 500px
:align: center
```

A user interface will be shown below.

```{image} img/ui_geo_sam.png
:alt: Geo SAM UI
:width: 600px
:align: center
```

### Add Prompts

Click the buttons to select between the `Foreground(FG)` and `Background(BG)` points. Use `FG` points to add areas you desire, and use `BG` points to remove areas you don't want.

Click the `BBox` button to choose to add a bounding box (BBox) on the canvas by pressing and dragging the mouse. BBox can be used together with adding points or independently.

### Save Current Results

After adding points and a BBox for segmenting a subject, you can save the segmentation results by clicking the `Save` button.

### Undo/Clear Prompts

You can use the `Undo` button to undo the last point or BBox prompt.

You can use the `Clear` button to clear the added points and BBox.

### Enable/Disable the Tool

You can uncheck the `Enable` button to temporally disable the tool and navigate on the map.

### Load Image Features

The plugin is initialized with features for demo purposes, and you can use the `Feature Folder` selection button to select the folder that includes the image features you need.

```{image} img/Select_feature_folder.png
:alt: Select feature folder
:width: 250px
:align: center
```


Then, press the `Load` button to load the selected image features. Remember to add the corresponding raster image to the QGIS project.

### Shortcuts

- `Tab`: loop between 3 prompt types (the cursor will also change to the corresponding types)
- `P`: Toggle to enable/disable executing SAM with `Preview mode`
- `C`: clear all prompts in canvas [same as `Clear` button]
- `Z`: undo the last prompt in canvas [same as `Undo` button]
- `S`: save SAM output features into polygon [same as `Save` button]
- `Ctrl+Z` or `command+Z`: undo the last saved segmentation results

### Tips for Using the Segmentation Tool

- Deal with only **One Subject** each time
- Use **Background Points** to exclude unwanted parts
- Use **Bounding Box (BBox)** to limit the segment polygon boundary
- The **BBox** should cover the entire subject
- Remember to press the `Save` button after the segmentation of the chosen subject

## Use the Geo SAM Encoding Tool

If you want to try your own images, you can use the Encoding Tool. This tool helps to preprocess geospatial images and generate image features using the SAM image encoder. The generated image features can then be used in our Geo-SAM tool to label the landforms by adding points and bounding box prompts.

### Download SAM Checkpoints

SAM model checkpoints should be downloaded in advance, and three versions (huge, large, and base) are available. The large version "vit_l" is recommended to try first. You need to specify the model type that matches the checkpoint version. Using the following links to download the checkpoints.

- `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

### Select Bands and Value Range for Processing

After selecting the raster layer or image file you want to process, you should also choose the proper bands. The SAM natively supports only three-band RGB images, but we have adapted the tool to support one or two-band images so that you can try grayscale images or NDVI spectral index images.

The values of the image input to the SAM should range from 0 to 255, and you may need to specify the value range (in `Advanced Parameters`) to be rescaled to [0, 255]. By default, the tool will help you to find the min and max values of the image data and rescale the value range of [min, max] to [0, 255].

### Patch Sampling

Since SAM only supports input images with sizes of (1024, 1204), small images will be resized to match the input size, while large images will be sampled into overlapped patches (patch_size=1024) in a grid-like fashion. The stride parameter will determine the overlap behavior, overlap = patch_size - stride.

### Demo Animation

The following animation shows how to use the encoder tool.

```{image} img/encoder_demo.gif
:alt: Try Geo SAM
:width: 600px
:align: center
```


After processing the image, by default, the generated features will automatically be loaded in the segmentation tool for you to start labeling. Or you can choose to load the image features manually afterward.

### Tips for Making the Encoding Process Faster

- Choose a smaller processing extent
- Reduce target resolution (in `Advanced Parameters`)
- Increase stride to minimize overlap
- Choose a smaller version of SAM model
- Use GPU
- Increase batch_size when using a GPU with sufficient GPU memory

## Future Works

- Existing polygon refinement

