# Geo-SAM Segmentation (QGIS plugin)

Click the `Geo-SAM Segmentation` icon to open the interactive segmentation widget. You will be shown a demo raster image with thaw slump and small pond landforms for you to try the tool. With a single click on the map, a segmentation result will be generated.

```{image} ../img/try_geo_sam.gif
:alt: try_geo_sam
:width: 500px
:align: center
```

A user interface will be shown below.

```{image} ../img/ui_geo_sam.png
:alt: Geo SAM UI
:width: 600px
:align: center
```

## Add Prompts

Click the buttons to select between the `Foreground(FG)` and `Background(BG)` points. Use `FG` points to add areas you desire, and use `BG` points to remove areas you don't want.

Click the `BBox` button to choose to add a bounding box (BBox) on the canvas by pressing and dragging the mouse. BBox can be used together with adding points or independently.

## Save Current Results

After adding points and a BBox for segmenting a subject, you can save the segmentation results by clicking the `Save` button.

## Undo/Clear Prompts

You can use the `Undo` button to undo the last point or BBox prompt.

You can use the `Clear` button to clear the added points and BBox.

## Enable/Disable the Tool

You can uncheck the `Enable` button to temporally disable the tool and navigate on the map.

## Load Image Features

The plugin is initialized with features for demo purposes, and you can use the `Feature Folder` selection button to select the folder that includes the image features you need.

```{image} ../img/Select_feature_folder.png
:alt: Select feature folder
:width: 250px
:align: center
```

Then, press the `Load` button to load the selected image features. Remember to add the corresponding raster image to the QGIS project.

## Shortcuts

- `Tab`: loop between 3 prompt types (the cursor will also change to the corresponding types)
- `P`: Toggle to enable/disable executing SAM with `Preview mode`
- `C`: clear all prompts in canvas [same as `Clear` button]
- `Z`: undo the last prompt in canvas [same as `Undo` button]
- `S`: save SAM output features into polygon [same as `Save` button]
- `Ctrl+Z` or `command+Z`: undo the last saved segmentation results

## Tips for Using the Segmentation Tool

- Deal with only **One Subject** each time
- Use **Background Points** to exclude unwanted parts
- Use **Bounding Box (BBox)** to limit the segment polygon boundary
- The **BBox** should cover the entire subject
- Remember to press the `Save` button after the segmentation of the chosen subject
