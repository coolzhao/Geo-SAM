# Encoder Copilot

The **Encoder Copilot** is an interactive tool that helps you find the right
encoding parameters before running the full Image Encoder.

## Background

SAM only supports 1024 x 1024 pixel image inputs (1008 x 1008 for SAM3), but
remote sensing images are generally much larger. Geo-SAM solves this by
splitting the image into overlapping patches. After the user provides prompts,
the patch whose center is closest to the prompt center is selected for SAM
execution.

However, if the spatial resolution is too high or the land surface objects are
too large, a single 1024 x 1024 patch may not cover the target object. To
avoid this, the image resolution may need to be reduced. The **Encoder
Copilot** displays all patches under different resolutions and strides in
real-time, so you can verify coverage before encoding.

## Usage

Click the **Geo-SAM Encoder Copilot** icon in the toolbar to open the
interactive widget.

1. **Select the raster** and bands you want to encode.
2. **Set the processing extent** -- calculated from the layer or drawn
   directly on the canvas.
3. **Adjust the resolution scale and overlap** values. The corresponding
   patch coverage is displayed on the canvas in real-time.

The Copilot also supports direct setting of other parameters:

- **Model checkpoint path** -- the model type is automatically detected from
  the filename. If you renamed the checkpoint, you may need to select the
  model type manually.
- **Value range** -- click **Parse from Raster** to auto-detect, or enter
  values manually. If you leave it empty, the Image Encoder will auto-detect
  during processing.

## Export Settings

### Copy Settings to Clipboard

When the parameters are determined, click **Copy Setting** in the Encoder
Copilot. Then in the **Geo-SAM Image Encoder** dialog, click **Advanced** >
**Paste Settings** to paste the parameters.

```{image} ../img/Paste_Settings.jpg
:alt: Paste settings
:width: 600px
:align: center
```

### Export Settings to a File

You can also export settings to a JSON file by clicking **Export Settings**.
This is useful for passing parameters to the {doc}`encoder_package` or for
reproducing an encoding run on a remote server.

## Demo

The following video shows how to use the Encoder Copilot:

```{image} ../_static/EncoderCopilotCover.jpg
:alt: Encoder Copilot demo
:width: 95%
:target: https://youtu.be/NWemi3xcCd0
```
