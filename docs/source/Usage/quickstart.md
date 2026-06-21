# Quick Start (5 Minutes)

This guide walks you through segmenting landforms using the **Live Encoding**
workflow -- the fastest way to get started with Geo-SAM.

## Prerequisites

1. **QGIS** 3.x/4.x installed.
2. **Geo-SAM plugin** installed and activated (see {doc}`/Installation`).
3. **Dependencies** installed via Settings > Dependencies > Install Missing.
4. **A SAM model** downloaded via Settings > Model Management (e.g.,
   **SAM2.1 Base**).
5. **A raster image** loaded in your QGIS project.

## Step 1: Open the Segmentation Tool

Click the **Geo-SAM Segmentation** icon in the toolbar. A dock widget appears
at the top of the QGIS window with a split-panel layout:

- **Left side**: Input / Output and Prompts tabs
- **Right side**: Styles and Options tabs

```{image} ../img/ui_geo_sam.png
:alt: Plugin menu
:width: 100%
:align: center
```

## Step 2: Select the Live Encoding Mode

In the **Input / Output** tab, ensure the source selector dropdown shows
**Live Encoding** (this is the default). Then:

1. Select a SAM model from the **Model** dropdown (e.g., *SAM2.1 Base*).
2. Select your raster layer from the **Layer** dropdown.
3. Under **Segmentation Result**, select an existing polygon layer or click
   **Load/Create** to specify a new Shapefile.

```{admonition} Tip
:class: tip

If you have not downloaded the selected model, Geo-SAM will prompt you to
open Model Management and download it.
```

## Step 3: Draw a Bounding Box

1. Go to the **Prompts** tab.
2. Click the **BBox** button to activate the bounding-box tool.
3. Drag a rectangle around the landform you want to segment. As you drag,
   the preview shows the landform inside the box segmented in real-time
   with well-defined boundaries.

The first drag triggers **background encoding** of the visible image area.
A progress bar appears in the QGIS task manager. Once encoding completes,
the segmentation result appears instantly.

```{admonition} First prompt takes longer
:class: note

The first prompt on a new area requires encoding the image features (a few
seconds to minutes depending on image size and hardware). Subsequent prompts
in the same area load from cache at **millisecond** speed.
```

## Step 4: Refine with More Prompts

- **Add a background point**: Click **BG**, then click on an area you want to
  *exclude* from the segmentation.
- **Add a foreground point**: Click **FG**, then click on an area you want to
  *include* in the segmentation.
- **Undo the last prompt**: Press **Z** or click **Undo**.
- **Clear all prompts**: Press **C** or click **Clear**.

You can combine multiple prompts to fine-tune the segmentation of a single
object.

## Step 5: Save the Result

When you are satisfied with the segmentation, press **S** or click **Save** to
save the current segmentation polygon.

## Step 6: Move to the Next Object

Press **C** to clear prompts, then repeat Steps 3-5 for the next object.

```{admonition} Remember
:class: warning

SAM segments **one object at a time**. Always save your results before
clearing prompts and moving to the next object.
```

## What's Next?

- {doc}`live_encoding_segmentation` -- full reference for Live Encoding mode
- {doc}`pre_encoded_segmentation` -- Pre-encoded mode
- {doc}`encoding` -- how to pre-encode images with the Image Encoder
- {doc}`/Settings/index` -- configure cache, performance, and models
