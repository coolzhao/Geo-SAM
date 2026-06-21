# Live Encoding Mode

The **Live Encoding Mode** is the default and recommended workflow since GeoSAM
v2.0. You select a raster layer and a SAM model, then segment immediately --
no pre-encoding step is required.

````{admonition} How it works
:class: tip

1. You add the first prompt (point or bounding box) on the map.
2. Geo-SAM encodes the visible image area in a **QGIS background task**.
3. Once encoding finishes, the segmentation result appears instantly.
4. Subsequent prompts in the same area load from the **feature cache** at
   millisecond speed -- no re-encoding needed.
````

## Opening the Tool

Click the **Geo-SAM Segmentation** icon in the toolbar. The dock widget
appears with a **split-panel layout**:

| Left Panel | Right Panel |
|---|---|
| **Input / Output** tab | **Styles** tab |
| **Prompts** tab | **Options** tab |

## Input / Output Tab

### Segmentation Input

The source selector dropdown at the top chooses the input mode:

- **Live Encoding** -- select a raster layer and model (this page)
- **Pre-encoded** -- load a pre-encoded feature folder (see
  {doc}`pre_encoded_segmentation`)

When **Live Encoding** is selected:

- **Layer** dropdown -- pick a raster layer from your QGIS project. Both local
  rasters and online tile layers (XYZ, WMS) are supported.
- **Model** dropdown -- pick a downloaded SAM model. If the model is not yet
  downloaded, you will be prompted to open Model Management.
- **Zoom to** -- zoom the map canvas to the selected layer's extent.

```{admonition} Online tile layers
:class: note

Online tile layers (e.g., Google Satellite via XYZ tiles) are supported.
Geo-SAM exports the visible tiles to a local raster before encoding. The
export is limited to 4096 x 4096 pixels and a maximum of 64 tiles per
request. Zoom in to reduce the area and speed up export.
```

### Segmentation Result

- **Layer dropdown** -- select an existing polygon layer to save results into,
  or pick *empty* and click **Load/Create** to specify a new Shapefile.
- **Load/Create** -- choose a Shapefile path. The file is created if it does
  not exist.

## Prompts Tab

### Prompt Types

There are three prompt types:

| Prompt | Button | Description |
|---|---|---|
| **Foreground Point (FG)** | `FG` | Marks the desired area (the object) |
| **Background Point (BG)** | `BG` | Marks the unwanted area (to exclude) |
| **Bounding Box (BBox)** | `BBox` | Constrains the object boundary |

```{admonition} Combining prompts
:class: tip

You can add multiple prompts of each type and combine them to segment one
object. Press **Tab** to cycle between the three prompt types. The cursor
changes to match the active type.
```

### Clear / Undo

- **Clear** button (shortcut: **C**) -- removes all prompts and unsaved
  segmentation results from the canvas.
- **Undo** button (shortcut: **Z**) -- removes the last prompt and re-runs
  SAM with the remaining prompts.

### Vectorization Mode

Choose how SAM mask results are converted to polygons:

- **Pixel-Level** -- preserves the exact pixel boundaries of the mask.
  Higher fidelity, larger polygons.
- **Simplified** -- simplifies the polygon geometry for smoother boundaries
  and smaller file sizes.

### Save Results

- **Save** button (shortcut: **S**) -- saves the current segmentation polygon
  to the selected output layer.

```{admonition} One object at a time
:class: warning

SAM segments one object per prompt set. **Save your results** before clearing
prompts and starting on the next object.
```

## Styles Tab

Customize the visual appearance of prompts and segmentation results.

### Point Appearance

| Control | Description |
|---|---|
| **Point Size** | Size of foreground and background points (1.0 - 99.0) |
| **Point Type** | Icon shape for points (Circle, Cross, etc.) |

### Colors

| Color Button | Controls |
|---|---|
| **Foreground Point** | Color of foreground (FG) points |
| **Background Point** | Color of background (BG) points |
| **BBox Color** | Color of bounding boxes |
| **Polygon Color** | Color of the pressed-prompt segmentation result |
| **Preview Color** | Color of the preview (hover) segmentation result |
| **Boundary Color** | Color of the source / chip boundary outline |

## Options Tab

| Control | Description |
|---|---|
| **Enable** | Toggle the segmentation tool on/off. When off, you can navigate the map normally. |
| **Max Polygon Only** | Keep only the largest polygon from the current mask. Useful when SAM produces multiple fragments. |
| **Preview Mode** | Execute SAM in real-time as you move the mouse, showing results before you click. Toggle with **P**. |
| **Show Boundary** | Show/hide the source extent and chip boundary outlines on the canvas. |
| **Reset** | Reset all settings (colors, options, model selection) to defaults. |

## Preview Mode

In **Preview mode**, SAM runs as you move the mouse, displaying the
segmentation result in real-time. This lets you find the best prompt position
before committing a click.

```{admonition} Preview vs. pressed prompts
:class: note

- **Preview results** (from mouse movement) are shown with the *Preview
  Color* and are **not saved**.
- **Pressed results** (from clicking) are shown with the *Polygon Color* and
  **can be saved** with **S**.
```

Only pressed prompts are included when you save. This means you can explore
freely in preview mode without affecting your saved results.

```{image} ../img/PreviewModeDemo.gif
:alt: Preview mode demo
:width: 500px
:align: center
```

## Keyboard Shortcuts

| Key | Action |
|---|---|
| **Tab** | Cycle between prompt types (BBox -> FG -> BG) |
| **P** | Toggle Preview mode on/off |
| **C** | Clear all prompts (same as Clear button) |
| **Z** | Undo the last prompt (same as Undo button) |
| **S** | Save current segmentation (same as Save button) |
| **Ctrl+Z** / **Cmd+Z** | Undo the last saved segmentation polygon |

## Background Encoding & Caching

When you add the first prompt on a new image area, Geo-SAM:

1. Identifies the chips (1024 x 1024 patches for SAM/SAM2, 1008 x 1008 for
   SAM3) overlapping your prompt.
2. Encodes those chips in a **QGIS background task** (visible in the task
   manager).
3. Caches the encoded features to disk (if caching is enabled).
4. Runs the SAM prompt query on the encoded features.

Subsequent prompts in the same area skip encoding and load directly from the
cache, giving **millisecond-level** response times.

```{admonition} Managing the cache
:class: seealso

The feature cache can be configured in {doc}`/Settings/cache` -- enable/disable,
set maximum size, choose performance mode, and clear cache.
```

## Performance Modes

The cache performance mode (configured in Settings > Cache) affects how
features are loaded:

| Mode | Description |
|---|---|
| **Balanced** | Compromise between speed and memory usage (default) |
| **Fastest** | Prioritizes loading speed, may use more memory |
| **Low Memory** | Minimizes memory usage at the cost of slower loading |

## Tips for Effective Segmentation

- Deal with **one object** at a time.
- Use **Background Points** to exclude unwanted parts.
- Use a **Bounding Box** to limit the polygon boundary -- the BBox should
  cover the entire object.
- Enable **Preview Mode** to find the best prompt positions before clicking.
- Enable **Max Polygon Only** if SAM produces unwanted small fragments.
- **Save** after each object before moving to the next.
