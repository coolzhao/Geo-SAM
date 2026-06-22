# FAQ & Troubleshooting

This page answers common questions and helps you solve problems you may
encounter while using Geo-SAM.

```{tip}
If you don't find your answer here, check the
[GitHub Issues](https://github.com/coolzhao/Geo-SAM/issues) or open a new
issue with details about your setup.
```

## Installation & Dependencies

:::{dropdown} The dependency installation is taking a long time
:open:
:color: info
:icon: question

The first installation downloads PyTorch, which is a large package (several
hundred MB). This is normal and depends on your network speed.

Wait for the installation to complete. The log area in the Dependencies tab
shows real-time progress. Subsequent launches use the cached installation and
load much faster.
:::

:::{dropdown} Dependency installation failed
:open:
:color: info
:icon: question

Check the log area at the bottom of the Dependencies tab for the error
message. Common causes:

- **Network issues**: Ensure you have a stable internet connection. If you
  are behind a proxy, configure pip accordingly.
- **Python version mismatch**: Geo-SAM requires Python 3.9+. PyTorch 2.11+
  requires Python 3.10+.
- **Disk space**: Ensure you have at least 3 GB of free disk space for
  dependencies.

If the installation is corrupted, use **Clear Current Runtime** or **Clear
All Runtimes** in the Dependencies tab, then click **Install Missing** again.
:::

:::{dropdown} After installing dependencies, the plugin still reports missing packages
:open:
:color: info
:icon: question

**Restart QGIS**. Dependencies are installed into a plugin-private directory
that is only recognized after QGIS restarts. The plugin prompts you to
restart after installation completes.
:::

:::{dropdown} Dependencies disappeared after updating QGIS
:open:
:color: info
:icon: question

QGIS updates may use a different Python runtime (different `sys.prefix`).
Plugin-managed dependencies are stored per-runtime, so a new QGIS version
needs a fresh install. Open Settings > Dependencies > **Install Missing**.
:::

## Models

:::{dropdown} Model download is slow or fails
:open:
:color: info
:icon: question

Model checkpoints are downloaded from Ultralytics (primary) or ModelScope
(fallback). If the download fails:

1. Check your network connection.
2. Try a different time (servers may be rate-limited).
3. Try a smaller model first (e.g., SAM2.1 Tiny at ~39 MB) to verify
   connectivity.
4. You can also download the checkpoint manually and place it in the
   **Model Folder** (configured in Settings > Model Management > Storage).
:::

:::{dropdown} "The selected model does not support reusable features"
:open:
:color: info
:icon: question

Some models (e.g., certain SAM3 variants) may not support feature reuse
(encoding once, querying many times). Use a model that supports feature
reuse (SAM, SAM2, SAM2.1 families) for the Image Encoder and Pre-encoded
mode.

The Live Encoding mode may still work with these models by
encoding on the fly.
:::

## Segmentation

:::{dropdown} The first prompt takes a long time to show a result
:open:
:color: info
:icon: question

In **Live Encoding mode**, the first prompt on a new area triggers background
encoding of the image features. This can take from a few seconds to several
minutes depending on the image size, model, and hardware. Subsequent prompts
in the same area load from the cache at millisecond speed.

To speed up the first encoding:

- Zoom in to reduce the visible area.
- Use a smaller model (e.g., SAM2.1 Tiny).
- Use a GPU (enable in the Options or use the Image Encoder with GPU).
- Pre-encode the image using the {doc}`/Usage/encoding` tool and switch to
  Pre-encoded mode.
:::

:::{dropdown} Segmentation results do not appear
:open:
:color: info
:icon: question

1. Ensure **Enable** is checked in the Options tab.
2. Ensure a valid **Layer** or **Pre-encoded** file is loaded.
3. Ensure a **Model** is selected and downloaded.
4. Check the QGIS message log (View > Panels > Log Messages) for errors.
5. Ensure the raster image CRS is valid. If the CRS is invalid, Geo-SAM
   cannot transform prompts to image coordinates.
:::

:::{dropdown} Preview mode is not showing results
:open:
:color: info
:icon: question

1. Ensure **Preview Mode** is enabled in the Options tab (or press **P**).
2. Ensure the mouse is over the loaded image area.
3. Preview mode defers bounding-box SAM until you release the mouse button
   when live encoding is required.
:::

:::{dropdown} The project CRS changed unexpectedly
:open:
:color: info
:icon: question

Geo-SAM temporarily changes the project CRS to match the active source CRS
(image layer or feature folder) so that prompts and results align correctly.
The original CRS is restored when the Segmentation tool is closed. A message
bar notification appears when this happens.
:::

## Encoding

:::{dropdown} Encoding runs out of memory
:open:
:color: info
:icon: question

1. Switch the **Encoding memory strategy** to **Low Memory**.
2. Reduce the processing extent.
3. Increase the stride to reduce the number of patches.
4. Use a smaller model.
5. Close other applications to free RAM.
:::

:::{dropdown} Encoding produces no output ("No available patch sample")
:open:
:color: info
:icon: question

The processing extent does not overlap with the raster layer, or the extent
is empty. Verify the extent in the encoder dialog and ensure it covers part
of the image.
:::

:::{dropdown} Value range error ("Data value range is invalid or constant")
:open:
:color: info
:icon: question

The specified value range has `min >= max`, or the image has constant values
in the selected bands. Either leave the value range empty (auto-detect) or
specify a valid `[min, max]` where `min < max`.
:::

## CRS & PROJ Issues

:::{dropdown} Valid EPSG codes fail to resolve ("PROJ database" errors)
:open:
:color: info
:icon: question

Geo-SAM automatically resolves rasterio's bundled PROJ database to prevent
this. If you still encounter CRS resolution failures:

1. Restart QGIS after installing dependencies.
2. Ensure no other `PROJ_LIB` or `PROJ_DATA` environment variable points to
   an incompatible PROJ database.
:::

## Cache

:::{dropdown} The cache is using too much disk space
:open:
:color: info
:icon: question

1. Reduce the **Maximum Size** in Settings > Cache. The default is 2048 MB.
   When the limit is exceeded, the oldest cached files are automatically
   removed.
2. Enable **Clear cache when the plugin closes** to free disk space after
   each session.
3. Click **Clear Cache Now** for an immediate cleanup.
:::

:::{dropdown} Clearing the cache deleted my encoded features
:open:
:color: info
:icon: question

Clearing the cache only removes the **cached copies** used for faster
reloading. It does not delete the original feature files generated by the
Image Encoder. You can always re-encode images to regenerate features.
:::

## Still Having Trouble?

```{admonition} Reporting an issue
:class: important

If none of the above solves your problem, please
[open a GitHub issue](https://github.com/coolzhao/Geo-SAM/issues) with:

- **QGIS version** and operating system.
- **Geo-SAM version** (shown in Settings > Help).
- **Error message** from the QGIS message log
  (View > Panels > Log Messages).
- **Steps to reproduce** the problem.

You can also join
[GitHub Discussions](https://github.com/coolzhao/Geo-SAM/discussions)
for questions and community support.
```
