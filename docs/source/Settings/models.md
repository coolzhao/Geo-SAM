# Model Management

The **Model Management** tab in Settings handles downloading, storing, and loading SAM model checkpoints. These checkpoints are required by both the Image Encoder and the Segmentation tool.

## Downloading a Model

1. Open **Geo-SAM Settings** and go to the **Model Management** tab.
2. Locate the model you want in the table — models not yet downloaded are marked in red.
3. Click the **Download** button in the Actions column.
4. Wait for the download to finish. Progress is shown below the table.

The model checkpoint is saved to the configured **Model Store Directory** (default: `models/` under the plugin root).

## Available Models

The plugin supports models from the SAM, SAM 2, SAM 2.1, and SAM 3 families:

| Model | Series | Approximate Size |
|---|---|---|
| SAM Base | SAM | ~375 MB |
| SAM Large | SAM | ~1.2 GB |
| SAM2 Tiny | SAM2 | ~39 MB |
| SAM2 Small | SAM2 | ~46 MB |
| SAM2 Base | SAM2 | ~170 MB |
| SAM2 Large | SAM2 | ~898 MB |
| SAM2.1 Tiny | SAM2 | ~39 MB |
| SAM2.1 Small | SAM2 | ~46 MB |
| SAM2.1 Base | SAM2 | ~170 MB |
| SAM2.1 Large | SAM2 | ~898 MB |
| SAM3 | SAM3 | — |

```{tip}
For most use cases, **SAM2.1 Base** offers a good balance between accuracy and speed. Choose **SAM2.1 Tiny** if you need faster encoding with limited GPU memory.
```

## Deleting a Model

Click the **Delete** button in the Actions column to remove a downloaded checkpoint from disk.

## Unloading Models from Memory

Click **Unload Current** to release all loaded model sessions from memory. This is useful when you want to free GPU or system memory without deleting the checkpoint files.

## Filtering the Model List

Use the filter bar above the table to show:

- **All** -- every known model.
- **Downloaded** -- models with a local checkpoint file.
- **Not Downloaded** -- models that still need to be downloaded.
- **In Memory** -- models currently loaded into memory for inference.

## Model Store Directory

The default model directory is `models/` inside the plugin folder. You can change it by clicking **Browse** in the Storage section and selecting a different folder. This is useful when you want to share model checkpoints across multiple machines or store them on a larger drive.
