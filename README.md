# Geo SAM

By Joey and [Fancy](https://github.com/Fanchengyan) from [Cryosphere Lab](https://cryocuhk.github.io/), ESSC, CUHK.

## Introduction

Geo SAM is a QGIS plugin tool that aims to help people segment, delineate or label landforms efficiently when using large-size geospatial raster images. [Segment Anything Model](https://segment-anything.com/) (SAM) is a foundation AI model with the superpower, but the model size is huge, and using it to process images can take a long time, even with a modern GPU. Our tool uses the strategies of encoding image features in advance and trimming the SAM model. The interactive segmentation process can be run in real-time on a laptop by only using a CPU, making it a convenient and efficient tool for dealing with satellite images.

The Geo SAM plugin includes two separate tools, the encoder tool for preprocessing (encoding) images and segmentation tool for interactively segmenting landforms. The encoder tool is designed to generate and save the image features using the SAM image encoder, and the encoding process only runs once per image. The segmentation tool can only be used to segment preprocessed images (whose features have been generated in advance using the encoder tool, as the included demo image).

## Installation

### Install QGIS

You are suggested to install the latest version of [QGIS](https://www.qgis.org/en/site/forusers/download.html) since the plugin has only been tested on the versions newer than QGIS 3.30.

### Install Library Dependencies

#### For Windows Users

![OsGeo4WShell](./assets/OsGeo4WShell.png)

<!-- <p align="center">
  <img src="./assets/OsGeo4WShell.png" width="100" title="OsGeo4WShell"> -->
  <!-- <img src="./assets/OsGeo4WShell.png" width="100" alt="OsGeo4WShell"> -->
<!-- </p> -->

Open the `OSGeo4W Shell` application from the Start menu, which is a dedicated shell for the QGIS. Then run the following command to install the libraries.

```bash
pip3 install torch torchvision
pip3 install torchgeo
pip3 install segment-anything
```

Our encoder tool now supports using CUDA GPU to accelerate the encoding process. If your PC has dedicated CUDA GPUs, you can install CUDA library first and then install the gpu-version pytorch using following command (using CUDA version 11.7 as an example):

```bash
# add `--force-reinstall` if you installed the cpu version before.
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

#### For Mac or Linux Users

Open your own terminal application, and change the directory to where the QGIS Python binary file locates.

```bash
# Mac
cd /Applications/QGIS.app/Contents/MacOS/bin
# Linux (Please confirm the python env by "import qgis")
cd /usr/bin
```

To confirm the QGIS Python environment:

```bash
./python3

>>> import qgis
```

Then install the libraries.

```bash
# add ./ to avoid using your default Python in the system
./pip3 install torch torchvision
./pip3 install torchgeo
./pip3 install segment-anything
```

For Linux users, if `pip3` is not found in `/usr/bin`, try the following commands:

```bash
sudo apt-get update
sudo apt-get install python3-pip
```

For Linux users, if your computer got available CUDA GPUs and with CUDA library installed, the above commands should have helped you installed the gpu-version pytorch. You can reach to pytorch official website for more information.

### Install the GeoSAM Plugin

Download the [plugin zip file](https://github.com/coolzhao/Geo-SAM/archive/refs/heads/dev.zip), unzip it, and put the `Geo-SAM` folder (please remove the version suffix of the folder to avoid potential path issues, be aware of undesired nested folders after unzipping) into the QGIS plugin folder, then restart QGIS if it's open already.

#### How to Locate the QGIS Plugin folder

From the `Settings` Menu, select `User Profiles`, then select `Open active profile folder.`  You'll be taken straight to the profile directory in Explorer or Finder. Under the profile folder, you may find a `python` folder; the `plugins` folder should be right inside the `python` folder. Open the `plugins` folder, then put the entire `Geo-SAM` folder in it, then restart QGIS.

Below are some general paths of different systems for your reference.

```bash
# Windows
%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins
# Mac
~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins
# Linux
~/.local/share/QGIS/QGIS3/profiles/default/python/plugins
```

#### Activate Geo SAM Plugin

After restarting QGIS, you may go to the `Plugins` menu, select `Manage and Install Plugins`, and under `Installed`, you may find the `Geo SAM` plugin; check it to activate the plugin.

![active geo sam](assets/Active_geo_sam.png)

#### Find the Geo SAM Tool

After activating the Geo SAM plugin, you may find the Geo SAM tool under the `Plugins` menu,

<p align="center">
  <img src="assets/Plugin_menu_geo_sam.png" width="350" title="Plugin menu">
</p>

You may also find a new toolbar including two icons.

<p align="center">
  <img src="assets/Toolbar_geo_sam.png" width="200" title="Plugin toolbar">
</p>

## Use the Geo SAM Segmentation Tool

Click the segmentation tool icon to open the interactive segmentation widget. You will be shown a demo raster image with thaw slump and small pond landforms for you to try the tool. With a single click on the map, a segmentation result will be generated.

<!-- ![try geo sam](assets/try_geo_sam.png) -->

<p align="center">
  <img src="assets/try_geo_sam.gif" width="500" title="Try Geo SAM">
</p>

A user interface will be shown below.

<!-- ![ui_geo_sam](assets/ui_geo_sam.png) -->

<p align="center">
  <img src="assets/ui_geo_sam.png" width="600" title="Geo SAM UI">
</p>

### Add Points

Click the buttons to select between the `Foreground(FG)` and `Background(BG)` points. Use `FG` points to add areas you desire, and use `BG` points to remove areas you don't want.

### Add Bounding Box (BBox)

Click the `BBox` button to activate the BBox tool to draw a rectangle on the map for segmenting a subject.
The BBox tool can be used together with adding points or independently.

### Save Current Results

After adding points and a rectangle for segmenting a subject, you can save the segmentation results by clicking the `Save` button.

### Clear Points and BBox

You can use the `Clear` button to clear the added points and rectangles.

### Undo the Last Prompt

You can use the `Undo` button to undo the last point or rectangle Prompt.

### Enable/Disable the Tool

You can uncheck the `Enable` button to temporally disable the tool and navigate on the map.

### Load Selected Image Features

The plugin is initialized with features for demo purposes, and you can use the `Feature Folder` selection button to select the folder that includes the image features you need.

<p align="center">
  <img src="assets/Select_feature_folder.png" width="250" title="Select feature folder">
</p>

Then, press the `Load` button to load the selected image features. Remember to add the corresponding raster image to the QGIS project.

### Shortcuts

- `Tab`: loop between 3 prompt types (the cursor will also change to the corresponding types)
- `C`: clear all prompts in canvas [same as `Clear` button]
- `Z`: undo the last prompt in canvas [same as `Undo` button]
- `S`: save SAM output features into polygon [same as `Save` button]
- `Ctrl+Z` or `command+Z`: undo the last saved segmentation results

### Tips for Using Geo-SAM Segmentation Tool

- Deal with only **One Subject** each time
- Use **Background Points** to exclude unwanted parts
- Use **Bounding Box (BBox)** to limit the segment polygon boundary
- The **BBox** should cover the entire subject
- Remember to press the `Save` button after the segmentation of the chosen subject

## Use the Geo SAM Encoder Tool

If you want to try your own images, you can use the Encoder Tool to preprocess the images. You need to download the SAM checkpoints in advance using the following links.

- `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

The following animation shows how to use the encoder tool.

<p align="center">
  <img src="assets/encoder_demo.gif" width="600" title="Try Geo SAM">
</p>

After processing the image, by default, the generated features will automatically be loaded in the segmentation tool for you to start labeling. Or you can choose to load the features manually afterwards.

## Future Works

- Existing polygon refinement

## Acknowledgement

This repo benefits from [Segment Anything](https://github.com/facebookresearch/segment-anything) and [TorchGeo](https://github.com/microsoft/torchgeo). Thanks for their wonderful work.
