# Geo-SAM

By Joey and [Fancy](https://github.com/Fanchengyan) from [CryoLab](https://cryocuhk.github.io/), ESSC, CUHK.

## Introduction

Geo SAM is a tool that aims to help people segment, delineate or label landforms with large-size geo-spatial raster images.
[Segment Anything Model](https://segment-anything.com/) (SAM) is a foundation AI model with super power, but the model size is large and using it to process images can take a long time even with a modern GPU.
With the pre-generated image features using the Vision Transformer image encoder, the interactive segmentation process can be run in real-time on a laptop by only using CPU.

## Installation

### Install QGIS

You are suggested to install the latest version of [QGIS](https://www.qgis.org/en/site/forusers/download.html), since the plugin has only been tested on the versions later than QGIS 3.30 (at least ver. 3.28 is recommended).

## Install Library Dependencies

### For Windows Users

![OsGeo4WShell](./assets/OsGeo4WShell.png)

<!-- <p align="center">
  <img src="./assets/OsGeo4WShell.png" width="100" title="OsGeo4WShell"> -->
  <!-- <img src="./assets/OsGeo4WShell.png" width="100" alt="OsGeo4WShell"> -->
<!-- </p> -->

Open the `OSGeo4W Shell` application from Start menu, which is a dedicated shell for the QGIS. Then run the following command to install the libraries.

```bash
pip3 install torch==1.13.1 torchvision==0.14.1
pip3 install torchgeo
pip3 install segment-anything
pip3 install rasterio==1.3.7
```

### For Mac or Linux Users

Open your own terminal application, and change the directory to the QGIS Python environment.

```bash
# Mac
cd /Applications/QGIS.app/Contents/MacOS/bin
# Linux (not confirmed)
cd /<qgispath>/share/qgis/python
```

Then install the libraries.

```bash
./pip3 install torch==1.13.1 torchvision==0.14.1
./pip3 install torchgeo
./pip3 install segment-anything
./pip3 install rasterio==1.3.7
```

## Install the GeoSAM Plugin

Download the [plugin zip file](https://github.com/coolzhao/Geo-SAM/archive/refs/heads/main.zip), unzip it (avoid nested folder after unzipping) and put the contents in the QGIS plugin folder, then restart QGIS.

### How to Locate the QGIS Plugin folder

From the `Settings` Menu, `User Profiles`, select `Open active profile folder.`  You'll be taken straight to the profile directory in Explorer or Finder. Under the profile folder you may find a `python` folder, the plugins folder should be right inside the python folder. Open the `plugins` folder, then put the entire `Geo-SAM`(or `Geo-SAM-main`) folder in it, then restart QGIS.

Below are some general paths of different systems for your reference.

```bash
# Windows
%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins
# Mac
~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins
# Linux
~/.local/share/QGIS/QGIS3/profiles/default/python/plugins
```

### Activate Geo SAM Plugin

After restarting QGIS, you may go to the `Plugins` menu, select `Manage and Install Plugins`, under `Installed`, you may find the `Geo SAM` plugin, check it to activate the plugin.

![active geo sam](assets/Active_geo_sam.png)

### Find the Geo SAM Tool

After activating the Geo SAM plugin, you may find the tool under the `Plugins` menu, 

![Plugin menu geo sam](assets/Plugin_menu_geo_sam.png)

or somewhere on the toolbar near the python plugin.

![Toolbar geo sam](assets/Toolbar_geo_sam.png)

## Use the GeoSAM Tool

Click the toolbar icon to open the widget of the tool. You will be shown a demo raster image with Thaw Slump and small pond landforms for you to try the tool. With a single click, a segmentation result will be generated.

![try geo sam](assets/try_geo_sam.png)


A user interface will be shown as below.

![ui_geo_sam](assets/ui_geo_sam.png)

### Add Points

Click the buttons to select between foreground and background points. Use Foreground points to add areas you desire, and use Background points to remove areas you don't want.

### Add Bounding Box (BBox)

Use rectangle tool to segment subject with a BBox, this can be used together with adding points or independently.

### Save Current Results

After adding points and rectangle for segmenting a subject, you can save the segmentation results by click the `Save` button.

### Clear Points and BBox

You can use the `Clear` button to clear the added points and rectangles.

### Disable the tool

You can uncheck the `enable` button to temporally disable the tool and navigate on the map.

## Tips for Using GeoSAM Tool

- Deal with only **One Subject** each time
- Use **Background Points** to exclude unwanted parts
- Use **Bounding Box (BBox)** to limit the segment polygon boundary
- The **BBox** should cover the entire subject


## Future Works

- Image encoder module
- Existing polygon refinement

## Acknowledgement

This repo benefits from Segment Anything and TorchGeo. Thanks for their wonderful works.
