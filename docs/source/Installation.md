
# Installation

## Install QGIS

You are suggested to install the latest version of [QGIS](https://www.qgis.org/en/site/forusers/download.html) since the plugin has mainly been tested on versions newer than QGIS 3.30 (QGIS 3.28 LTR should also work fine).

## Install Library Dependencies

### For Windows Users

<!-- ![OsGeo4WShell](./img/OsGeo4WShell.png) -->

<p align="left">
  Open the <b>OSGeo4W Shell</b>
  <img src="./img/OsGeo4WShell.png" width="100" title="OsGeo4WShell"> application from the Start menu, which is a dedicated shell for the QGIS. Then run the following command to install the libraries.
</p>

```bash
pip3 install torch torchvision
pip3 install torchgeo
pip3 install segment-anything
```

Our encoder tool now supports using CUDA GPU to accelerate the encoding process. If your PC has dedicated CUDA GPUs, you can install the CUDA library first and then install the gpu-version pytorch using the following command (using CUDA version 11.7 as an example):

```bash
# add `--force-reinstall` if you installed the cpu version before.
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### For Mac or Linux Users

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
# !important, add ./ to avoid using your default Python in the system
./pip3 install torch torchvision
./pip3 install torchgeo
./pip3 install segment-anything
```

For Linux users, if `pip3` is not found in `/usr/bin`, try the following commands:

```bash
sudo apt-get update
sudo apt-get install python3-pip
```

For Linux users, if your computer got available CUDA GPUs and with CUDA library installed, the above commands should have helped you install the gpu-version pytorch. You can reach [pytorch official website](https://pytorch.org/get-started/locally/) for more information.

## Install the Geo SAM Plugin

### Download the Plugin

Download the [plugin zip file](https://github.com/coolzhao/Geo-SAM/releases/latest), unzip it, and rename the folder as `Geo-SAM` (be aware of undesired nested folders after unzipping).

### Locate the QGIS Plugin folder

In QGIS, Go to the menu `Settings` > `User Profiles` > `Open active profile folder.`  You'll be taken straight to the profile directory. Under the profile folder, you may find a `python` folder; the `plugins` folder should be right inside the `python` folder (create the `plugins` folder if it does not exist). Put the entire `Geo-SAM` folder inside the `plugins` folder, then restart QGIS. The directory tree structure should be the same as the following.

```bash
python
└── plugins
    └── Geo-SAM
        ├── checkpoint
        ├── docs
        ├── ...
        ├── tools
        └── ui
```

Below are some general paths of the plugin folder for your reference.

```bash
# Windows
%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins
# Mac
~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins
# Linux
~/.local/share/QGIS/QGIS3/profiles/default/python/plugins
```

### Activate the Geo SAM Plugin

After restarting QGIS, go to the menu `Plugins` > `Manage and Install Plugins`, and under `Installed`, you may find the `Geo SAM` plugin; check it to activate the plugin.

<p align="center">
  <img src="img/Active_geo_sam.png" width="600" title="Plugin menu">
</p>

After activating the Geo SAM plugin, you may find the Geo SAM tools under the `Plugins` menu,

<p align="center">
  <img src="img/Plugin_menu_geo_sam.png" width="350" title="Plugin menu">
</p>

You may also find a new toolbar, including two icons.

<p align="center">
  <img src="img/Toolbar_geo_sam.png" width="200" title="Plugin toolbar">
</p>
