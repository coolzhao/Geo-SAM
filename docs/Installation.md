
# Installation

## Install QGIS

You are suggested to install the latest version of [QGIS](https://www.qgis.org/en/site/forusers/download.html) since the plugin has mainly been tested on versions newer than QGIS 3.30 (QGIS 3.28 LTR should also work fine).

## Install Library Dependencies

Some dependencies need to be installed into the Python environment in QGIS beforehand to use Geo-SAM. `Pytorch` is a fundamental dependency. If you want to install the GPU version of `Pytorch`, it is recommended to refer to the official website for installation: <https://pytorch.org/>

After installing `PyTorch`, `torchgeo` and `segment-anything` need to be installed subsequently. Below are tutorials for installing these dependencies on different operating systems.

### For Windows Users

Open the **OSGeo4W Shell** ![OsGeo4WShell](img/OsGeo4WShell.png) application from the Start menu, which is a dedicated shell for the QGIS. Then run the following command to install the libraries.

```bash
pip3 install torch torchvision
pip3 install torchgeo
pip3 install segment-anything
```

Our encoder tool now supports using CUDA GPU to accelerate the encoding process. If your PC has dedicated CUDA GPUs, you can install the CUDA library first and then install the gpu-version pytorch using the following command (using CUDA version 11.7 as an example):

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 
# or if you have installed the CPU version before.
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --force-reinstall
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


```{image} img/Active_geo_sam.png
:alt: Plugin menu
:width: 600px
:align: center
```

After activating the Geo SAM plugin, you may find the Geo SAM tools under the `Plugins` menu,

```{image} img/Plugin_menu_geo_sam.png
:alt: Plugin menu
:width: 350px
:align: center
```

You may also find a new toolbar, including two icons.

```{image} img/Toolbar_geo_sam.png
:alt: Plugin toolbar
:width: 200px
:align: center
```
