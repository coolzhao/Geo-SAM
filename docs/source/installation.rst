
Installation
============

Install QGIS
------------

You are suggested to install the latest version of `QGIS <https://www.qgis.org/en/site/forusers/download.html>`_ since the plugin has mainly been tested on versions newer than QGIS 3.30 (QGIS 3.28 LTR should also work fine).


Install Python Dependencies
---------------------------

Some dependencies need to be installed into the Python environment in QGIS beforehand to use Geo-SAM. ``Pytorch`` is a fundamental dependency. If you want to install the GPU version of ``Pytorch``, it is recommended to refer to the official website for installation: `<https://pytorch.org/get-started/locally/#start-locally>`_ .

After installing ``PyTorch``, ``torchgeo`` and ``segment-anything`` need to be installed subsequently. Below are tutorials for installing these dependencies on different operating systems.

For Windows Users
~~~~~~~~~~~~~~~~~

.. |OsGeo4WShell| image:: img/OsGeo4WShell.png
    :alt: OsGeo4WShell

Open the **OSGeo4W Shell** |OsGeo4WShell| application from the Start menu, which is a dedicated shell for the QGIS. Then run the following command to install the libraries.

.. code-block:: bash

    pip3 install torch torchvision torchgeo segment-anything

``Geo-SAM Encoder Tool`` now supports using CUDA GPU to accelerate the encoding process. If your PC has dedicated CUDA GPUs, you can install the CUDA library first and then install the gpu-version pytorch using the following command (using CUDA version 11.7 as an example):

.. code-block:: bash

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

.. note::
    If you have installed the CPU version of PyTorch before, you may need to add the ``--force-reinstall`` option to force the reinstallation of the GPU version.

    .. code-block:: bash
        
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --force-reinstall



For Mac/Linux Users
~~~~~~~~~~~~~~~~~~~~

Open your own terminal application, and change the directory to where the QGIS Python binary file locates.

.. code-block:: bash

    # Mac
    cd /Applications/QGIS.app/Contents/MacOS/bin

    # Linux
    cd /usr/bin


.. important::
    Do not ignore the ``./`` before ``python`` and ``pip`` in the following commands to avoid using your default Python/pip in the system


Run the following command to check if the QGIS Python environment is correctly set up.

.. code-block:: bash
    
    ./python3
    >>> import qgis

Then install the Python Dependencies of Geo-SAM.

.. code-block:: bash

    ./pip3 install torch torchvision torchgeo segment-anything



For Linux users, if ``pip3`` is not found in ``/usr/bin``, try the following commands:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install python3-pip


For Linux users, if your computer got available CUDA GPUs and with CUDA library installed, the above commands should have helped you install the gpu-version pytorch. You can reach `pytorch official website <https://pytorch.org/get-started/locally/>`_ for more information.

.. warning::
    If QGIS 3.34/3.36 crash when you try to run the plugin, you may need to install the ``rtree`` package using the system package manager instead of using pip. More details can be found in blog `Crash on QGIS 3.34/3.36 <https://geo-sam.readthedocs.io/en/latest/blog/2024/05-02_crash_on_QGIS.html>`_. 
    
    Below is a brief guide for Ubuntu/Debian users:

    .. code-block:: bash
        
        # Remove the rtree package using pip in QGIS:
        ./pip3 uninstall rtree
        
        # Install the rtree package using system package manager:
        sudo apt-get install python3-rtree

Install the Geo-SAM Plugin
--------------------------

Download the Plugin
~~~~~~~~~~~~~~~~~~~

Download the ``stable version``: `plugin zip file <https://github.com/coolzhao/Geo-SAM/releases/tag/v1.1.1>`_ or the ``dev version`` (more features and capabilities, but not rigorous tested): `plugin zip file <https://github.com/coolzhao/Geo-SAM/releases/tag/v1.3.1-dev>`_, unzip it, and rename the folder as ``Geo-SAM`` (be aware of undesired nested folders after unzipping).


Locate the QGIS Plugin folder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In QGIS, go to the menu ``Settings`` > ``User Profiles`` > ``Open active profile folder``.  You'll be taken straight to the profile directory. Under the profile folder, you may find a ``python`` folder; the ``plugins`` folder should be right inside the ``python`` folder (create the ``plugins`` folder if it does not exist). Put the entire ``Geo-SAM`` folder inside the ``plugins`` folder, then restart QGIS. The directory tree structure should be the same as the following.


.. code-block:: bash

    python
    └── plugins
        └── Geo-SAM
           ├── checkpoint
           ├── docs
           ├── ...
           ├── tools
           └── ui


Below are some general paths of the plugin folder for your reference.

.. code-block:: bash

    # Windows
    %APPDATA%\QGIS\QGIS3\profiles\default\python\plugins

    # Mac
    ~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins
    
    # Linux
    ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins


Activate the Geo-SAM Plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After restarting QGIS, go to the menu ``Plugins`` > ``Manage and Install Plugins``, and under ``Installed``, you may find the ``Geo SAM`` plugin; check it to activate the plugin.


.. image:: img/Active_geo_sam.png
    :alt: Plugin menu
    :width: 90%
    :align: center


After activating the Geo-SAM plugin, you may find the Geo SAM tools under the ``Plugins`` menu,


.. image:: img/Plugin_menu_geo_sam.png
    :alt: Plugin menu
    :width: 60%
    :align: center

You may also find a new toolbar, including three icons.

.. image:: img/Toolbar_geo_sam.png
    :alt: Plugin toolbar
    :width: 33%
    :align: center

