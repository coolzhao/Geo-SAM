
Crash on Latest QGIS in Linux
=============================

- Author : Chengyan Fan (Fancy)
- Date Created : 2024-05-02
- Last Updated : 2024-12-16


The latest version of QGIS 3.34/3.36 will crash when trying to load features using the Geo-SAM plugin. This issue may be due to library mismatch of ``rtree`` in **torchgeo** and **QGIS**. More information can be found in the following links: `<https://github.com/qgis/QGIS/issues/57320>`_ and `<https://github.com/coolzhao/Geo-SAM/issues/43>`_.

Reason
------

Currently, the ``rtree`` in QGIS is still using the old version (<1.0). However, **torchgeo** requires the version of ``rtree`` should be greater than 1.0 for Python 3.10. This leads to **torchgeo** will upgrade the ``rtree`` to the latest version. When running the Geo-SAM plugin in QGIS, the ``rtree`` version mismatch will cause the crash.

At the moment, it appears that this issue is only causing crashes on Linux platforms.


Solution
--------

To resolve this issue, first uninstall the ``rtree`` package using ``pip``, then reinstall it with the appropriate version. The easiest way to install the correct version of ``rtree`` is by using your system's package manager.

Following solution only illustrates how to solve the issue using the system package manager in Ubuntu/Debian-based systems.

.. note::

  If you're using other tools for managing Python packages, such as ``flatpak`` or ``conda``, try installing a version of ``rtree`` earlier than 1.0, such as ``0.9.7`` or ``0.9.6``. You can find a list of available ``rtree`` versions on the `PyPI <https://pypi.org/project/Rtree/#history>`_. 

1. Remove the ``rtree`` package using pip in QGIS:

.. code-block:: bash

    cd /usr/bin

    ./pip3 uninstall rtree

2. Install the ``rtree`` package using system package manager:

.. code-block:: bash

    sudo apt-get install python3-rtree

3. Download the latest version of the Geo-SAM plugin from ``main`` branch of `Geo-SAM <https://github.com/coolzhao/Geo-SAM>`_ and reinstall it.

After following the above steps, you should be able to run the Geo-SAM plugin in QGIS without any crashes.

