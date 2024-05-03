
Crash on QGIS 3.34/3.36
=======================

- Author : Chengyan Fan (Fancy)
- Date : 2024-05-02


The latest version of QGIS 3.34/3.36 will crash when trying to load features using the Geo-SAM plugin. This issue may be due to library mismatch of ``rtree`` in **torchgeo** and **QGIS**. More information can be found in the following links: `<https://github.com/qgis/QGIS/issues/57320>`_ and `<https://github.com/coolzhao/Geo-SAM/issues/43>`_.

Reason
------

Currently, the ``rtree`` in QGIS is still using the old version (<1.0). However, **torchgeo** requires the version of ``rtree`` should be greater than 1.0 for Python 3.10. This leads to **torchgeo** will upgrade the ``rtree`` to the latest version when installing. When running the Geo-SAM plugin in QGIS, the ``rtree`` version mismatch will cause the crash.


Solution
--------

To solve this issue, you need to remove the ``rtree`` package using pip and then reinstall it using system package manager. 

.. note::

    This solution only illustrates how to solve the issue in Ubuntu. If you are using other operating systems, please refer to the corresponding package manager to install the ``rtree`` package.

1. Remove the ``rtree`` package using pip in QGIS:

.. code-block:: bash

    cd /usr/bin

    ./pip3 uninstall rtree

2. Install the ``rtree`` package using system package manager:

.. code-block:: bash

    sudo apt-get install python3-rtree

3. Download the latest version of the Geo-SAM plugin from ``main`` branch of `Geo-SAM <https://github.com/coolzhao/Geo-SAM>`_ and reinstall it.

After following the above steps, you should be able to run the Geo-SAM plugin in QGIS without any crashes.

