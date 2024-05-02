
Crash on QGIS 3.34/3.36
=======================

- Author : Chengyan Fan (Fancy)
- Date : 2024-05-02



The latest version of QGIS 3.34/3.36 will crash when trying to load features using the Geo-SAM plugin. This issue may be due to library mismatch of ``rtree`` in **torchgeo** and **QGIS**. More information can be found in the following links: `<https://github.com/qgis/QGIS/issues/57320>`_ and `<https://github.com/coolzhao/Geo-SAM/issues/43>`_.

.. important::

    Currently, you are recommended to use QGIS 3.32 or earlier versions to avoid this issue.