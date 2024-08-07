Usage of Geo-SAM Tools
======================

The workflow of using Geo-SAM tools is as follows:

1. Encode the image into feature files using the Geo-SAM encoder. The encoding process can be done either by:

   * QGIS plugin: :ref:`encoding`
   * A standalone Python package: :ref:`encoder_package`.

2. Open the :ref:`segmentation` , Then:

   1. Select the encoded features generated in the previous step. 
   2. Load the image to the QGIS canvas. 
   3. Select the output shapefile directory for SAM results. 

Then, you can use the segmentation tool to segment the image.

.. figure:: ../img/Geo_SAM_workflow.png
   :width: 100%
   :align: center
   :alt: Geo-SAM Workflow

----------------

.. toctree:: 
   :maxdepth: 1

   segmentation
   encoding
   encoder_package
   copilot

