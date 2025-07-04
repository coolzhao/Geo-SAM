.. Geo-SAM documentation master file, created by
   sphinx-quickstart on Sun Aug 13 16:07:25 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================
Welcome to Geo-SAM's documentation!
===================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8191039.svg
   :target: https://zenodo.org/record/8191039
   :alt: DOI

.. image:: https://readthedocs.org/projects/geo-sam/badge/?version=latest
    :target: https://geo-sam.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

By `Zhuoyi Zhao (Joey) <https://github.com/coolzhao>`_ and `Chengyan Fan (Fancy) <https://github.com/Fanchengyan>`_ from `Cryosphere Lab <https://cryocuhk.github.io/>`_, ESSC, CUHK.

Introduction
------------

Geo-SAM is a QGIS plugin that aims to help people segment, delineate or label landforms efficiently when using large-size geospatial raster images. `Segment Anything Model <https://segment-anything.com/>`_ (SAM) is a foundation AI model with the superpower, but the model size is huge, and using it to process images can take a long time, even with a modern GPU. Our tool uses the strategies of encoding image features in advance and trimming the SAM model. **The interactive segmentation process can be run in real-time (at** ``millisecond`` **speeds) on a laptop by only using a CPU**, making it a convenient and efficient tool for dealing with remote sensing images.

The Geo-SAM plugin includes two separate parts: the ``Image Encoding Part``, and the ``Interactive Segmentation Part``. The image encoding part is designed to generate and save the image features using the SAM image encoder, and the encoding process only needs to run once per image. The segmentation part is for interactively segmenting landforms, and it can only be used to segment preprocessed images (whose features have been generated in advance using the encoding tool, as the included demo image).


.. figure:: img/Geo_SAM.png
   :width: 100%
   :align: center
   :alt: Geo-SAM

   Comparison of the workflow between Geo-SAM and the original SAM. The original SAM model encodes prompts and image simultaneously, while the Geo-SAM model encodes image into feature files at once and queries prompts in real-time (at ``millisecond`` speeds) by loading those saved features. 

Reasons for choosing Geo-SAM
----------------------------

* Based on QGIS for a user-friendly GUI and cross-platform compatibility without programming skills needed.
* It provides segmentation results ``instantly after giving prompts``, and can even display results ``in real-time following the mouse cursor`` (Preview mode, currently only available in the dev version, will be added to the stable version after being rigorously tested). Users can have a smooth, interactive experience.This can greatly improve the efficiency and user experience of segmentation.


.. note::
   
   - SAM is designed to **segment one object once with a series of prompts**, so you should save the current results before getting to the next one when using the Geo-SAM tool.
   - SAM natively supports only three-band images, but we have adapted Geo-SAM to support one or two-band images so that you can try grayscale images, spectral index images (like NDVI, NDWI), or even SAR images.
   - The Geo-SAM plugin is currently in active development. We will continue making improvements and welcome your feedback. If you have any questions or suggestions, please feel free to open an issue or discussion on our GitHub repository at `GitHub Issues <https://github.com/coolzhao/Geo-SAM/issues>`_ or `GitHub Discussions <https://github.com/coolzhao/Geo-SAM/discussions>`_.


Video Tutorials
---------------

The following video is a good tutorial that shows how to install and use Geo-SAM.

.. youtube:: GSKmK7qERUw
   :align: center



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   Usage/index
   blog/index
   more
   


