.. _geo-sam-encoder_python_package:

GeoSAM-Image-Encoder (Python package)
======================================

.. image:: https://img.shields.io/pypi/v/GeoSAM-Image-Encoder
    :target: https://pypi.org/project/GeoSAM-Image-Encoder/
    :alt: PyPI Version

.. image:: https://static.pepy.tech/badge/GeoSAM-Image-Encoder
    :target: https://pepy.tech/project/GeoSAM-Image-Encoder
    :alt: Downloads

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/coolzhao/Geo-SAM/blob/main/GeoSAM-Image-Encoder/examples/geosam-image-encoder.ipynb
    :alt: Open In Colab

This package is part of the `Geo-SAM <https://github.com/coolzhao/Geo-SAM>`_ project and is a standalone Python package that does not depend on QGIS. This package allows you to **encode remote sensing images into features that can be recognized by Geo-SAM using a remote server**, such as ``Colab``, ``AWS``, ``Azure`` or your own ``HPC``.


Installation
------------
.. note::

    Installing ``GeoSAM-Image-Encoder`` may directly install the CPU version of ``PyTorch``. Therefore, it is recommended to install the appropriate version of ``PyTorch`` before installing ``GeoSAM-Image-Encoder`` in your machine. You can install the corresponding version based on the official ``PyTorch`` website: https://pytorch.org/get-started/locally/#start-locally

After installing PyTorch, you can install ``GeoSAM-Image-Encoder`` via pip.

.. code-block:: bash

    pip install GeoSAM-Image-Encoder
    # or
    pip install git+https://github.com/Fanchengyan/GeoSAM-Image-Encoder.git


Usage
-----

There are **two ways** to use GeoSAM-Image-Encoder. You can call it in Python or Terminal. We recommend using Python interface directly which will have greater flexibility.

Using Python
~~~~~~~~~~~~

After install GeoSAM-Image-Encoder, you can import it using ``geosam``

.. code-block:: python

    import geosam
    from geosam import ImageEncoder

check if gpu available

.. code-block:: python

    geosam.gpu_available()

Run by specify parameters directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to specify the parameters directly, you can run it like this:

.. code-block:: python

    checkpoint_path = '/content/sam_vit_l_0b3195.pth'
    image_path = '/content/beiluhe_google_img_201211_clip.tif'
    feature_dir = './'

    ## init ImageEncoder
    img_encoder = ImageEncoder(checkpoint_path)
    ## encode image
    img_encoder.encode_image(image_path,feature_dir)


Run by parameters from setting.json file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to using ``settings.json`` file which exported from Geo-SAM plugin to provide parameters, you can run it like this:

.. code-block:: python

    setting_file = "/content/setting.json"
    feature_dir = './'

    ### parse settings from the setting,json file
    settings = geosam.parse_settings_file(setting_file)

    ### setting file not contains feature_dir, you need add it
    settings.update({"feature_dir":feature_dir})

    ### split settings into init_settings, encode_settings
    init_settings, encode_settings = geosam.split_settings(settings)

    print(f"settings: {settings}")
    print(f"init_settings: {init_settings}")
    print(f"encode_settings: {encode_settings}")

Then, you can run image encoding by parameters from ``setting.json`` file

.. code-block:: python

    img_encoder = ImageEncoder(**init_settings)
    img_encoder.encode_image(**encode_settings)

Using Terminal
~~~~~~~~~~~~~~

check the folder of geosam

.. code-block:: bash

    print(geosam.folder)

add this folder into environment of your machine. Then run in terminal:

.. code-block:: bash

    image_encoder.py -i /content/beiluhe_google_img_201211_clip.tif -c /content/sam_vit_l_0b3195.pth -f ./

You can overwrite the settings from file by specify the parameter values. For Example

.. code-block:: bash

    image_encoder.py -s /content/setting.json  -f ./ --stride 256 --value_range "10,255"

check all available parameters:

.. code-block:: bash

    image_encoder.py -h

.. code-block:: none
    
    This script is for encoding image to SAM features.

    =====
    Usage
    =====
    using settings.json:

        image_encoder.py -s <settings.json> -f <feature_dir>
    
    
    or directly using parameters:
    
        image_encoder.py -i <image_path> -c <checkpoint_path> -f <feature_dir>
        
    All Parameters:
    -------------------
    -s, --settings:         Path to the settings json file.
    -i, --image_path:       Path to the input image.
    -c, --checkpoint_path:  Path to the SAM checkpoint.
    -f, --feature_dir:      Path to the output feature directory.
    --model_type: one of ["vit_h", "vit_l", "vit_b"] or [0, 1, 2] or None, optional
        The type of the SAM model. If None, the model type will be 
        inferred from the checkpoint path. Default: None. 
    --bands: list of int, optional .
        The bands to be used for encoding. Should not be more than three bands.
        If None, the first three bands (if available) will be used. Default: None.
    --stride: int, optional
        The stride of the sliding window. Default: 512.
    --extent: str, optional
        The extent of the image to be encoded. Should be in the format of
        "minx, miny, maxx, maxy, [crs]". If None, the extent of the input
        image will be used. Default: None.
    --value_range: tuple of float, optional
        The value range of the input image. If None, the value range will be
        automatically calculated from the input image. Default: None.
    --resolution: float, optional
        The resolution of the output feature in the unit of raster crs.
        If None, the resolution of the input image will be used. Default: None.
    --batch_size: int, optional
        The batch size for encoding. Default: 1.
    --gpu_id: int, optional
        The device id of the GPU to be used. Default: 0.

Colob Example
-------------


You can click on the link below to experience GeoSAM-Image-Encoder in ``Colab``: 

`<https://colab.research.google.com/github/coolzhao/Geo-SAM/blob/main/GeoSAM-Image-Encoder/examples/geosam-image-encoder.ipynb>`_
