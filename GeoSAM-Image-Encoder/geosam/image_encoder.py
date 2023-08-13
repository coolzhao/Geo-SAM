#!/usr/bin/env python
"""
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
"""

import sys
import getopt
import json
from pathlib import Path
from typing import List, Tuple, Union
from tqdm import tqdm
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
import torch
from geosam.torchgeo_sam import SamTestGridGeoSampler, SamTestRasterDataset
from torchgeo.samplers import Units
from torchgeo.datasets import BoundingBox, stack_samples
from torch.utils.data import DataLoader
import rasterio
from rasterio import warp
from rasterio.crs import CRS
import numpy as np
import pandas as pd
from torch import Tensor
import hashlib


SAM_Model_Types = ["vit_h",
                   "vit_l",
                   "vit_b"]

Parameter_Mapping = {
    "INPUT": "image_path",
    "CKPT": "checkpoint_path",
    "MODEL_TYPE": "model_type",
    "BANDS": "bands",
    "STRIDE": "stride",
    "EXTENT": "extent",
    "RANGE": "value_range",
    "RESOLUTION": "resolution",
    "BATCH_SIZE": "batch_size",
    "CUDA_ID": "gpu_id"
}

Init_Settings = [
    'checkpoint_path',
    'model_type',
    'batch_size',
    'gpu_id'
]
Encode_Settings = [
    'image_path',
    'feature_dir',
    'bands',
    'stride',
    'extent',
    'value_range',
    'resolution'
]


class ImageEncoder:
    '''Encode image to SAM features.'''

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        model_type: str = None,
        batch_size: int = 1,
        gpu: bool = True,
        gpu_id: int = 0,
    ):
        '''Initialize the ImageEncoder.

        Parameters:
        ----------
        checkpoint_path: str or Path
            Path to the SAM checkpoint.
        model_type: one of ["vit_h", "vit_l", "vit_b"] or [0, 1, 2] or None, optional
            The type of the SAM model. If None, the model type will be 
            inferred from the checkpoint path. Default: None. 
        batch_size: int, optional
            The batch size for encoding. Default: 1.
        gpu: bool, optional
            Whether to use GPU for encoding if available. Default: True.
        gpu_id: int, optional
            The device id of the GPU to be used. Default: 0.
        '''
        self.checkpoint_path = Path(checkpoint_path)

        self.batch_size = batch_size
        self.gpu = gpu
        self.gpu_id = gpu_id
        self.model_type = check_model_type(model_type, self.checkpoint_path)
        self.device = detect_device(self.gpu, self.gpu_id)
        self.sam_model = self.initialize_sam()

    def encode_image(
        self,
        image_path: Union[str, Path],
        feature_dir: Union[str, Path],
        bands: List[int] = None,
        stride: int = 512,
        extent: str = None,
        value_range: Tuple[float, float] = None,
        resolution: float = None,
    ):
        '''Encode image to SAM features.

        Parameters:
        ----------
        image_path: str or Path
            Path to the input image.
        feature_dir: str or Path
            Path to the output feature directory.
        bands: list of int, optional
            The bands to be used for encoding. Should not be more than three bands.
            If None, the first three bands (if available) will be used. Default: None.
        stride: int, optional
            The stride of the sliding window. Default: 512.
        extent: str, optional
            The extent of the image to be encoded. Should be in the format of
            "minx, miny, maxx, maxy, [crs]". If None, the extent of the input
            image will be used. Default: None.
        value_range: tuple of float, optional
            The value range of the input image. If None, the value range will be
            automatically calculated from the input image. Default: None.
        resolution: float, optional
            The resolution of the output feature in the unit of raster crs.
            If None, the resolution of the input image will be used. Default: None.
        '''
        image_path = Path(image_path)
        feature_dir = Path(feature_dir)

        print('\n----------------------------------------------')
        print('     Start encoding image to SAM features')
        print('----------------------------------------------\n')

        # load image and check extent
        with rasterio.open(image_path) as src:
            arr = src.read()
            meta = src.meta.copy()

            if extent is None:
                extent = [i for i in src.bounds]
                extent_crs = src.crs
            else:
                extent_crs = extent.split(' ')[-1].strip()[1:-1].strip()
                extent = [float(i.strip(',').strip())
                          for i in extent.split(' ')[:-1]]
                if extent_crs != '':
                    extent_crs = CRS.from_user_input(extent_crs)
                    extent = warp.transform_bounds(
                        extent_crs,
                        src.crs,
                        *extent,
                    )

        # check bands
        if bands is None:
            max_band = min(3, meta['count'])
            bands = list(range(1, max_band+1))

        if len(bands) > 3:
            raise ValueError(
                "SAM only supports no more than three bands,"
                f" but {len(bands)} bands are given."
            )
        if max(bands) > meta['count']:
            raise ValueError(
                f"The band number of the input image is {meta['count']}. "
                f"But the band index {max(bands)} is given."
            )
        # ensure only three bands are used, less than three bands will be broadcasted to three bands
        bands = [str(i) for i in (bands * 3)[0:3]]

        # check resolution
        if resolution is None:
            resolution = meta['transform'][0]

        # get pixel number in the extent
        img_width_in_extent = round(
            (extent[2] - extent[0])/resolution)
        img_height_in_extent = round(
            (extent[3] - extent[1])/resolution)

        # Print input parameters
        print('Input Parameters:')
        print('----------------------------------------------')
        # check value range
        if value_range is not None:
            if isinstance(value_range, str):
                try:
                    value_range = eval(value_range)
                except:
                    raise ValueError(
                        f"Could not evaluate the value range. {value_range}"
                        "Please check the format of the value range."
                    )
            if value_range[0] >= value_range[1]:
                raise ValueError(
                    "Data value range is wrongly set or the image is with constant values.")
            print(' Input data value range to be rescaled: '
                  f'{value_range} (set by user)')
        else:
            value_range = [np.nanmin(arr),
                           np.nanmax(arr)]
            print(
                f' Input data value range to be rescaled: {value_range} '
                '(automatically set based on min-max value of input image inside the processing extent.)'
            )

        print(f' Image path: {image_path}')
        print(f' Bands selected: {bands}')
        print(f' Target resolution: {resolution}')
        print(f' Processing extent: {extent}')
        print(f' Processing image size: (width {img_width_in_extent}, '
              f'height {img_height_in_extent})')
        print('----------------------------------------------\n')

        img_dir = str(image_path.parent)
        img_name = image_path.name

        SamTestRasterDataset.filename_glob = img_name
        SamTestRasterDataset.all_bands = [
            str(i) for i in range(1, meta['count']+1)
        ]

        sam_ds = SamTestRasterDataset(
            root=img_dir,
            crs=None,
            res=resolution,
            bands=bands,
            cache=False
        )

        print(
            f'\nRasterDataset info '
            '\n----------------------------------------------'
            f'\n filename_glob: {sam_ds.filename_glob}, '
            f'\n all bands: {sam_ds.all_bands}, '
            f'\n input bands: {sam_ds.bands}, '
            f'\n resolution: {sam_ds.res}, '
            f'\n bounds: {sam_ds.index.bounds}, '
            f'\n num: {len(sam_ds.index)}'
            '\n----------------------------------------------\n'
        )

        extent_bbox = BoundingBox(
            minx=extent[0],
            maxx=extent[2],
            miny=extent[1],
            maxy=extent[3],
            mint=sam_ds.index.bounds[4],
            maxt=sam_ds.index.bounds[5]
        )

        ds_sampler = SamTestGridGeoSampler(
            sam_ds,
            size=self.sam_model.image_encoder.img_size,
            stride=stride,
            roi=extent_bbox,
            units=Units.PIXELS  # Units.CRS or Units.PIXELS
        )

        if len(ds_sampler) == 0:
            raise ValueError(
                f'!!!No available patch sample inside the chosen extent!!!')

        ds_dataloader = DataLoader(
            sam_ds,
            batch_size=self.batch_size,
            sampler=ds_sampler,
            collate_fn=stack_samples
        )

        print('----------------------------------------------')
        print(f' SAM model initialized. \n '
              f' SAM model type:  {self.model_type}')

        if not self.gpu or not gpu_available():
            print(' !!!No GPU available, using CPU instead!!!')
            self.batch_size = 1
        print(f' Device type: {self.device}')

        print(f' Patch size: {ds_sampler.patch_size} \n'
              f' Batch size: {self.batch_size}')
        print(f' Patch sample num: {len(ds_sampler)}')
        print(f' Total batch num: {len(ds_dataloader)}')
        print('----------------------------------------------\n')

        for patch_idx, batch in tqdm(
                enumerate(ds_dataloader),
                desc='Encoding image',
                unit='batch',
                total=len(ds_dataloader)
        ):

            batch_input = rescale_img(
                batch_input=batch['image'], value_range=value_range)

            features = get_sam_feature(batch_input, self.sam_model)

            # print(f'\nBatch no. {patch_idx+1} loaded')
            # print(f'img_shape: ' + str(batch['img_shape'][0]))
            # print('patch_size: ' + str(batch['image'].shape))
            # print(f'feature_shape: {features.shape}')

            # TODO: show gpu usage info

            save_sam_feature(
                sam_ds,
                feature_dir,
                batch,
                features,
                extent,
                patch_idx,
                self.model_type
            )

        print(f'"Output feature path": {feature_dir}')

    def initialize_sam(self) -> Sam:
        print("Initializing SAM model...\n")
        sam_model = sam_model_registry[self.model_type](
            checkpoint=self.checkpoint_path)
        sam_model.to(device=self.device)
        return sam_model


def check_model_type(model_type, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    model_type_errors_str = (
        f"model_type should be one of {SAM_Model_Types} or {range(len(SAM_Model_Types))}, "
        f"but {model_type} is given."
    )
    if isinstance(model_type, int):
        if model_type not in range(len(SAM_Model_Types)):
            raise ValueError(model_type_errors_str)
        else:
            model_type = SAM_Model_Types[model_type]
    elif isinstance(model_type, str):
        if model_type not in SAM_Model_Types:
            raise ValueError(model_type_errors_str)
    elif model_type is None:
        # infer model type from checkpoint path
        flag = False
        for model_type in SAM_Model_Types:
            if model_type in checkpoint_path.name:
                flag = True
                break
        if not flag:
            raise ValueError(
                "Could not infer the model type from the checkpoint path. "
                "Please specify the model type manually."
            )
    else:
        raise ValueError(model_type_errors_str)
    return model_type


def gpu_available() -> bool:
    '''Check if GPU is available.'''
    return cuda_available() or mps_available()


def cuda_available() -> bool:
    '''Check if CUDA (NVIDIA or ROCm) is available.'''
    return torch.cuda.is_available()


def mps_available() -> bool:
    '''Check if MPS (Mac) is available.'''
    return torch.backends.mps.is_available()


def detect_device(gpu: bool, gpu_id: int) -> str:
    '''Detect device for pytorch.'''
    if not gpu:
        return 'cpu'
    elif cuda_available():
        return f'cuda:{gpu_id}'
    elif mps_available():
        return f'mps:{gpu_id}'
    else:
        print('No GPU available, using CPU instead.')
        return 'cpu'


def rescale_img(batch_input: Tensor, value_range: List[float]) -> Tensor:
    'rescale input image to [0,255]'
    range_min = value_range[0]
    range_max = value_range[1]
    batch_output = (batch_input - range_min)*255/(range_max - range_min)
    return batch_output


@torch.no_grad()
def get_sam_feature(batch_input: Tensor, sam_model) -> bool:
    # TODO: if the input image are all zero(batch_input.any()), directly return features with all zero and give a message
    # should know the shape of the feature in advance
    batch_input = batch_input.to(device=sam_model.device)
    batch_input = ((batch_input - sam_model.pixel_mean) /
                   sam_model.pixel_std)
    features = sam_model.image_encoder(batch_input)
    return features.cpu().numpy()


def save_sam_feature(
    raster_ds: SamTestRasterDataset,
    export_dir: Path,
    data_batch: Tensor,
    feature: np.ndarray,
    extent: List[float],
    patch_idx: int,
    model_type: str = "vit_h"
) -> int:
    # iterate over batch_size dimension
    for idx in range(feature.shape[-4]):
        band_num = feature.shape[-3]
        height = feature.shape[-2]
        width = feature.shape[-1]
        bbox = data_batch['bbox'][idx]
        rio_transform = rasterio.transform.from_bounds(
            bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height)  # west, south, east, north, width, height
        filepath = Path(data_batch['path'][idx])
        bbox_list = [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy]
        bbox_str = '_'.join(map("{:.6f}".format, bbox_list))
        extent_str = '_'.join(
            map("{:.6f}".format, extent)) + f"_res_{raster_ds.res:.6f}"
        #  Unicode-objects must be encoded before hashing with hashlib and
        #  because strings in Python 3 are Unicode by default (unlike Python 2),
        #  you'll need to encode the string using the .encode method.
        bbox_hash = hashlib.sha256(bbox_str.encode("utf-8")).hexdigest()
        extent_hash = hashlib.sha256(
            extent_str.encode("utf-8")).hexdigest()

        bands_str = '_'.join([str(band) for band in raster_ds.band_indexes])
        export_dir_sub = (export_dir / filepath.stem /
                          f"sam_feat_{model_type}_bands_{bands_str}_{extent_hash[0:16]}")
        export_dir_sub.mkdir(parents=True, exist_ok=True)
        feature_tiff = (export_dir_sub /
                        f"sam_feat_{model_type}_{bbox_hash}.tif")
        feature_csv = (export_dir_sub / f"{export_dir_sub.name}.csv")
        with rasterio.open(
                feature_tiff,
                mode="w",
                driver="GTiff",
                height=height, width=width,
                count=band_num,
                dtype='float32',
                crs=data_batch['crs'][idx],
                transform=rio_transform
        ) as feature_dataset:
            # index start from 1, feature[idx, :, :, :] = feature[idx, ...], later is faster
            feature_dataset.write(feature[idx, ...], range(1, band_num+1))
            # pr_mask_dataset.set_band_description(1, '')
            tags = {
                "img_shape": data_batch["img_shape"][idx],
                "input_shape": data_batch["input_shape"][idx],
                "model_type": model_type,
            }
            feature_dataset.update_tags(**tags)
            feature_crs = feature_dataset.crs

        index_df = pd.DataFrame(columns=['minx', 'maxx', 'miny', 'maxy', 'mint', 'maxt',
                                         'filepath',
                                         'crs', 'res'],
                                index=[patch_idx])
        index_df['filepath'] = [feature_tiff.name]
        index_df['minx'] = [bbox.minx]
        index_df['maxx'] = [bbox.maxx]
        index_df['miny'] = [bbox.miny]
        index_df['maxy'] = [bbox.maxy]
        index_df['mint'] = [bbox.mint]
        index_df['maxt'] = [bbox.maxt]
        index_df['crs'] = [str(feature_crs)]
        index_df['res'] = [raster_ds.res]
        index_df['model_type'] = [model_type]
        # append data frame to CSV file, index=False
        index_df.to_csv(feature_csv, mode='a',
                        header=not feature_csv.exists(), index=True)


class Usage(Exception):
    """Usage context manager"""

    def __init__(self, msg):
        self.msg = msg


def encode_image_from_cmd(argv=None):
    if argv == None:
        argv = sys.argv

    settings_path = None
    image_path = None
    checkpoint_path = None
    feature_dir = None
    model_type = None
    bands = None
    stride = 512
    extent = None
    value_range = None
    resolution = None
    batch_size = 1
    gpu_id = 0

    try:
        opts, args = getopt.getopt(argv[1:], "hs:i:c:f:", [
            "help",
            "settings=",
            "image_path=",
            "checkpoint_path=",
            "feature_dir=",
            "model_type=",
            "bands=",
            "stride=",
            "extent=",
            "value_range=",
            "resolution=",
            "batch_size=",
            "gpu_id="
        ])
    except getopt.error as msg:
        raise Usage(msg)

    for o, a in opts:
        if o in ['-h', '--help']:
            print(__doc__)
            return 0
        elif o in ['-s', '--settings']:
            settings_path = Path(a).absolute()
        elif o in ['-i', '--image_path']:
            image_path = Path(a).absolute()
        elif o in ['-c', '--checkpoint_path']:
            checkpoint_path = Path(a).absolute()
        elif o in ['-f', '--feature_dir']:
            feature_dir = Path(a).absolute()
        elif o == '--model_type':
            model_type = a
        elif o == '--bands':
            bands = eval(a)
        elif o == '--stride':
            stride = int(a)
        elif o == '--extent':
            extent = a
        elif o == '--value_range':
            value_range = a
        elif o == '--resolution':
            resolution = float(a)
        elif o == '--batch_size':
            batch_size = int(a)
        elif o == '--gpu_id':
            gpu_id = int(a)

    settings = {}
    if feature_dir is not None:
        settings['feature_dir'] = feature_dir
    else:
        raise ValueError('feature_dir is not specified.')

    if settings_path is not None:
        settings.update(parse_settings_file(settings_path))

    if image_path is not None:
        settings.update({'image_path': image_path})
    if checkpoint_path is not None:
        settings.update({'checkpoint_path': checkpoint_path})
    if model_type is not None:
        if len(model_type) == 1:
            model_type = int(model_type)
        settings.update({'model_type': model_type})
    if bands is not None:
        settings.update({'bands': bands})
    if stride is not None:
        settings.update({'stride': stride})
    if extent is not None:
        settings.update({'extent': extent})
    if value_range is not None:
        settings.update({'value_range': value_range})
    if resolution is not None:
        settings.update({'resolution': resolution})
    if batch_size is not None:
        settings.update({'batch_size': batch_size})
    if gpu_id is not None:
        settings.update({'gpu_id': gpu_id})

    print(f"\nsettings:\n {settings}\n")
    init_settings, encode_settings = split_settings(settings)
    img_encoder = ImageEncoder(**init_settings)
    img_encoder.encode_image(**encode_settings)


def parse_settings_file(settings_path):
    with open(settings_path) as f:
        settings = json.load(f)
        settings = settings['inputs']
    settings = {Parameter_Mapping[k]: v for k,
                v in settings.items() if k != 'CRS'}
    return settings


def split_settings(settings):
    init_settings = {k: v for k, v in settings.items()
                     if k in Init_Settings}
    encode_settings = {k: v for k, v in settings.items()
                       if k in Encode_Settings}
    return init_settings, encode_settings


if __name__ == "__main__":
    sys.exit(encode_image_from_cmd())
