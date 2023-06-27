import os
import time
from typing import Dict, Any
from pathlib import Path
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import QAction, QDockWidget
from qgis.gui import QgsDockWidget, QgsFileWidget
from qgis.utils import iface
from qgis.core import (QgsProcessing,
                       QgsCoordinateTransform,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterBand,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterBoolean,
                       QgsProcessingParameterFile,
                       QgsProcessingParameterString,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterExtent,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink)
from qgis import processing
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam
import torch
from .torchgeo_sam import SamTestGridGeoSampler, SamTestRasterDataset
from torchgeo.samplers import Units
from torchgeo.datasets import BoundingBox, stack_samples
from torch.utils.data import DataLoader
import rasterio
import numpy as np
import pandas as pd
from torch import Tensor
import hashlib
from ..ui.icons import QIcon_GeoSAMEncoder


class SamProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This is an example algorithm that takes a vector layer and
    creates a new identical one.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is not need for additional work.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT = 'INPUT'
    CKPT = 'CKPT'
    MODEL_TYPE = 'MODEL_TYPE'
    BANDS = 'BANDS'
    STRIDE = 'STRIDE'
    EXTENT = 'EXTENT'
    LOAD = 'LOAD'
    OUTPUT = 'OUTPUT'

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # We add the input vector features source. It can have any kind of
        # geometry.
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                self.tr('Input Layer')
            )
        )
        # add ParameterRasterCalculatorExpression to normalize raster values to 0-255

        self.addParameter(
            QgsProcessingParameterBand(
                name=self.BANDS,
                description=self.tr('Bands'),
                defaultValue=[1, 2, 3],
                parentLayerParameterName=self.INPUT,
                allowMultiple=True
            )
        )

        self.addParameter(
            QgsProcessingParameterExtent(
                self.EXTENT,
                self.tr(
                    'Processing extent'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.STRIDE,
                self.tr('Stride'),
                defaultValue=512,
                minValue=256,
                maxValue=1024
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                name=self.CKPT,
                description=self.tr('Checkpoint Path'),
                extension='pth',
            )
        )

        self.model_type_options = ['vit_h', 'vit_l', 'vit_b']
        self.addParameter(
            QgsProcessingParameterEnum(
                name=self.MODEL_TYPE,
                description=self.tr('Model Type'),
                options=self.model_type_options,
                defaultValue=0,  # 'vit_h'
            )
        )
        # We add a feature sink in which to store our processed features (this
        # usually takes the form of a newly created vector layer when the
        # algorithm is run in QGIS).
        # self.addParameter(
        #     QgsProcessingParameterFolderDestination(
        #         self.OUTPUT,
        #         self.tr('Output Folder')
        #     )
        # )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT,
                self.tr("Output Folder"),
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOAD,
                self.tr("Load output features after processing"),
                defaultValue=True
            )
        )

        # self.addOutput()

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        # Retrieve the feature source and sink. The 'dest_id' variable is used
        # to uniquely identify the feature sink, and must be included in the
        # dictionary returned by the processAlgorithm function.
        rlayer = self.parameterAsRasterLayer(
            parameters, self.INPUT, context)
        # If source was not found, throw an exception to indicate that the algorithm
        # encountered a fatal error. The exception text can be any string, but in this
        # case we use the pre-built invalidSourceError method to return a standard
        # helper text for when a source cannot be evaluated
        if rlayer is None:
            raise QgsProcessingException(
                self.invalidRasterError(parameters, self.INPUT))
        selected_bands = self.parameterAsInts(
            parameters, self.BANDS, context)
        ckpt_path = self.parameterAsFile(
            parameters, self.CKPT, context)
        model_type_idx = self.parameterAsEnum(
            parameters, self.MODEL_TYPE, context)
        stride = self.parameterAsInt(
            parameters, self.STRIDE, context)

        extent = self.parameterAsExtent(parameters, self.EXTENT, context)
        load_feature = self.parameterAsBoolean(parameters, self.LOAD, context)
        # if bbox.isNull() and not rlayer:
        #     raise QgsProcessingException(
        #         self.tr("No reference layer selected nor extent box provided"))

        if not extent.isNull():
            extentCrs = self.parameterAsExtentCrs(
                parameters, self.EXTENT, context)
            if extentCrs != rlayer.crs():
                transform = QgsCoordinateTransform(
                    extentCrs, rlayer.crs(), context.transformContext())
                extent = transform.transformBoundingBox(extent)

        if extent.isNull() and rlayer:
            extent = rlayer.extent()  # QgsProcessingUtils.combineLayerExtents(layers, crs, context)

        # output_dir = self.parameterAsFileOutput(
        #     parameters, self.OUTPUT, context)

        output_dir = self.parameterAsString(
            parameters, self.OUTPUT, context)

        # Send some information to the user
        # feedback.pushInfo('CRS is {}'.format(rlayer.crs().authid()))
        # feedback.pushInfo('Band number is {}'.format(rlayer.bandCount()))
        # feedback.pushInfo('Band name is {}'.format(rlayer.bandName(1)))
        feedback.pushInfo('Layer path: {}'.format(
            rlayer.dataProvider().dataSourceUri()))
        feedback.pushInfo('Layer name: {}'.format(rlayer.name()))
        feedback.pushInfo(f'Bands selected: {selected_bands}')
        # feedback.pushInfo('Layer display band name is {}'.format(
        #     rlayer.dataProvider().displayBandName(1)))
        feedback.pushInfo(
            f'Layer extent: minx:{extent.xMinimum():.2f}, maxx:{extent.xMaximum():.2f}, miny:{extent.yMinimum():.2f}, maxy:{extent.yMaximum():.2f}')

        model_type = self.model_type_options[model_type_idx]
        if model_type not in os.path.basename(ckpt_path):
            raise QgsProcessingException(
                self.tr("Model type does not match the checkpoint"))

        sam_model = self.initialize_sam(
            model_type=model_type, sam_ckpt_path=ckpt_path)

        feedback.pushInfo(
            f'SAM Image Size: {sam_model.image_encoder.img_size}')

        rlayer_path = rlayer.dataProvider().dataSourceUri()
        rlayer_dir = os.path.dirname(rlayer_path)
        rlayer_name = os.path.basename(rlayer_path)

        SamTestRasterDataset.filename_glob = rlayer_name
        SamTestRasterDataset.all_bands = [
            rlayer.bandName(i_band) for i_band in range(1, rlayer.bandCount()+1)
        ]
        # currently only support rgb bands
        input_bands = [rlayer.bandName(i_band) for i_band in selected_bands]

        rlayer_ds = SamTestRasterDataset(
            root=rlayer_dir, crs=rlayer.crs().toWkt(), bands=input_bands, cache=False)
        feedback.pushInfo(
            f'RasterDS info, input bands: {rlayer_ds.bands}, \n all bands: {rlayer_ds.all_bands}, \
            \n raster_ds crs: {rlayer_ds.crs}, \
            \n raster_ds index: {rlayer_ds.index}')
        extent_bbox = BoundingBox(minx=extent.xMinimum(), maxx=extent.xMaximum(), miny=extent.yMinimum(), maxy=extent.yMaximum(),
                                  mint=rlayer_ds.index.bounds[4], maxt=rlayer_ds.index.bounds[5])
        ds_sampler = SamTestGridGeoSampler(
            rlayer_ds, size=1024, stride=stride, roi=extent_bbox, units=Units.PIXELS)  # Units.CRS or Units.PIXELS

        if len(ds_sampler) == 0:
            feedback.pushInfo(
                f'No available patch sample inside the chosen extent')
            return {'Input layer dir': rlayer_dir, 'Sample num': len(ds_sampler)}

        feedback.pushInfo(f'Sample number: {len(ds_sampler)}')

        ds_dataloader = DataLoader(
            rlayer_ds, batch_size=1, sampler=ds_sampler, collate_fn=stack_samples)

        self.iPatch = 0
        total = 100 / len(ds_dataloader) if len(ds_dataloader) else 0
        for current, batch in enumerate(ds_dataloader):
            start_time = time.time()
            # Stop the algorithm if cancel button has been clicked
            if feedback.isCanceled():
                break
            feedback.pushInfo(' '.join(str(size)
                              for size in list(batch['image'].shape)))

            batch_input = batch['image']  # .to(device=device)
            features = self.get_sam_feature(sam_model, batch_input)

            end_time = time.time()
            # get the execution time of sam predictor, ms
            elapsed_time = (end_time - start_time) * 1000
            feedback.pushInfo(', '.join(str(size)
                              for size in list(features.shape)))
            feedback.pushInfo(
                f"SAM encoding executed with {elapsed_time:.3f} ms")
            feature_dir = self.save_sam_feature(
                output_dir, batch, features, extent_bbox, model_type)

            # Update the progress bar
            feedback.setProgress(int((current+1) * total))

        if load_feature:
            sam_tool_action: QAction = iface.mainWindow().findChild(QAction,
                                                                    "mActionGeoSamTool")
            sam_tool_action.trigger()
            start_time = time.time()
            while True:
                sam_tool_widget: QgsDockWidget = iface.mainWindow().findChild(QDockWidget, 'GeoSAM')
                current_time = time.time()
                elapsed_time = (current_time - start_time) * 1000
                if sam_tool_widget:
                    load_feature_widget: QgsFileWidget = sam_tool_widget.QgsFile_feature
                    load_feature_widget.setFilePath(feature_dir)
                    sam_tool_widget.pushButton_load_feature.click()  # try sender
                    feedback.pushInfo(
                        'GeoSAM widget found and load feature button clicked in {elapsed_time:.3f} ms')
                    break
                # try 3 seconds
                if elapsed_time > 3000:
                    feedback.pushInfo(
                        'GeoSAM widget not found {elapsed_time:.3f} ms')
                    break
        # Return the results of the algorithm. In this case our only result is
        # the feature sink which contains the processed features, but some
        # algorithms may return multiple feature sinks, calculated numeric
        # statistics, etc. These should all be included in the returned
        # dictionary, with keys matching the feature corresponding parameter
        # or output names.
        return {self.OUTPUT: feature_dir, 'Patch sample num': len(ds_sampler)}

    def initialize_sam(self, model_type: str, sam_ckpt_path: str) -> Sam:
        sam_model = sam_model_registry[model_type](checkpoint=sam_ckpt_path)
        # self.sam_model.to(device=device)
        return sam_model

    @torch.no_grad()
    def get_sam_feature(self, sam_model: Sam, batch_input: Tensor) -> np.ndarray:
        # batch_input = (batch_input - sam_model.pixel_mean) / sam_model.pixel_std
        batch_input = sam_model.preprocess(batch_input)
        features = sam_model.image_encoder(batch_input)
        return features.cpu().numpy()

    def save_sam_feature(
        self,
        export_dir_str: str,
        data_batch: Tensor,
        feature: np.ndarray,
        extent: BoundingBox,
        model_type: str = "vit_h"
    ) -> str:
        export_dir = Path(export_dir_str)
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
            extent_list = [extent.minx, extent.miny, extent.maxx, extent.maxy]
            extent_str = '_'.join(map("{:.6f}".format, extent_list))
            #  Unicode-objects must be encoded before hashing with hashlib and
            #  because strings in Python 3 are Unicode by default (unlike Python 2),
            #  you'll need to encode the string using the .encode method.
            bbox_hash = hashlib.sha256(bbox_str.encode("utf-8")).hexdigest()
            extent_hash = hashlib.sha256(
                extent_str.encode("utf-8")).hexdigest()

            export_dir_sub = (export_dir / filepath.stem /
                              f"sam_feat_{model_type}_{extent_hash[0:16]}")
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
                }
                feature_dataset.update_tags(**tags)
                feature_res = feature_dataset.res[0]
                feature_crs = feature_dataset.crs

            index_df = pd.DataFrame(columns=['minx', 'maxx', 'miny', 'maxy', 'mint', 'maxt',
                                             'filepath',
                                             'crs', 'res'],
                                    index=[self.iPatch])
            index_df['filepath'] = [feature_tiff.name]
            index_df['minx'] = [bbox.minx]
            index_df['maxx'] = [bbox.maxx]
            index_df['miny'] = [bbox.miny]
            index_df['maxy'] = [bbox.maxy]
            index_df['mint'] = [bbox.mint]
            index_df['maxt'] = [bbox.maxt]
            index_df['crs'] = [str(feature_crs)]
            index_df['res'] = [feature_res]
            # append data frame to CSV file, index=False
            index_df.to_csv(feature_csv, mode='a',
                            header=not feature_csv.exists(), index=True)
            self.iPatch += 1

        return str(export_dir_sub)

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return SamProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'geo_sam_encoder'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Geo-SAM Image Encoder')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr('')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return ''

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        return self.tr("Generate image features using SAM image encoder.")

    def icon(self):
        return QIcon_GeoSAMEncoder


class SamEncoder:
    def __init__(self, ckp_path: str, model_type: str = "vit_h", device: str = "cpu") -> None:
        sam_checkpoint = "./checkpoint/sam_vit_h_4b8939.pth"
        self.model_type = model_type
        device = "cuda"
