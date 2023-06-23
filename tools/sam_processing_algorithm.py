import os
import time
from typing import Dict, Any
from pathlib import Path
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsCoordinateTransform,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterBand,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFile,
                       QgsProcessingParameterString,
                       QgsProcessingParameterExtent,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink)
from qgis import processing
from segment_anything import sam_model_registry, SamPredictor
import torch
from .torchgeo_sam import TestGridGeoSampler, SamTestRasterDataset
from torchgeo.samplers import Units
from torchgeo.datasets import BoundingBox, stack_samples
from torch.utils.data import DataLoader
import rasterio
import numpy as np
from torch import Tensor
import hashlib


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
    OUTPUT = 'OUTPUT'
    EXTENT = 'EXTENT'

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

        self.addParameter(
            QgsProcessingParameterString(
                name=self.MODEL_TYPE,
                description=self.tr('Model Type'),
                defaultValue='vit_h',
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
                # createByDefault=True,
                # behavior=QgsProcessingParameterFile.Folder
            )
        )

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
        model_type = self.parameterAsString(
            parameters, self.MODEL_TYPE, context)
        stride = self.parameterAsInt(
            parameters, self.STRIDE, context)

        bbox = self.parameterAsExtent(parameters, self.EXTENT, context)
        # if bbox.isNull() and not rlayer:
        #     raise QgsProcessingException(
        #         self.tr("No reference layer selected nor extent box provided"))

        if not bbox.isNull():
            bboxCrs = self.parameterAsExtentCrs(
                parameters, self.EXTENT, context)
            if bboxCrs != rlayer.crs():
                transform = QgsCoordinateTransform(
                    bboxCrs, rlayer.crs(), context.transformContext())
                bbox = transform.transformBoundingBox(bbox)

        if bbox.isNull() and rlayer:
            bbox = rlayer.extent()  # QgsProcessingUtils.combineLayerExtents(layers, crs, context)

        # output_dir = self.parameterAsFileOutput(
        #     parameters, self.OUTPUT, context)

        output_dir = self.parameterAsString(
            parameters, self.OUTPUT, context)

        # Send some information to the user
        # feedback.pushInfo('CRS is {}'.format(rlayer.crs().authid()))
        # feedback.pushInfo('Band number is {}'.format(rlayer.bandCount()))
        # feedback.pushInfo('Band name is {}'.format(rlayer.bandName(1)))
        feedback.pushInfo('Layer path is {}'.format(
            rlayer.dataProvider().dataSourceUri()))
        feedback.pushInfo('Layer name is {}'.format(rlayer.name()))
        feedback.pushInfo(f'Bands selected: {selected_bands}')
        # feedback.pushInfo('Layer display band name is {}'.format(
        #     rlayer.dataProvider().displayBandName(1)))
        feedback.pushInfo(
            f'Layer extent: minx:{bbox.xMinimum():.2f}, maxx:{bbox.xMaximum():.2f}, miny:{bbox.yMinimum():.2f}, maxy:{bbox.yMaximum():.2f}')

        # If sink was not created, throw an exception to indicate that the algorithm
        # encountered a fatal error. The exception text can be any string, but in this
        # case we use the pre-built invalidSinkError method to return a standard
        # helper text for when a sink cannot be evaluated
        # if sink is None:
        #     raise QgsProcessingException(
        #         self.invalidSinkError(parameters, self.OUTPUT))

        # Compute the number of steps to display within the progress bar and
        # get features from source
        # total = 100.0 / source.featureCount() if source.featureCount() else 0
        # features = source.getFeatures()

        # for current, feature in enumerate(features):
        #     # Stop the algorithm if cancel button has been clicked
        #     if feedback.isCanceled():
        #         break

        #     # Add a feature in the sink
        #     sink.addFeature(feature, QgsFeatureSink.FastInsert)

        #     # Update the progress bar
        #     feedback.setProgress(int(current * total))

        # To run another Processing algorithm as part of this algorithm, you can use
        # processing.run(...). Make sure you pass the current context and feedback
        # to processing.run to ensure that all temporary layer outputs are available
        # to the executed algorithm, and that the executed algorithm can send feedback
        # reports to the user (and correctly handle cancellation and progress reports!)
        if False:
            buffered_layer = processing.run("native:buffer", {
                'INPUT': dest_id,
                'DISTANCE': 1.5,
                'SEGMENTS': 5,
                'END_CAP_STYLE': 0,
                'JOIN_STYLE': 0,
                'MITER_LIMIT': 2,
                'DISSOLVE': False,
                'OUTPUT': 'memory:'
            }, context=context, feedback=feedback)['OUTPUT']

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
        roi = BoundingBox(minx=bbox.xMinimum(), maxx=bbox.xMaximum(), miny=bbox.yMinimum(), maxy=bbox.yMaximum(),
                          mint=rlayer_ds.index.bounds[4], maxt=rlayer_ds.index.bounds[5])
        ds_sampler = TestGridGeoSampler(
            rlayer_ds, size=1024, stride=stride, roi=roi, units=Units.PIXELS)  # Units.CRS or Units.PIXELS

        if len(ds_sampler) == 0:
            return {'Input layer dir': rlayer_dir, 'Sample num': len(ds_sampler)}
        feedback.pushInfo(f'Sample number: {len(ds_sampler)}')

        ds_dataloader = DataLoader(
            rlayer_ds, batch_size=1, sampler=ds_sampler, collate_fn=stack_samples)

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
            self.save_sam_feature(output_dir, batch, features, model_type)

            # Update the progress bar
            feedback.setProgress(int(current * total))

        # Return the results of the algorithm. In this case our only result is
        # the feature sink which contains the processed features, but some
        # algorithms may return multiple feature sinks, calculated numeric
        # statistics, etc. These should all be included in the returned
        # dictionary, with keys matching the feature corresponding parameter
        # or output names.
        return {self.OUTPUT: output_dir, 'Input layer dir': rlayer_dir, 'Sample num': len(ds_sampler)}

    def initialize_sam(self, model_type: str, sam_ckpt_path: str):
        sam_model = sam_model_registry[model_type](checkpoint=sam_ckpt_path)
        # self.sam_model.to(device=device)
        return sam_model

    @torch.no_grad()
    def get_sam_feature(self, sam_model, batch_input):
        batch_input = (batch_input - sam_model.pixel_mean) / \
            sam_model.pixel_std
        features = sam_model.image_encoder(batch_input)
        return features.cpu().numpy()

    def save_sam_feature(self, export_dir_str: str, data_batch: Tensor, feature: np.ndarray, model_type: str = "vit_h"):
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
            bbox = [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy]
            bbox_str = '_'.join(map("{:.6f}".format, bbox))
            # bbox_hash = hashlib.md5()
            #  Unicode-objects must be encoded before hashing with hashlib and
            #  because strings in Python 3 are Unicode by default (unlike Python 2),
            #  you'll need to encode the string using the .encode method.
            # bbox_hash.update(bbox_str.encode("utf-8"))
            bbox_hash = hashlib.sha256(bbox_str.encode("utf-8")).hexdigest()

            export_dir_sub = export_dir / filepath.stem
            # display(export_dir_sub)
            export_dir_sub.mkdir(parents=True, exist_ok=True)
            feature_tiff = export_dir_sub / "sam_feat_{model}_{bbox}.tif".format(
                model=model_type, bbox=bbox_hash)
            # print(feature_tiff)
            with rasterio.open(
                    feature_tiff,
                    mode="w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=band_num,
                    dtype='float32',
                    crs=data_batch['crs'][idx],
                    transform=rio_transform
            ) as feature_dataset:
                # index start from 1, feature[idx, :, :, :] = feature[idx, ...], later is faster
                feature_dataset.write(feature[idx, ...], range(1, band_num+1))
                # pr_mask_dataset.set_band_description(1, 'heatmap')
        return True

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
        return 'gem_sam_encoder'

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
        return self.tr("Example algorithm short description")


class SamEncoder:
    def __init__(self, ckp_path: str, model_type: str = "vit_h", device: str = "cpu") -> None:
        sam_checkpoint = "./checkpoint/sam_vit_h_4b8939.pth"
        self.model_type = model_type
        device = "cuda"
