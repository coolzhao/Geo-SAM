import os
import time
from typing import Dict, Any, List
from pathlib import Path
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import QAction, QDockWidget
from qgis.gui import QgsDockWidget, QgsFileWidget
from qgis.utils import iface
from qgis.core import (QgsProcessing, Qgis,
                       QgsRectangle,
                       QgsCoordinateReferenceSystem,
                       QgsUnitTypes,
                       QgsRasterBandStats,
                       QgsUnitTypes,
                       QgsCoordinateTransform,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingFeedback,
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
                       QgsProcessingParameterCrs,
                       QgsProcessingParameterScale,
                       QgsProcessingParameterExpression,
                       QgsProcessingParameterRange,
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
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
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
    RANGE = 'RANGE'
    RESOLUTION = 'RESOLUTION'
    CRS = 'CRS'
    CUDA = 'CUDA'
    BATCH_SIZE = 'BATCH_SIZE'

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # We add the input vector features source. It can have any kind of
        # geometry.
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.INPUT,
                description=self.tr('Input raster layer or tif file path')
            )
        )
        # add ParameterRasterCalculatorExpression to normalize raster values to 0-255

        self.addParameter(
            QgsProcessingParameterBand(
                name=self.BANDS,
                description=self.tr(
                    'Select no more than three bands (preferably in R G B order)'),
                defaultValue=[1, 2, 3],
                parentLayerParameterName=self.INPUT,
                allowMultiple=True
            )
        )

        self.addParameter(
            QgsProcessingParameterCrs(
                name=self.CRS,
                description=self.tr('Target CRS (default to original CRS)'),
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.RESOLUTION,
                description=self.tr(
                    'Target resolution in meters (default to native resolution)'),
                type=QgsProcessingParameterNumber.Double,
                optional=True,
                minValue=0,
                maxValue=100000
            )
        )

        # expression for scaling the raster values to [0,255]
        self.addParameter(
            QgsProcessingParameterRange(
                name=self.RANGE,
                description=self.tr(
                    'The input data value range to be rescaled to [0, 255] (default to min and max values of the image)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=None,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterExtent(
                name=self.EXTENT,
                description=self.tr(
                    'Processing extent (default to the entire image)'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.STRIDE,
                # large images will be sampled into patches in a grid-like fashion
                description=self.tr(
                    'Stride (the bigger the stride, the smaller the overlap)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=512,
                minValue=1,
                maxValue=1024
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                name=self.CKPT,
                description=self.tr(
                    'SAM checkpoint path (download in advance)'),
                extension='pth',
            )
        )

        self.model_type_options = ['vit_h', 'vit_l', 'vit_b']
        self.addParameter(
            QgsProcessingParameterEnum(
                name=self.MODEL_TYPE,
                description=self.tr(
                    'SAM model type: b for base, l for large, h for huge'),
                options=self.model_type_options,
                defaultValue=0,  # 'vit_h'
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT,
                self.tr("Output folder"),
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CUDA,
                self.tr("Use GPU if CUDA is available."),
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.BATCH_SIZE,
                # large images will be sampled into patches in a grid-like fashion
                description=self.tr(
                    'Batch size (take effect if choose to use GPU and CUDA is available)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=1,
                maxValue=1024
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOAD,
                self.tr("Load output features in Geo-SAM tool after processing"),
                defaultValue=True
            )
        )

        # self.addOutput()

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        rlayer = self.parameterAsRasterLayer(
            parameters, self.INPUT, context)
        if rlayer is None:
            raise QgsProcessingException(
                self.invalidRasterError(parameters, self.INPUT))

        self.selected_bands = self.parameterAsInts(
            parameters, self.BANDS, context)
        if len(self.selected_bands) > 3:
            raise QgsProcessingException(
                # self.tr("SAM only supports three-band RGB image!")
                self.tr("Please choose no more three bands")
            )

        ckpt_path = self.parameterAsFile(
            parameters, self.CKPT, context)
        model_type_idx = self.parameterAsEnum(
            parameters, self.MODEL_TYPE, context)
        stride = self.parameterAsInt(
            parameters, self.STRIDE, context)
        res = self.parameterAsDouble(
            parameters, self.RESOLUTION, context)
        crs = self.parameterAsCrs(
            parameters, self.CRS, context)
        extent = self.parameterAsExtent(
            parameters, self.EXTENT, context)
        self.load_feature = self.parameterAsBoolean(
            parameters, self.LOAD, context)
        self.use_gpu = self.parameterAsBoolean(
            parameters, self.CUDA, context)
        batch_size = self.parameterAsInt(
            parameters, self.BATCH_SIZE, context)
        range_value = self.parameterAsRange(
            parameters, self.RANGE, context)
        output_dir = self.parameterAsString(
            parameters, self.OUTPUT, context)

        rlayer_data_provider = rlayer.dataProvider()
        # handle value range
        if (not np.isnan(range_value[0])) and (not np.isnan(range_value[1])):
            feedback.pushInfo(
                f'Input data value range to be rescaled: {range_value[0]},{range_value[1]}')
        else:
            band_stats = rlayer_data_provider.bandStatistics(
                bandNo=self.selected_bands[0], stats=QgsRasterBandStats.Min)
            range_value[0] = band_stats.minimumValue
            band_stats = rlayer_data_provider.bandStatistics(
                bandNo=self.selected_bands[0], stats=QgsRasterBandStats.Max)
            range_value[1] = band_stats.maximumValue
            feedback.pushInfo(
                f'Input data value range to be rescaled: {range_value[0]},{range_value[1]}(automatically created based on min-max value of raster layer.)')
        # if bbox.isNull() and not rlayer:
        #     raise QgsProcessingException(
        #         self.tr("No reference layer selected nor extent box provided"))

        # handle crs
        if crs is None or not crs.isValid():
            crs = rlayer.crs()
            # feedback.pushInfo(
            #     f'Layer CRS unit is {crs.mapUnits()}')  # 0 for meters, 6 for degrees, 9 for unknown
            # feedback.pushInfo(
            #     f'whether the CRS is a geographic CRS (using lat/lon coordinates) {crs.isGeographic()}')
            # if crs.mapUnits() == Qgis.DistanceUnit.Degrees:
            #     crs = self.estimate_utm_crs(rlayer.extent())

        # target crs should use meters as units
        # if crs.mapUnits() != Qgis.DistanceUnit.Meters:
        #     feedback.pushInfo(
        #         f'Layer CRS unit is {crs.mapUnits()}')
        #     feedback.pushInfo(
        #         f'whether the CRS is a geographic CRS (using lat/lon coordinates) {crs.isGeographic()}')
        #     raise QgsProcessingException(
        #         self.tr("Only support CRS with the units as meters")
        #     )

        # if res is not provided, get res info from rlayer
        if np.isnan(res) or res == 0:
            res = rlayer.rasterUnitsPerPixelX()  # rasterUnitsPerPixelY() is negative
        else:
            # when given res in meters by users, convert crs to utm if the original crs unit is degree
            if crs.mapUnits() != Qgis.DistanceUnit.Meters:
                if rlayer.crs().mapUnits() == Qgis.DistanceUnit.Degrees:
                    # estimate utm crs based on layer extent
                    crs = self.estimate_utm_crs(rlayer.extent())
                else:
                    raise QgsProcessingException(
                        f"Resampling of image with the CRS of {crs.authid()} in meters is not supported.")
            # else:
            #     res = (rlayer_extent.xMaximum() -
            #            rlayer_extent.xMinimum()) / rlayer.width()
        self.res = res

        # handle extent
        if extent.isNull():
            extent = rlayer.extent()  # QgsProcessingUtils.combineLayerExtents(layers, crs, context)
            # rlayer_extent = rlayer.extent()
            extent_crs = rlayer.crs()
        else:
            extent_crs = self.parameterAsExtentCrs(
                parameters, self.EXTENT, context)
        # if extent crs != target crs, convert it to target crs
        if extent_crs != crs:
            transform = QgsCoordinateTransform(
                extent_crs, crs, context.transformContext())
            extent = transform.transformBoundingBox(extent)
            # if rlayer.crs().mapUnits() != Qgis.DistanceUnit.Meters:
            #     rlayer_extent = transform.transformBoundingBox(
            #         rlayer.extent())

        # Send some information to the user
        feedback.pushInfo(
            f'Layer path: {rlayer_data_provider.dataSourceUri()}')
        # feedback.pushInfo(
        #     f'Layer band scale: {rlayer_data_provider.bandScale(self.selected_bands[0])}')
        feedback.pushInfo(f'Layer name: {rlayer.name()}')
        feedback.pushInfo(f'Layer CRS is {rlayer.crs().authid()}')
        feedback.pushInfo(
            f'Layer Pixel size is {rlayer.rasterUnitsPerPixelX()}, {rlayer.rasterUnitsPerPixelY()}')
        feedback.pushInfo(f'Bands selected: {self.selected_bands}')

        feedback.pushInfo(f'Target CRS is {crs.authid()}')
        # feedback.pushInfo('Band number is {}'.format(rlayer.bandCount()))
        # feedback.pushInfo('Band name is {}'.format(rlayer.bandName(1)))
        feedback.pushInfo(f'Target resolution: {self.res}')
        # feedback.pushInfo('Layer display band name is {}'.format(
        #     rlayer.dataProvider().displayBandName(1)))
        feedback.pushInfo(
            (f'Processing extent: minx:{extent.xMinimum():.6f}, maxx:{extent.xMaximum():.6f},'
             f'miny:{extent.yMinimum():.6f}, maxy:{extent.yMaximum():.6f}'))

        model_type = self.model_type_options[model_type_idx]
        if model_type not in os.path.basename(ckpt_path):
            raise QgsProcessingException(
                self.tr("Model type does not match the checkpoint"))

        self.sam_model = self.initialize_sam(
            model_type=model_type, sam_ckpt_path=ckpt_path)

        # feedback.pushInfo(
        #     f'SAM Image Size: {self.sam_model.image_encoder.img_size}')

        rlayer_path = rlayer.dataProvider().dataSourceUri()
        rlayer_dir = os.path.dirname(rlayer_path)
        rlayer_name = os.path.basename(rlayer_path)

        SamTestRasterDataset.filename_glob = rlayer_name
        SamTestRasterDataset.all_bands = [
            rlayer.bandName(i_band) for i_band in range(1, rlayer.bandCount()+1)
        ]
        # currently only support rgb bands
        input_bands = [rlayer.bandName(i_band)
                       for i_band in self.selected_bands]
        # ensure only three bands are used, less than three bands will be broadcasted to three bands
        input_bands = (input_bands * 3)[0:3]

        if self.res:
            rlayer_ds = SamTestRasterDataset(
                root=rlayer_dir, crs=crs.toWkt(), res=self.res, bands=input_bands, cache=False)
        else:
            rlayer_ds = SamTestRasterDataset(
                root=rlayer_dir, crs=crs.toWkt(), res=None, bands=input_bands, cache=False)
        # \n raster_ds crs: {str(CRS(rlayer_ds.crs))}, \
        feedback.pushInfo(
            f'\n RasterDS info: \
            \n filename_glob: {rlayer_ds.filename_glob} \
            \n input bands: {rlayer_ds.bands}, \
            \n all bands: {rlayer_ds.all_bands}, \
            \n resolution: {rlayer_ds.res}, \
            \n index: {rlayer_ds.index} \n')
        extent_bbox = BoundingBox(minx=extent.xMinimum(), maxx=extent.xMaximum(), miny=extent.yMinimum(), maxy=extent.yMaximum(),
                                  mint=rlayer_ds.index.bounds[4], maxt=rlayer_ds.index.bounds[5])
        ds_sampler = SamTestGridGeoSampler(
            rlayer_ds, size=1024, stride=stride, roi=extent_bbox, units=Units.PIXELS)  # Units.CRS or Units.PIXELS

        if len(ds_sampler) == 0:
            feedback.pushInfo(
                f'No available patch sample inside the chosen extent')
            return {'Input layer dir': rlayer_dir, 'Sample num': len(ds_sampler.res),
                    'Sample size': len(ds_sampler.size), 'Sample stride': len(ds_sampler.stride)}

        if not torch.cuda.is_available() or not self.use_gpu:
            batch_size = 1
        ds_dataloader = DataLoader(
            rlayer_ds, batch_size=batch_size, sampler=ds_sampler, collate_fn=stack_samples)

        feedback.pushInfo(f'Patch sample number: {len(ds_sampler)}')
        feedback.pushInfo(f'Total batch number: {len(ds_dataloader)}')

        self.iPatch = 0
        self.feature_dir = ""
        total = 100 / len(ds_dataloader) if len(ds_dataloader) else 0
        for current, batch in enumerate(ds_dataloader):
            start_time = time.time()
            # Stop the algorithm if cancel button has been clicked
            if feedback.isCanceled():
                self.load_feature = False
                feedback.pushInfo(self.tr("Processing is canceled by user."))
                break
            feedback.pushInfo(f'Batch no. {current+1} loaded')
            feedback.pushInfo('img_shape: ' + ','.join(str(size)
                              for size in list(batch['img_shape'])))
            feedback.pushInfo('patch_size: ' + ','.join(str(size)
                              for size in list(batch['image'].shape)))

            batch_input = batch['image']  # .to(device=device)
            if (not np.isnan(range_value[0])) and (not np.isnan(range_value[1])):
                batch_input = self.rescale_img(
                    batch_input=batch_input, range_value=range_value)
            if not self.get_sam_feature(batch_input, feedback):
                break

            end_time = time.time()
            # get the execution time of sam predictor, ms
            elapsed_time = (end_time - start_time) * 1000
            feedback.pushInfo('feature_shape:' + ','.join(str(size)
                              for size in list(self.features.shape)))
            feedback.pushInfo(
                f"SAM encoding executed with {elapsed_time:.3f} ms")
            self.feature_dir = self.save_sam_feature(
                output_dir, batch, self.features, extent_bbox, model_type)

            # Update the progress bar
            feedback.setProgress(int((current+1) * total))

        if torch.cuda.is_available() and self.use_gpu:
            # self.sam_model.to(device='cpu')
            # batch_input.to(device='cpu')
            del self.sam_model, batch_input
            torch.cuda.empty_cache()
        # Return the results of the algorithm. In this case our only result is
        # the feature sink which contains the processed features, but some
        # algorithms may return multiple feature sinks, calculated numeric
        # statistics, etc. These should all be included in the returned
        # dictionary, with keys matching the feature corresponding parameter
        # or output names.
        return {"Output feature path": self.feature_dir, 'Patch samples saved': self.iPatch, self.LOAD: self.load_feature}

    # used to handle any thread-sensitive cleanup which is required by the algorithm.
    def postProcessAlgorithm(self, context, feedback) -> Dict[str, Any]:
        if self.load_feature:
            sam_tool_action: QAction = iface.mainWindow().findChild(QAction,
                                                                    "mActionGeoSamTool")
            sam_tool_action.trigger()
            start_time = time.time()
            while True:
                if feedback.isCanceled():
                    feedback.pushInfo(
                        self.tr("Loading feature is canceled by user."))
                    break
                sam_tool_widget: QgsDockWidget = iface.mainWindow().findChild(QDockWidget, 'GeoSAM')
                current_time = time.time()
                elapsed_time = (current_time - start_time) * 1000
                if sam_tool_widget:
                    load_feature_widget: QgsFileWidget = sam_tool_widget.QgsFile_feature
                    load_feature_widget.setFilePath(self.feature_dir)
                    sam_tool_widget.pushButton_load_feature.click()  # try sender
                    feedback.pushInfo(
                        f'GeoSAM widget found and load feature button clicked in {elapsed_time:.3f} ms')
                    break
                # try 3 seconds
                if elapsed_time > 3000:
                    feedback.pushInfo(
                        f'GeoSAM widget not found {elapsed_time:.3f} ms')
                    break
        return {"Output feature path": self.feature_dir, 'Patch samples saved': self.iPatch, self.LOAD: self.load_feature}

    def initialize_sam(self, model_type: str, sam_ckpt_path: str) -> Sam:
        sam_model = sam_model_registry[model_type](
            checkpoint=sam_ckpt_path)
        if torch.cuda.is_available() and self.use_gpu:
            sam_model.to(device='cuda')
        return sam_model

    @torch.no_grad()
    def get_sam_feature(self, batch_input: Tensor, feedback: QgsProcessingFeedback) -> bool:
        batch_input = batch_input.to(device=self.sam_model.device)
        batch_input = ((batch_input - self.sam_model.pixel_mean) /
                       self.sam_model.pixel_std)
        # batch_input = sam_model.preprocess(batch_input)
        try:
            features = self.sam_model.image_encoder(batch_input)
        except RuntimeError as inst:
            # torch.cuda.OutOfMemoryError
            if 'CUDA out of memory' in inst.args[0]:
                # del self.sam_model, batch_input
                # torch.cuda.empty_cache()
                feedback.pushWarning(
                    "\n !!!CUDA out of memory, try to choose a smaller batch size.!!!")
                feedback.pushWarning(
                    f'Error type: {type(inst).__name__}, context: {inst}'
                )
            # raise QgsProcessingException(
            #     f'Error type: {type(inst).__name__}, context: {inst}')
            return False
        except Exception as err:
            raise QgsProcessingException(f"Unexpected {err=}, {type(err)=}")
        # batch_input = batch_input.to(device='cpu')
        # torch.cuda.empty_cache()
        self.features = features.cpu().numpy()
        return True

    def rescale_img(self, batch_input: Tensor, range_value: List[float]) -> Tensor:
        'rescale input image to [0,255]'
        range_min = range_value[0]
        range_max = range_value[1]
        batch_output = (batch_input - range_min)*255/(range_max - range_min)
        return batch_output

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

            bands_str = '_'.join([str(band) for band in self.selected_bands])
            export_dir_sub = (export_dir / filepath.stem /
                              f"sam_feat_{model_type}_res_{round(self.res)}m_bands_{bands_str}_{extent_hash[0:16]}")
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

    def estimate_utm_crs(self, extent: QgsRectangle):
        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=extent.xMinimum(),
                south_lat_degree=extent.yMinimum(),
                east_lon_degree=extent.xMaximum(),
                north_lat_degree=extent.yMaximum(),
            ),
        )
        utm_crs = CRS.from_epsg(utm_crs_list[0].code)
        utm_crs = QgsCoordinateReferenceSystem(str(utm_crs))
        return utm_crs

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
