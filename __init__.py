# -----------------------------------------------------------
# Copyright (C) 2023 CryoLab CUHK
# -----------------------------------------------------------
import os
import inspect
from .geo_sam_tool import Geo_SAM

cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]


def classFactory(iface):
    return Geo_SAM(iface, cmd_folder)
