#!/usr/bin/python
# -*- coding: UTF-8 -*-

# import pandas as pd
# from osgeo import gdal
# import numpy as np
import sys
import re
import pickle
from collections import OrderedDict
import gc
import pathlib
import os
import shutil
import pandas as pd
import mne


class FileReader(object):

    @staticmethod
    def load_pickle(fname, path=None):

        if path is not None:
            fname = os.path.join(path, fname)

        gc.disable()
        with open(fname, 'rb') as f:
            ret = pickle.load(f)
        gc.enable()
        return ret


class FileWriter(object):

    @staticmethod
    def dump_pickle(obj, fname, path=None):

        if path is not None:
            fname = os.path.join(path, fname)

        gc.disable()
        with open(fname, 'wb') as f:
            pickle.dump(obj, f, protocol=-1)
        gc.enable()


class DirProcessor(object):

    @staticmethod
    def create_dir(path, recursive=True):
        path_ = pathlib.Path(path)
        path_.mkdir(parents=recursive, exist_ok=True)
