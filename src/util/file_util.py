# -*- coding: UTF-8 -*-

import pickle
from collections import OrderedDict
import gc
import pathlib
import os
import shutil
import json
from typing import Union

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

    @staticmethod
    def load_dict_from_json(path):
        with open(path, 'r') as f:
            return json.load(f)


class FileWriter(object):

    @staticmethod
    def save_dict_to_json(dict_: Union[dict, OrderedDict], path, excluded_keys=None):

        exclude = set(excluded_keys or [])
        filtered = {k: v for k, v in dict_.items() if k not in exclude}
        with open(path, 'w') as f:
            json.dump(filtered, f, indent=4)

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

    # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
    @staticmethod
    def clear_dir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    @staticmethod
    def list_dir(path, list_full=False):
        if not list_full:
            return os.listdir(path)
        return [os.path.join(path, sub_path) for sub_path in os.listdir(path)]

    @staticmethod
    def recursive_get_all_fnames_under_path(path):

        """
        Get the absolute directories of all files under a path and all its subpaths (recursively)
        """
        return [str(p.resolve()) for p in pathlib.Path(path).rglob("*") if p.is_file()]



