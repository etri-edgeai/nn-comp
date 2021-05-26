from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pathlib import Path

MLCK_NAME = ".mlck"

def get_mlck_path():
    mlck_path = os.path.join(str(Path.home()), MLCK_NAME)
    if not os.path.exists(mlck_path):
        os.mkdir(mlck_path)
    return mlck_path

def get_data_path():
    mlck_path = get_mlck_path()
    data_path = os.path.join(get_mlck_path(), "data")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    return data_path

def get_saved_model_path():
    save_path = os.path.join(get_mlck_path(), "saved_models")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    return save_path
