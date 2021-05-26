from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.keras import backend as K

def count_all_params(model, trainable_only=False):
    trainable = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    if trainable_only:
        return trainable
    else:
        return trainable + non_trainable
