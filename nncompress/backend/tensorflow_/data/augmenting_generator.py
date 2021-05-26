from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(images, labels):
    lam = np.random.beta(1.0, 1.0)
    rand_index = np.random.permutation(images.shape[0])

    labels_a = labels
    labels_b = labels[rand_index]
    images_ = np.copy(images)
    for idx in range(images.shape[0]):
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.shape, lam)
        images[idx, bbx1:bbx2, bby1:bby2, :] = images_[rand_index[idx], bbx1:bbx2, bby1:bby2, :]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.shape[1] * images.shape[2]))
    labels = lam * labels_a + (1 - lam) * labels_b
    return images, labels

class AugmentingGenerator(tf.keras.utils.Sequence):

    def __init__(self, generator, method=None, **kwargs):
       self._generator = generator
       self._method = method
       self._kwargs = kwargs

    def __getitem__(self, index):
        if self._method is None: # mirror
            return self._generator[index]
        else:
            return self._method(*self._generator[index], **self._kwargs)

    def __len__(self):
        return len(self._generator)
