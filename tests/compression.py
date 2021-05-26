from __future__ import print_function

import os

from unittest import TestCase

import tensorflow as tf
from tensorflow import keras

from tests import common
from nncompress.compression.lowrank import decompose
from nncompress.compression.pruning import prune
from nncompress.backend.tensorflow_.utils import count_all_params

class CompressionTest(TestCase):

    def compress(self, model_key, method):
        model = common.request_model(model_key)
        nparams = count_all_params(model)
        acc = common.helper.evaluate(model)
        ret = method(model)
        if type(ret) == tuple:
            if len(ret) == 2:
                model, replace_mappings = ret
            elif len(ret) == 3:
                model, replace_mappings, history = ret
        new_nparams = count_all_params(model)
        new_acc = common.helper.evaluate(model)
        return model, nparams, acc, new_nparams, new_acc

    def test_01_svd(self):
        model, nparams, acc, new_nparams, new_acc = self.compress("seq", method=lambda x:decompose(x, [("dense", 0.5)]))
        self.assertEqual(int(new_nparams), 838276)

    def test_02_tucker(self):
        model, nparams, acc, new_nparams, new_acc = self.compress("seq", method=lambda x:decompose(x, [("conv2d_3", 0.5)]))
        self.assertEqual(int(new_nparams), 1273476)

    def test_03_weight_pruning(self):
        model, nparams, acc, new_nparams, new_acc = self.compress("seq", method=lambda x:prune(x, [("conv2d_3", 0.5)], mode="weight"))
        self.assertEqual(int(new_nparams), 1297028)

    def test_04_channel_pruning(self):
        model, nparams, acc, new_nparams, new_acc = self.compress("seq", method=lambda x:prune(x, [("conv2d_3", 0.5)]))
        self.assertEqual(int(new_nparams), 707749)
