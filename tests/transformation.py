from __future__ import print_function

import os
import json
from unittest import TestCase

import tensorflow as tf
from tensorflow import keras

from tests import common
from nncompress.compression.lowrank import decompose
from nncompress.compression.pruning import prune
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser

def compute_nodes_edges(model):
    model_dict = json.loads(model.to_json())
    n = 0
    m = 0
    for layer in model_dict["config"]["layers"]:
        n += 1
        for flow in layer["inbound_nodes"]:
            for inbound in flow:
                m += 1
    return n, m

class TransformationTest(TestCase):

    def test_gate_injection_01(self):
        resnet = common.request_model("json")
        parser = PruningNNParser(resnet)
        parser.parse()

        model = parser.inject()
        keras.utils.plot_model(model, to_file="model.png")
        n, m = compute_nodes_edges(model)
        self.assertEqual(n, 320)
        self.assertEqual(m, 404)
