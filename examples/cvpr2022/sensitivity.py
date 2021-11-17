import types
import json
from collections import OrderedDict
import math

import tensorflow as tf
from tensorflow import keras
import numpy as np
from numba import njit
from numpy import dot
from numpy.linalg import norm as npnorm

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer
from nncompress.backend.tensorflow_.transformation import parse, inject
from nncompress.backend.tensorflow_ import SimplePruningGate, DifferentiableGate
from nncompress.backend.tensorflow_.transformation import parse, inject, cut, unfold
from nncompress import backend as M

from dc import init_gate

def training_loss(model, train_data_generator):
    loss = 0.0
    cnt = 0
    for X, y in train_data_generator:
        logits = model(X)
        loss += tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(logits, y))
        cnt += 1
        print(cnt % 100)
    return loss/cnt

def sensitivity_test(model,
                model_handler,
                train_data_generator,
                custom_objects=None,
                with_splits=True,
                validate=None):

    model = unfold(model, custom_objects)

    parser = PruningNNParser(model, custom_objects=custom_objects, gate_class=SimplePruningGate)
    parser.parse()

    gmodel, gate_mapping = parser.inject(avoid=None, with_mapping=True, with_splits=with_splits)
    init_gate(gmodel)

    # compute groups
    def compare_key(x):
        id_ = x.name.split("_")[-1]
        if id_ == "gate":
            id_ = 0
        else:
            id_ = int(id_)
        return id_

    l2g = {}
    for layer, flow in gate_mapping:
        l2g[layer] = gate_mapping[(layer, flow)][0]["config"]["name"]

    groups = []
    groups_ = parser.get_sharing_groups()
    for g in groups_:
        gate_group = set()
        for layer in g:
            if layer in l2g:
                gate_group.add(gmodel.get_layer(l2g[layer]))
        if len(gate_group) == 0:
            continue
        gate_group = sorted(list(gate_group), key=compare_key) # sorted by alphabetical order
        groups.append(gate_group)

    v = parser.traverse()
    torder = {
        name:idx
        for idx, (name, _) in enumerate(v)
    }

    groups_ = parser.get_sharing_groups()
    ordered_groups = []
    for g in groups_:
        ordered_groups.append((g, torder[g[0]]))

    ordered_groups = sorted(ordered_groups, key=lambda x: x[1])[:-1] # remove last

    # Sensitivity test
    ordered_groups.reverse()
    for g in ordered_groups:
        print(g)
        print(l2g[g[0][0]])
        _gate = gmodel.get_layer(l2g[g[0][0]])
        length = _gate.gates.shape[0]
        mask = np.ones(length,)
        for i in range(length):
            mask[i] = 0.0
            for l in g[0]:
                gate = gmodel.get_layer(l2g[l])
                gate.gates.assign(mask)

            print(i, validate(gmodel))
            #print(training_loss(gmodel, train_data_generator))
            mask[i] = 1.0
            for l in g[0]:
                gate = gmodel.get_layer(l2g[l])
                gate.gates.assign(mask)
        break
