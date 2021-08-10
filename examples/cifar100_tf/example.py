from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100

from cifar100_handler import CIFAR100Handler
from nncompress.compression.pruning import prune, prune_filter
from nncompress.search.nncompress import NNCompress
from nncompress.search.projection import extract_sample_features
from nncompress.search.projection import least_square_projection
from nncompress.backend.tensorflow_.transformation.parser import NNParser
from nncompress import backend as M

data_holder = {}

# load models
def load_model(model_, dataset):
    if model_ == "resnet":
        if dataset == "cifar100":
            path = "saved_models/cifar100_ResNet56v2_model.144.h5"
        else:
            path = "saved_models/cifar10_ResNet56v2_model.151.h5"
    elif model_ == "resnet20":
        if dataset == "cifar100":
            path = "saved_models/cifar100_ResNet20v2_model.100.h5"
        else:
            path = "saved_models/cifar10_ResNet20v2_model.148.h5"
    elif model_ == "resnet164":
        if dataset == "cifar100":
            path = "saved_models/cifar100_ResNet164v2_model.092.h5"
        else:
            path = "saved_models/cifar10_ResNet164v2_model.155.h5"
    elif model_ == "densenet":
        if dataset == "cifar100":
            path = "saved_models/cifar100_densenet_model.186.h5"
        else:
            path = "saved_models/cifar10_densenet_model.192.h5"
    else:
        raise NotImplementedError("")

    model = tf.keras.models.load_model(path)
    return model

def run():

    num_classes = 100
    comp_ratio = 0.25
    model_ = "resnet20"
    model = load_model(model_, "cifar100")

    handler = CIFAR100Handler(num_classes)
    handler.setup(model)

    scores = handler.evaluate(model)
    print(scores)

    keras.utils.plot_model(model, to_file='model.png')

    parser = NNParser(model, None)
    parser.parse()

    nodes = parser.get_nodes(["input_1"])
    v = parser.traverse(nodes)

    torder = {
        name:idx
        for idx, (name, _) in enumerate(v)
    }
    
    sharing_groups = M.get_sharing_groups(model)
    feat_data = {}
    domain = set()
    for g in sharing_groups:
        if len(g) == 1:
            continue
        g_ = [(i, torder[i]) for i in g]
        g_ = sorted(g_, key=lambda x: x[1])
        _, _, history = prune(model, [(g_[0][0], comp_ratio)], mode="channel", method="group_sum")
        for l in history:
            layer = model.get_layer(l)
            if layer not in domain and (layer.__class__.__name__ == "Conv2D" and layer.kernel_size == (1,1) and layer.strides==(1,1)):
                domain.add(layer)

    temp_data = extract_sample_features(model, domain, handler, nsamples=100)
    for layer_name, feat_data_ in temp_data.items():
        feat_data[layer_name] = feat_data_

    best_idx = [-1 for _ in sharing_groups]
    worst_idx = [-1 for _ in sharing_groups]
    for gidx, g in enumerate(sharing_groups):
        if len(g) == 1:
            continue
        g_ = [(i, torder[i]) for i in g]
        g_ = sorted(g_, key=lambda x: x[1])

        # conduct test
        best_ = -1
        worst_ = -1
        for idx, target in enumerate(g_):
            model_, replace_mappings, history = prune(model, [(target[0], comp_ratio)], mode="channel", method="magnitude", handler=handler, calibration=True, nsamples=100, feat_data=feat_data)
            model_.compile(loss='categorical_crossentropy',
                          metrics=['accuracy'])
            scores = handler.evaluate(model_)
            print(scores)
            if scores > best_:
                best_ = scores
                best_idx[gidx] = idx
            if scores < worst_ or worst_ == -1:
                worst_ = scores
                worst_idx[gidx] = idx

        print("group_sum")
        model_, replace_mappings, history = prune(model, [(target[0], comp_ratio)], mode="channel", method="group_sum", handler=handler, calibration=True, nsamples=100, feat_data=feat_data)

        model_.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'])
        scores = handler.evaluate(model_)
        print(scores)

        print("w_group_sum")
        model_, replace_mappings, history = prune(model, [(target[0], comp_ratio)], mode="channel", method="w_group_sum", handler=handler, calibration=True, nsamples=100, feat_data=feat_data)

        model_.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'])
        scores = handler.evaluate(model_)
        print(scores)

        print("random")
        model_, replace_mappings, history = prune(model, [(target[0], comp_ratio)], mode="channel", method="random", handler=handler, calibration=True, nsamples=100, feat_data=feat_data)

        model_.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'])
        scores = handler.evaluate(model_)
        print(scores)


    # best-only
    print("best-only")
    model_ = model
    for gidx, g in enumerate(sharing_groups):
        if len(g) == 1:
            continue

        g_ = [(i, torder[i]) for i in g]
        g_ = sorted(g_, key=lambda x: x[1])

        target = g_[best_idx[gidx]]
        model_, replace_mappings, history = prune(model_, [(target[0], comp_ratio)], mode="channel", method="magnitude", handler=handler, calibration=True, nsamples=100, feat_data=feat_data)

        model_.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'])
        scores = handler.evaluate(model_)
        print(scores)

    print("worst-only")
    model_ = model
    for gidx, g in enumerate(sharing_groups):
        if len(g) == 1:
            continue

        g_ = [(i, torder[i]) for i in g]
        g_ = sorted(g_, key=lambda x: x[1])

        target = g_[worst_idx[gidx]]
        model_, replace_mappings, history = prune(model_, [(target[0], comp_ratio)], mode="channel", method="magnitude", handler=handler, calibration=True, nsamples=100, feat_data=feat_data)

        model_.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'])
        scores = handler.evaluate(model_)
        print(scores)

    print("group_sum")
    model_ = model
    for gidx, g in enumerate(sharing_groups):
        if len(g) == 1:
            continue

        g_ = [(i, torder[i]) for i in g]
        g_ = sorted(g_, key=lambda x: x[1])

        model_, replace_mappings, history = prune(model_, [(g_[0][0], comp_ratio)], mode="channel", method="group_sum", handler=handler, calibration=True, nsamples=100, feat_data=feat_data)

        model_.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'])
        scores = handler.evaluate(model_)
        print(scores)

    print(model.count_params(), model_.count_params())

    print("w_group_sum")
    model_ = model
    for gidx, g in enumerate(sharing_groups):
        if len(g) == 1:
            continue

        g_ = [(i, torder[i]) for i in g]
        g_ = sorted(g_, key=lambda x: x[1])

        model_, replace_mappings, history = prune(model_, [(g_[0][0], comp_ratio)], mode="channel", method="w_group_sum", handler=handler, calibration=True, nsamples=100, feat_data=feat_data)

        model_.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'])
        scores = handler.evaluate(model_)
        print(scores)

    print("random")
    model_ = model
    for gidx, g in enumerate(sharing_groups):
        if len(g) == 1:
            continue

        g_ = [(i, torder[i]) for i in g]
        g_ = sorted(g_, key=lambda x: x[1])

        model_, replace_mappings, history = prune(model_, [(g_[0][0], comp_ratio)], mode="channel", method="random", handler=handler, calibration=True, nsamples=100, feat_data=feat_data)

        model_.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'])
        scores = handler.evaluate(model_)
        print(scores)


if __name__ == "__main__":
    run()
