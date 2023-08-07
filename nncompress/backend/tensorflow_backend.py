from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def get_type(cls_name):
    from nncompress.backend import tensorflow_
    if hasattr(tensorflow_, cls_name):
        return getattr(tensorflow_, cls_name)
    else:
        raise NotImplementedError

def cast(x, dtype=np.float32):
    if type(dtype) == str:
        return tf.cast(x, dtype=getattr(tf, dtype))
    else:
        return tf.cast(x, dtype=dtype)

def function(func, *args, **kwargs):
    if hasattr(tf, func):
        f = getattr(tf, func)
    elif hassattr(K, func):
        f = getattr(K, func)
    else:
        raise NotImplementedError("`%s` is not supported." % func)
    assert(callable(f))
    return f(*args, **kwargs)

def floor(x):
    return tf.math.floor(x)

def round(x):
    return tf.math.round(x)

def sum(x):
    return tf.math.reduce_sum(x)

def norm(x, p):
    return tf.norm(x, ord=p)

def cmul(data, mask):
    # N W H C
    if data.dtype != mask.dtype:
        mask = tf.cast(mask, data.dtype)
    return data * mask

def concat(x, y, dim=0):
    return tf.concat([x, y], axis=dim)

def get_out_channel_idx():
    return -1

def get_weights(model, layer_name):
    return model.get_layer(layer_name).get_weights()

def weight_transfer(a, b, exclude=None):
    if exclude is None:
        exclude = set()
    elif type(exclude) != set:
        exclude = set(exclude)
    for layer in a.layers:
        if layer.name not in exclude:
            b.get_layer(layer.name).set_weights(layer.get_weights())

def copy_(model):
    model_ = tf.keras.models.clone_model(model)
    model_.set_weights(model.get_weights())
    return model_

def prune_filter(model, domain, mode="channel", custom_objects=None):
    from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser
    domain = copy.deepcopy(domain)
    if mode == "channel": # it supports `channel_pruning` only now.
        parser = PruningNNParser(model, custom_objects=custom_objects)
        parser.parse()
        avoid = parser.get_last_transformers()
        for a in avoid:
            domain.remove(a)
    return domain

def get_sharing_layers(model, target, custom_objects=None):
    from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser
    parser = PruningNNParser(model, custom_objects=custom_objects)
    parser.parse()

    if type(target) == list:
        ret = {}
        for t in target:
            try:
                ret[t] = parser.get_sharing_layers(t)
            except ValueError:
                ret[t] = None
        return ret
    else:
        return parser.get_sharing_layers(target)

def get_sharing_groups(model, custom_objects=None):
    from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser
    parser = PruningNNParser(model, custom_objects=custom_objects)
    parser.parse()
    model_ = parser.inject()
    tf.keras.utils.plot_model(model_, to_file="gmodel.png", show_shapes=True)
    return parser.get_sharing_groups()

def get_topology(model, custom_objects=None):
    from nncompress.backend.tensorflow_.transformation.parser import NNParser
    parser = NNParser(model, custom_objects)
    parser.parse()
    return parser.get_topology()

def prune(model, masking, mode="channel", custom_objects=None):
    from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser
    if mode == "channel":
        parser = PruningNNParser(model, custom_objects=custom_objects)
        parser.parse()
        model_ = parser.inject()
        for t, g in parser.get_t2g().items():
            g = model_.get_layer(g)
            old = g.get_weights()[0]
            g.set_weights([np.ones_like(old)])
    else:
        model_ = copy_(model)
        model = model_
        history_ = {}

    if mode == "channel":
        avoid = parser.get_last_transformers()
    for target, mask in masking:
        if mode == "channel" and target in avoid:
            continue
        layer = model.get_layer(target)
        weights = layer.get_weights()
        if weights[0].shape != mask.shape: # channel pruning
            t2g = parser.get_t2g()
            gate_layer = model_.get_layer(t2g[target])
            gate_layer.set_weights([mask])
        else: # weight pruning
            w_ = weights[0] * mask
            new_weights = [w_]
            for w in weights[1:]:
                new_weights.append(w)
            layer.set_weights(new_weights)
            history_[layer.name] = (mask == 1.0)
    if mode == "channel":
        model, history = parser.cut(model_, return_history=True)
        history_ = {}
        for layer, mask in history.items():
            if mask is None:
                continue
            if (mask[0] is None or np.sum(mask[0]) == mask[0].shape[0]) and\
                (mask[1] is None or np.sum(mask[1]) == mask[1].shape[0]):
                continue
            history_[layer] = mask
    return model, history_

def add_prefix(model, prefix, custom_objects=None, val_check=None, not_change_model_name=False, not_change_input=False):
    model_dict = json.loads(model.to_json())
    if not not_change_model_name:
        model_dict["config"]["name"] = prefix + model_dict["config"]["name"]
    is_input = set()
    for layer in model_dict["config"]["layers"]:
        if layer["class_name"] == "InputLayer":
            if "name" in layer:
                is_input.add(layer["name"])
            else:
                is_input.add(layer["config"]["name"])
    for layer in model_dict["config"]["layers"]:
        if layer["class_name"] == "InputLayer" and not_change_input:
            continue
        layer["name"] = prefix + layer["name"]
        if "name" in layer["config"]:
            layer["config"]["name"] = prefix + layer["config"]["name"]

        for flow in layer["inbound_nodes"]:
            if type(flow[0]) == str:
                inbound = flow
                if inbound[0] in is_input and not_change_input:
                    continue
                if "value" in inbound[-1] and inbound[-1]["value"][0] == inbound[0]:
                    inbound[-1]["value"][0] = prefix + inbound[0]
                inbound[0] = prefix + inbound[0]
            else:
                for inbound in flow:
                    if inbound[0] in is_input and not_change_input:
                        continue
                    if "value" in inbound[-1] and inbound[-1]["value"][0] == inbound[0]:
                        inbound[-1]["value"][0] = prefix + inbound[0]
                    inbound[0] = prefix + inbound[0]

    if not not_change_input:
        for input_layer in model_dict["config"]["input_layers"]:
            input_layer[0] = prefix + input_layer[0]
    for output_layer in model_dict["config"]["output_layers"]:
        output_layer[0] = prefix + output_layer[0]

    model_json = json.dumps(model_dict)
    ret = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
    for layer in model.layers:
        if layer.name in is_input:
            continue
        ret.get_layer(prefix+layer.name).set_weights(layer.get_weights())

    if val_check is not None:
        data = np.random.rand(*val_check)
        left = model(data)
        right = ret(data)
        assert np.all(left == right)
    return ret
