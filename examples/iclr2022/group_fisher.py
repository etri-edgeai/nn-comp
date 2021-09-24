
import types
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser
from nncompress.backend.tensorflow_.transformation import parse, inject
from nncompress.backend.tensorflow_ import SimplePruningGate, DifferentiableGate


def find_all(model, Target):
    ret = []
    for layer in model.layers:
        if hasattr(layer, "layers"):
            ret += find_all(layer, Target)
        else:
            if layer.__class__ == Target:
                ret.append(layer)
    return ret

def find_all_dict(model_dict, target):
    ret = []
    for layer in model_dict["config"]["layers"]:
        if "layers" in layer["config"]:
            ret += find_all_dict(layer, target)
        else:
            if layer["class_name"] == target:
                ret.append(layer)
    return ret

def get_layers(model):
    ret = {}
    for layer in model.layers:
        if hasattr(layer, "layers"):
            ret.update(get_layers(layer))
        else:
            ret[layer.name] = layer
    return ret

def compute_sparsity(gates):
    sum_ = 0
    alive = 0
    for g in gates:
        sum_ += g.ngates
        alive += np.sum(g.gates.numpy())
    return 1.0 - alive / sum_

class PruningCallback(keras.callbacks.Callback):

    def __init__(self, model, norm, targets, target_ratio=0.5, period=10):
        if not hasattr(model, "grad_holder"):
            raise ValueError('make_group_fisher calling first.')
        self.model = model
        self.norm = norm
        self.targets = targets
        self.period = period
        self.target_ratio = target_ratio
        self._iter = 0

    def on_train_batch_end(self, batch, logs=None):
        self._iter += 1
        if self._iter % self.period == 0 and not (len(self.model.grad_holder) == 1 and self.model.grad_holder[0] is None):
            # Pruning upon self.grad_holder
            #gates = find_all(self.model, SimplePruningGate)
            gates = self.targets

            min_val = -1
            min_idx = (-1, -1)
            for idx, layer in enumerate(gates):
                gates_ = layer.gates.numpy()
                if np.sum(gates_) == 1.0: # min channels.
                    continue
                norm_ = self.norm[layer.name]
                sum_ = 0
                for grad in layer.grad_holder:
                    sum_ += grad
                cscore = sum_ / (len(layer.grad_holder) * norm_)

                for i in range(cscore.shape[0]):
                    if (min_idx[0] == -1 or min_val > cscore[i]) and gates_[i] == 1.0:
                        min_val = cscore[i]
                        min_idx = (idx, i)
                assert min_idx[0] != -1

            min_layer = gates[min_idx[0]] 
            gates_ = min_layer.gates.numpy()
            gates_[min_idx[1]] = 0.0
            min_layer.gates.assign(gates_)

            print(min_layer.name, np.sum(min_layer.gates.numpy()) / min_layer.ngates, compute_sparsity(gates))
            collecting = compute_sparsity(gates) <= self.target_ratio
            for layer in gates:
                layer.grad_holder = []
                layer.collecting = collecting


def compute_act(layers, layer_name, batch_size, is_input_gate=False):
    layer = layers[layer_name]
    if layer.__class__.__name__ == "Conv2D" or\
        (layer.__class__.__name__ == "DepthwiseConv2D" and not is_input_gate):
        # DepthwiseConv2D only contributes for the gate associated with its outputs.
        assert layer.groups == 1
        return batch_size * np.prod(layer.get_weights()[0].shape[0:2])
    else:
        return 0.0

def make_group_fisher(model, batch_size, custom_objects=None, avoid=None, period=25, target_ratio=0.5, enable_norm=True):
    """

    """
    parsers = parse(model, PruningNNParser, custom_objects=custom_objects, gate_class=SimplePruningGate)
    gmodel, gate_mapping = inject(parsers, avoid=avoid)
    keras.utils.plot_model(gmodel, to_file="gg.png", expand_nested=True)

    targets = find_all(gmodel, SimplePruningGate)
    targets_ = set([target.name+"/gates:0" for target in targets])
    gmodel.grad_holder = []

    gmodel_dict = json.loads(gmodel.to_json())
    gates_dict = find_all_dict(gmodel_dict, "SimplePruningGate")
    layers = get_layers(gmodel)

    if enable_norm:
        # Compute normalization score
        norm = {
            gate_dict["name"]: 0.0
            for gate_dict in gates_dict
        }
        def compute_norm(n, level, parser):
            # Handling input gates
            for e in parser._graph.in_edges(n, data=True):
                src, dst, level_change, inbound_idx = e[0], e[1], e[2]["level_change"], e[2]["inbound_idx"]
                if level_change[1] != level:
                    continue
                elif (src, level_change[0]) in gate_mapping:
                    norm[gate_mapping[(src, level_change[0])][0]["config"]["name"]] +=\
                        compute_act(layers, n, batch_size, is_input_gate=True)
            if (n, level) in gate_mapping: # Handling layers without gates
                norm[gate_mapping[(n, level)][0]["config"]["name"]] += compute_act(layers, n, batch_size)
        for _, p in parsers.items():
            p.traverse(node_callbacks=[lambda n, level: compute_norm(n, level, p)])

        for key, val in norm.items():
            norm[key] = float(max(val, 1.0)) / 1e6
    else:
        norm = {
            gate_dict["name"]: 1.0
            for gate_dict in gates_dict
        }

    return gmodel, PruningCallback(gmodel, norm, targets, period=period, target_ratio=target_ratio)
