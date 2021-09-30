
import types
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser
from nncompress.backend.tensorflow_.transformation import parse, inject, cut
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

def compute_sparsity(groups, target_gidx=-1):
    sum_ = 0
    alive = 0
    for gidx, group in enumerate(groups):
        if target_gidx != -1 and target_gidx != gidx:
            continue
        g = group[0]
        sum_ += g.ngates
        alive += np.sum(g.gates.numpy())
    return 1.0 - alive / sum_

class PruningCallback(keras.callbacks.Callback):

    def __init__(self,
                 model,
                 norm,
                 targets,
                 gate_groups=None,
                 target_ratio=0.5,
                 period=10,
                 target_gidx=-1,
                 target_idx=-1,
                 prefix="prefix",
                 validate=None,
                 parsers=None):
        if not hasattr(model, "grad_holder"):
            raise ValueError('make_group_fisher calling first.')
        self.model = model
        self.norm = norm
        self.targets = targets
        self.period = period
        self.target_ratio = target_ratio
        self.continue_pruning = True
        self.gate_groups = gate_groups
        self.target_idx = target_idx
        self.target_gidx = target_gidx
        self.prefix = prefix
        self._iter = 0
        self.validate = validate
        self.parsers = parsers

    def on_train_batch_end(self, batch, logs=None):
        self._iter += 1
        if self._iter % self.period == 0 and self.continue_pruning:
            # Pruning upon self.grad_holder
            #gates = find_all(self.model, SimplePruningGate)

            if self.gate_groups is not None:
                groups = self.gate_groups
            else:
                groups = [
                    [gate] for gate in self.targets
                ]

            min_val = -1
            min_idx = (-1, -1)
            for gidx, group in enumerate(groups):
                if self.target_gidx != -1 and gidx != self.target_gidx: # only consider target_gidx
                    continue
                cscore = None
                for lidx, layer in enumerate(group):
                    if self.target_idx != -1 and lidx != self.target_idx:
                        continue
                    gates_ = layer.gates.numpy()
                    if np.sum(gates_) == 1.0: # min channels.
                        continue
                    norm_ = self.norm[layer.name]
                    sum_ = 0
                    for grad in layer.grad_holder:
                        sum_ += grad
                    if cscore is None:
                        cscore = sum_ / (len(layer.grad_holder) * norm_)
                    else:
                        cscore += sum_ / (len(layer.grad_holder) * norm_)

                for i in range(cscore.shape[0]):
                    if (min_idx[0] == -1 or min_val > cscore[i]) and gates_[i] == 1.0:
                        min_val = cscore[i]
                        min_idx = (gidx, i)
                assert min_idx[0] != -1

            min_group = groups[min_idx[0]]
            for min_layer in min_group:
                gates_ = min_layer.gates.numpy()
                gates_[min_idx[1]] = 0.0
                min_layer.gates.assign(gates_)
                print(min_idx[1], min_layer.name, np.sum(min_layer.gates.numpy()) / min_layer.ngates, compute_sparsity(groups, self.target_gidx))

            collecting = compute_sparsity(groups, self.target_gidx) < self.target_ratio
            self.continue_pruning = collecting
            for layer in self.targets:
                layer.grad_holder = []
                layer.collecting = collecting

            if not self.continue_pruning:
                self.model.save(self.prefix+".h5")
                if self.validate is not None:
                    cmodel = cut(self.parsers, self.model)
                    print(cmodel.summary())
                    self.validate(cmodel)


def compute_act(layers, layer_name, batch_size, is_input_gate=False):
    layer = layers[layer_name]
    if layer.__class__.__name__ == "Conv2D" or\
        (layer.__class__.__name__ == "DepthwiseConv2D" and not is_input_gate):
        # DepthwiseConv2D only contributes for the gate associated with its outputs.
        assert layer.groups == 1
        return batch_size * np.prod(layer.get_weights()[0].shape[0:2])
    else:
        return 0.0

def make_group_fisher(model,
                      batch_size,
                      custom_objects=None,
                      avoid=None,
                      period=25,
                      target_ratio=0.5,
                      enable_norm=True,
                      with_splits=False,
                      target_gidx=-1,
                      target_idx=-1,
                      prefix="",
                      validate=None,
                      exploit_topology=False):
    """

    """
    parsers = parse(model, PruningNNParser, custom_objects=custom_objects, gate_class=SimplePruningGate)
    gmodel, gate_mapping = inject(parsers, avoid=avoid, with_splits=with_splits)
    keras.utils.plot_model(gmodel, to_file="gg.png", expand_nested=True)

    targets = find_all(gmodel, SimplePruningGate)
    targets_ = set([target.name+"/gates:0" for target in targets])
    gmodel.grad_holder = []

    gmodel_dict = json.loads(gmodel.to_json())
    gates_dict = find_all_dict(gmodel_dict, "SimplePruningGate")
    layers = get_layers(gmodel)

    if with_splits:
        l2g = {}
        for layer, flow in gate_mapping:
            l2g[layer] = gate_mapping[(layer, flow)][0]["config"]["name"]

        groups = []
        for _, p in parsers.items():
            groups_ = p.get_sharing_groups()
            for g in groups_:
                gate_group = set()
                for layer in g:
                    if layer in l2g:
                        gate_group.add(layers[l2g[layer]])
                if len(gate_group) == 0:
                    continue
                gate_group = sorted(list(gate_group), key=lambda x: x.name) # sorted by alphabetical order
                groups.append(gate_group)
    else:
        groups = [
            [t] for t in targets
        ]

    if target_gidx != -1:
        for gidx, g in enumerate(groups):
            if len(g) > 1:
                target_gidx = gidx
                break
        print("TARGET: ", [layer.name for layer in groups[target_gidx]])

    if exploit_topology:
         
        pass 

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

    return gmodel, PruningCallback(gmodel, norm, targets, gate_groups=groups, period=period, target_ratio=target_ratio, target_gidx=target_gidx, target_idx=target_idx, prefix=prefix, validate=validate, parsers=parsers)
