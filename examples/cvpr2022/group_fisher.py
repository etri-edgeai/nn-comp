
import types
import json
from collections import OrderedDict

import tensorflow as tf
from tensorflow import keras
import numpy as np
from numba import njit
from numpy import dot
from numpy.linalg import norm as npnorm

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

def compute_grad_score(gate_layer):
    sum_ = 0
    for grad in gate_layer.grad_holder:
        sum_ += grad
    return sum_ / len(gate_layer.grad_holder)

def compute_vertical_consensus(affected_layers, l2g, layers):
    from lookahead import build_db_matrix
    vcon = {}
    for l, affected in affected_layers.items():
        gate_score_l = compute_grad_score(layers[l2g[l]])
        sim_sum = 0.0
        for a in affected:
            layer_a = layers[a]
            gate_score_a = compute_grad_score(layers[l2g[a]])

            weight_tensor = layer_a.get_weights()[0]
            W = np.sum(np.abs(weight_tensor), axis=(0,1))
            iW = np.linalg.pinv(W)

            gate_score_a_inbound = np.matmul(gate_score_a, iW)
            sim_sum += dot(gate_score_l, gate_score_a_inbound)/(npnorm(gate_score_l)*npnorm(gate_score_a_inbound))
        sim_ = sim_sum / len(affected)
        vcon[l2g[l]] = max(sim_, 0.0)

    max_val = -1
    min_val = -1
    for key, val in vcon.items():
        if max_val < val:
            max_val = val
        if min_val == -1 or min_val > val:
            min_val = val
    for key, val in vcon.items():
        vcon[key] = (val - min_val) / (max_val-min_val)

    return vcon

@njit
def find_min(cscore, gates, min_val, min_idx, gidx, ncol):
    for i in range(ncol):
        if (min_idx[0] == -1 or min_val > cscore[i]) and gates[i] == 1.0:
            min_val = cscore[i]
            min_idx = (gidx, i)
    assert min_idx[0] != -1
    return min_val, min_idx

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
                 parsers=None,
                 affected_layers=None,
                 layers = None,
                 l2g = None,
                 num_remove=1):
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
        self.affected_layers = affected_layers
        self.layers = layers
        self.l2g = l2g
        self.num_remove = num_remove


    def on_train_batch_end(self, batch, logs=None):
        self._iter += 1
        if self._iter % self.period == 0 and self.continue_pruning:
            # Pruning upon self.grad_holder
            if self.affected_layers is not None:
                vcon = compute_vertical_consensus(self.affected_layers, self.l2g, self.layers)
            else:
                vcon = {}

            if self.gate_groups is not None:
                groups = self.gate_groups
            else:
                groups = [
                    [gate] for gate in self.targets
                ]

            cscore_ = {}
            for gidx, group in enumerate(groups):
                if self.target_gidx != -1 and gidx != self.target_gidx: # only consider target_gidx
                    continue

                sim_sum = 0
                for lidx, layer in enumerate(group):
                    sim_sum += vcon[layer.name] if layer.name in vcon else 1.0

                cscore = None
                for lidx, layer in enumerate(group):
                    if self.target_idx != -1 and lidx != self.target_idx:
                        continue

                    gates_ = layer.gates.numpy()
                    if np.sum(gates_) < 10.0: # min channels.
                        continue

                    norm_ = self.norm[layer.name]
                    sum_ = 0
                    for grad in layer.grad_holder:
                        sum_ += grad

                    if sim_sum == 0:
                        sim_ = 1.0
                    else:
                        sim_ = vcon[layer.name] / sim_sum if layer.name in vcon else 1.0

                    #sim_ = 1.0
                    if cscore is None:
                        cscore = (sum_ / (len(layer.grad_holder) * norm_)) * sim_
                    else:
                        cscore += (sum_ / (len(layer.grad_holder) * norm_)) * sim_

                cscore_[gidx] = cscore

            for _ in range(self.num_remove):
                min_val = -1
                min_idx = (-1, -1)
                for gidx, group in enumerate(groups):
                    if self.target_gidx != -1 and gidx != self.target_gidx: # only consider target_gidx
                        continue

                    cscore = cscore_[gidx]
                    gates_ = group[0].gates.numpy()
                    if cscore is not None:
                        min_val, min_idx = find_min(cscore, gates_, min_val, min_idx, gidx, cscore.shape[0])

                min_group = groups[min_idx[0]]
                for min_layer in min_group:
                    gates_ = min_layer.gates.numpy()
                    gates_[min_idx[1]] = 0.0
                    min_layer.gates.assign(gates_)

                if compute_sparsity(groups, self.target_gidx) >= self.target_ratio:
                    break

            print("----------")
            print("SPARSITY:", compute_sparsity(groups, self.target_gidx))
            print("SPARSITY:", compute_sparsity(groups, self.target_gidx))
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

    def on_epoch_end(self, epoch, logs):
        cmodel = cut(self.parsers, self.model)
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
                      exploit_topology=False,
                      num_remove=1):
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
        gidx_ = 0
        for gidx, g in enumerate(groups):
            if len(g) > 1:
                if gidx_ == target_gidx:
                    target_gidx = gidx
                    break
                gidx_ += 1
        print("TARGET: ", [layer.name for layer in groups[target_gidx]], target_gidx)

    if exploit_topology:
        convs = set()
        for _, p in parsers.items():
            groups_ = p.get_sharing_groups()
            for g in groups_:
                for layer in g:
                    if layer in l2g:
                        convs.add(layer)

        affected_layers = {}
        for _, p in parsers.items():
            adict = p.get_affecting_layers()
            for layer, level in adict:
                if layer not in convs:
                    continue
                g_ = adict[(layer, level)]
                for l, level, tensor_idx in g_:
                    if l not in convs:
                        continue
                    if l not in affected_layers:
                        affected_layers[l] = set()
                    affected_layers[l].add(layer)
    else:
        affected_layers = None

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

    return gmodel, PruningCallback(
        gmodel,
        norm,
        targets,
        gate_groups=groups,
        period=period,
        target_ratio=target_ratio,
        target_gidx=target_gidx,
        target_idx=target_idx,
        prefix=prefix,
        validate=validate,
        parsers=parsers,
        affected_layers=affected_layers,
        l2g=l2g,
        layers=layers,
        num_remove=num_remove)
