
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

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser
from nncompress.backend.tensorflow_ import SimplePruningGate, DifferentiableGate

from nncompress.backend.tensorflow_.transformation import parse, inject, cut, unfold


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
        sum_ += grad * grad
    return tf.reduce_sum(sum_, axis=0)

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
                 norm,
                 targets,
                 gate_groups=None,
                 target_ratio=0.5,
                 period=10,
                 target_gidx=-1,
                 target_idx=-1,
                 l2g = None,
                 num_remove=1,
                 fully_random=False,
                 logging=False):
        super(PruningCallback, self).__init__()
        self.norm = norm
        self.targets = targets
        self.period = period
        self.target_ratio = target_ratio
        self.continue_pruning = True
        self.gate_groups = gate_groups
        self.target_idx = target_idx
        self.target_gidx = target_gidx
        self._iter = 0
        self.l2g = l2g
        self.num_remove = num_remove
        self.fully_random = fully_random
        self.logging = logging
        if self.logging:
            self.logs = []
        else:
            self.logs = None

    def on_train_batch_end(self, batch, logs=None):
        self._iter += 1
        if self._iter % self.period == 0 and self.continue_pruning:

            if self.gate_groups is not None:
                groups = self.gate_groups
            else:
                groups = [
                    [gate] for gate in self.targets
                ]

            cscore_ = {}
            frozen = {}
            #groups.reverse()
            for gidx, group in enumerate(groups):
                if self.target_gidx != -1 and gidx != self.target_gidx: # only consider target_gidx
                    continue

                is_frozen = False
                for lidx, layer in enumerate(group):
                    if layer.freeze:
                        is_frozen = True
                        break
                if is_frozen:
                    frozen[gidx] = True
                    continue

                # compute grad based si
                num_batches = len(group[0].grad_holder)
                sum_ = 0
                cscore = None
                for bidx in range(num_batches):
                    grad = 0
                    for lidx, layer in enumerate(group):
                        if self.target_idx != -1 and lidx != self.target_idx:
                            continue

                        gates_ = layer.gates.numpy()
                        if np.sum(gates_) < 2.0: # min channels.
                            break

                        grad += layer.grad_holder[bidx]

                    if type(grad) == int and grad == 0:
                        continue

                    grad = pow(grad, 2)
                    sum_ += tf.reduce_sum(grad, axis=0)
                    cscore = 0.0

                # compute normalization
                norm_ = 0
                for lidx, layer in enumerate(group):
                    if self.target_idx != -1 and lidx != self.target_idx:
                        continue

                    gates_ = layer.gates.numpy()
                    if np.sum(gates_) < 2.0: # min channels.
                        break

                    norm_ += self.norm[layer.name]

                if cscore is not None: # To handle cscore is undefined.
                    #cscore = sum_
                    cscore = sum_ / norm_
                cscore_[gidx] = cscore

                if self.logs is not None and len(self.logs) == 0:
                    self.logs.append((group, cscore))
                    print([g.name for g in group])
                    for ii in range(cscore.shape[0]):
                        print(ii, float(cscore[ii]))
                    import sys
                    sys.exit(0)

            for _ in range(self.num_remove):
                min_val = -1
                min_idx = (-1, -1)
                for gidx, group in enumerate(groups):
                    if self.target_gidx != -1 and gidx != self.target_gidx: # only consider target_gidx
                        continue

                    if gidx in frozen:
                        continue

                    cscore = cscore_[gidx]
                    gates_ = group[0].gates.numpy()
                    if np.sum(gates_) < 2.0:
                        continue
                    if cscore is not None:
                        if self.fully_random:
                            cscore = np.random.rand(*tuple(cscore.shape))
                        else:
                            min_val, min_idx = find_min(cscore, gates_, min_val, min_idx, gidx, cscore.shape[0])

                min_group = groups[min_idx[0]]
                for min_layer in min_group:
                    gates_ = min_layer.gates.numpy()
                    gates_[min_idx[1]] = 0.0
                    min_layer.gates.assign(gates_)

                if compute_sparsity(groups, self.target_gidx) >= self.target_ratio:
                    break

            print("SPARSITY:", compute_sparsity(groups, self.target_gidx))
            collecting = compute_sparsity(groups, self.target_gidx) < self.target_ratio
            self.continue_pruning = collecting
            for layer in self.targets:
                layer.grad_holder = []
                layer.collecting = collecting

            if not self.continue_pruning and hasattr(self, "model") and hasattr(self.model, "stop_training"):
                self.model.stop_training = True
            return True
        else:
            return False

def compute_act(layer, batch_size, is_input_gate=False):

    if is_input_gate:
        if layer.__class__.__name__ == "Conv2D":
            w = layer.get_weights()[0].shape
            return batch_size * np.prod(list(w[0:2])+[w[2]])
        elif layer.__class__.__name__ == "SeparableConv2D":
            w1 = layer.get_weights()[0].shape
            w2 = layer.get_weights()[1].shape
            print(w1, w2)
            xxxxxxxxxxxx # later check.
            return batch_size * np.prod(list(w2[0:2])+[w2[2]])
        else:
            return 0.0
    else: 
        if layer.__class__.__name__ == "Conv2D" or\
            (layer.__class__.__name__ == "DepthwiseConv2D" and not is_input_gate):
            # DepthwiseConv2D only contributes for the gate associated with its outputs.
            assert layer.groups == 1
            return batch_size * np.prod(layer.get_weights()[0].shape[0:3])
        elif layer.__class__.__name__ == "SeparableConv2D":
            return batch_size * np.prod(layer.get_weights()[0].shape[0:3]) + batch_size * np.prod(layer.get_weights()[1].shape[0:3])
        else:
            return 0.0

def make_group_fisher(model,
                      model_handler,
                      batch_size,
                      custom_objects=None,
                      avoid=None,
                      period=25,
                      target_ratio=0.5,
                      enable_norm=True,
                      target_gidx=-1,
                      target_idx=-1,
                      num_remove=1,
                      num_blocks=3,
                      fully_random=False,
                      logging=False):
    """

    """
    model = unfold(model, custom_objects)
    parser = PruningNNParser(model, custom_objects=custom_objects, gate_class=SimplePruningGate)
    parser.parse()

    gmodel, gate_mapping = parser.inject(avoid=avoid, with_mapping=True, with_splits=True)
    tf.keras.utils.plot_model(model, "ddd.png")
    tf.keras.utils.plot_model(gmodel, "ggg.png")

    targets = find_all(gmodel, SimplePruningGate)
    targets_ = set([target.name+"/gates:0" for target in targets])
    gmodel.grad_holder = []

    v = parser.traverse()
    torder = {
        name:idx
        for idx, (name, _) in enumerate(v)
    }

    gmodel_dict = json.loads(gmodel.to_json())
    gates_dict = find_all_dict(gmodel_dict, "SimplePruningGate")

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

    groups_ = parser.get_sharing_groups()
    ordered_groups = []
    for g in groups_:
        ###
        g_ = sorted(list(g), key=lambda x: torder[x])
        ordered_groups.append((g, torder[g_[0]]))
        ###
        #ordered_groups.append((g, torder[g_[0]]))
    ordered_groups = sorted(ordered_groups, key=lambda x: x[1])[:-1] # remove last

    groups = []
    for g, _ in ordered_groups:
        gate_group = []
        for l in g:
            gate_group.append(gmodel.get_layer(l2g[l]))
        groups.append(gate_group)

    if target_gidx != -1:
        gidx_ = 0
        for gidx, g in enumerate(groups):
            if len(g) > 1:
                if gidx_ == target_gidx:
                    target_gidx = gidx
                    break
                gidx_ += 1
        print("TARGET: ", [layer.name for layer in groups[target_gidx]], target_gidx)


    model_handler.initial_lr = 0.001

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
                    if not gate_mapping[(src, level_change[0])][0]["class_name"] == "Concatenate":
                        norm[gate_mapping[(src, level_change[0])][0]["config"]["name"]] +=\
                            compute_act(gmodel.get_layer(n), batch_size, is_input_gate=True)
            if (n, level) in gate_mapping: # Handling layers without gates
                if not gate_mapping[(n, level)][0]["class_name"] == "Concatenate":
                    norm[gate_mapping[(n, level)][0]["config"]["name"]] += compute_act(gmodel.get_layer(n), batch_size)
        parser.traverse(node_callbacks=[lambda n, level: compute_norm(n, level, parser)])

        for key, val in norm.items():
            norm[key] = float(max(val, 1.0)) / 1e6
    else:
        norm = {
            gate_dict["name"]: 1.0
            for gate_dict in gates_dict
        }

    joints = parser.get_joints()

    convs = [
        layer.name for layer in model.layers if "Conv2D" in layer.__class__.__name__
    ]

    blocks = [[]]
    current_id = 0
    for i, (g, idx) in enumerate(ordered_groups):

        des = parser.first_common_descendant(list(g), joints)

        blocks[current_id].append((g, des))
        if len(blocks[current_id]) >= len(ordered_groups) // num_blocks and current_id != num_blocks-1:
            current_id += 1
            blocks.append([])

    return gmodel, model, blocks, ordered_groups, parser, PruningCallback(
        norm,
        targets,
        gate_groups=groups,
        period=period,
        target_ratio=target_ratio,
        target_gidx=target_gidx,
        target_idx=target_idx,
        l2g=l2g,
        num_remove=num_remove,
        fully_random=fully_random,
        logging=logging)
