
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

from train import train_step

def find_all(model, Target):
    ret = []
    for layer in model.layers:
        if hasattr(layer, "layers"):
            ret += find_all(layer, Target)
        else:
            if layer.__class__ == Target:
                ret.append(layer)
    return ret

def get_num_all_channels(groups):
    sum_ = 0
    for gidx, group in enumerate(groups):
        g = group[0]
        sum_ += g.ngates
    return sum_

def compute_sparsity(groups):
    sum_ = 0
    alive = 0
    for gidx, group in enumerate(groups):
        g = group[0]
        sum_ += g.ngates
        alive += np.sum(g.gates.numpy())
    return 1.0 - alive / sum_

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
                 l2g = None,
                 num_remove=1,
                 fully_random=False,
                 callback_after_deletion=None,
                 logging=False):
        super(PruningCallback, self).__init__()
        self.norm = norm
        self.targets = targets
        self.period = period
        self.target_ratio = target_ratio
        self.continue_pruning = True
        self.gate_groups = gate_groups
        self._iter = 0
        self._num_removed = 0
        self.l2g = l2g
        self.num_remove = num_remove
        self.fully_random = fully_random
        self.callback_after_deletion = callback_after_deletion
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
            #groups.reverse()
            for gidx, group in enumerate(groups):

                # compute grad based si
                num_batches = len(group[0].grad_holder)
                sum_ = 0
                cscore = None
                for bidx in range(num_batches):
                    grad = 0
                    for lidx, layer in enumerate(group):
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

                self._num_removed += 1

                if self.callback_after_deletion is not None:
                   self.callback_after_deletion(self._num_removed)

                if compute_sparsity(groups) >= self.target_ratio:
                    break

            self.continue_pruning = compute_sparsity(groups) < self.target_ratio
            for layer in self.targets:
                layer.grad_holder = []
                if not self.continue_pruning:
                    layer.collecting = False

            if not self.continue_pruning:
                print("SPARSITY:", compute_sparsity(groups))

            # for fit
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


def add_gates(model, custom_objects=None, avoid=None):

    model = unfold(model, custom_objects)
    parser = PruningNNParser(model, custom_objects=custom_objects, gate_class=SimplePruningGate)
    parser.parse()

    gmodel, gate_mapping = parser.inject(avoid=avoid, with_mapping=True, with_splits=True)
    #tf.keras.utils.plot_model(model, "ddd.png")
    #tf.keras.utils.plot_model(gmodel, "ggg.png")

    v = parser.traverse()
    torder = {
        name:idx
        for idx, (name, _) in enumerate(v)
    }

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

    return gmodel, model, l2g, ordered_groups, torder, parser, gate_mapping


def compute_positions(model, ordered_groups, torder, parser, position_mode, num_blocks):

    convs = [
        layer.name for layer in model.layers if "Conv2D" in layer.__class__.__name__
    ]

    blocks = [[]]
    current_id = 0
    for i, (g, idx) in enumerate(ordered_groups):

        #des = parser.first_common_descendant(list(g), joints)
        des = parser.first_common_descendant(list(g), convs)
        blocks[current_id].append((g, des))
        if len(blocks[current_id]) >= len(ordered_groups) // num_blocks and current_id != num_blocks-1:
            current_id += 1
            blocks.append([])

    # compute positions
    convs = [
        layer.name for layer in model.layers if "Conv2D" in layer.__class__.__name__
    ]
    all_ = [
        layer.name for layer in model.layers
    ]

    all_acts_ = [
        layer.name for layer in model.layers if layer.__class__.__name__ in ["Activation", "ReLU", "Softmax"]
    ]

    if position_mode == 0:
        positions = all_acts_

    elif position_mode == 4: # 1x1 conv
        positions = []
        for c in convs:
            if gmodel.get_layer(c).kernel_size == (1,1):
                positions.append(c)
    elif position_mode == 1: # joints
        positions = []
        for b in blocks:
            act = parser.get_first_activation(b[-1][1])
            if act not in positions:
                positions.append(act)

    elif position_mode == 2: # random
        positions = [
            all_acts_[int(random.random() * len(all_acts_))] for i in range(len(blocks))
        ]

    elif position_mode == 3: # cut
        positions  = []
        for b in blocks:
            g = b[-1][0]
            des = parser.first_common_descendant(list(g), all_acts_, False)

            if des not in positions:
                positions.append(des)
            """
            des_g = None
            for g_, idx in ordered_groups:
                if des in g_:
                    des_g = g_
                    break

            if des_g is not None:
                for l in des_g:
                    if l not in positions:
                        positions.append(l)
            else:
                if des not in positions:
                    positions.append(des) # maybe the last transforming layer.
            """

    elif position_mode == 5: # cut
        affecting_layers = parser.get_affecting_layers()

        cnt = {}
        for layer_ in affecting_layers:
            for a in affecting_layers[layer_]:
                if a[0] not in cnt:
                    cnt[a[0]] = 0
                cnt[a[0]] += 1

        node_list = [
            (key, value) for key, value in  cnt.items()
        ]
        node_list = sorted(node_list, key=lambda x: x[1])

        k = 5
        positions = [
            parser.get_first_activation(n) for n, _ in node_list[-1*k:]
        ]

    elif position_mode == 6: # cut
        affecting_layers = parser.get_affecting_layers()

        cnt = {
            layer_[0]:0 for layer_ in affecting_layers
        }
        for layer_ in affecting_layers:
            cnt[layer_[0]] = len(affecting_layers[layer_])

        node_list = [
            (key, value) for key, value in  cnt.items()
        ]
        node_list = sorted(node_list, key=lambda x: x[1])

        k = 5
        positions = [
            parser.get_first_activation(n) for n, _ in node_list[-1*k:]
        ]

    elif position_mode == 7:

        cons_ = [
            (c, torder[c]) for c in convs
        ]

        node_list = sorted(cons_, key=lambda x: x[1])

        k = 5
        positions = [
            n for n, _ in node_list[-1*k:]
        ]

    elif position_mode == 8:
        alll_ = [
            (c, torder[c]) for c in all_ if gmodel.get_layer(c).__class__.__name__ in ["Activation", "ReLU", "Softmax"]
        ]
        node_list = sorted(alll_, key=lambda x: x[1])

        k = 5
        positions = [
            n for n, _ in node_list[-1*k:]
        ]
        print(node_list)

    elif position_mode == 9:
        alll_ = [
            (c, torder[c]) for c in all_
        ]
        node_list = sorted(alll_, key=lambda x: x[1])

        k = 5
        positions = [
            n for n, _ in node_list[-1*k:]
        ]
        print(node_list)

    elif position_mode == 10: # cut
        positions  = []
        for b in blocks:
            g = b[-1][0]
            des = parser.first_common_descendant(list(g), convs, False)

            des_g = None
            for g_, idx in ordered_groups:
                if des in g_:
                    des_g = g_
                    break

            if des_g is not None:
                des = parser.first_common_descendant(list(des_g), all_acts_, False)
                if des not in positions:
                    positions.append(des)
            else:
                act = parser.get_first_activation(des)
                if act is None:
                    act = des
                if act not in positions:
                    positions.append(act) # maybe the last transforming layer.

    return positions

def make_group_fisher(model,
                      model_handler,
                      batch_size,
                      custom_objects=None,
                      avoid=None,
                      period=25,
                      target_ratio=0.5,
                      enable_norm=True,
                      num_remove=1,
                      num_blocks=3,
                      position_mode=0,
                      fully_random=False,
                      save_steps=-1,
                      save_prefix=None,
                      save_dir=None,
                      logging=False):

    gmodel, model, l2g, ordered_groups, torder, parser, gate_mapping = add_gates(model, custom_objects, avoid)
    targets = find_all(gmodel, SimplePruningGate)

    groups = []
    for g, _ in ordered_groups:
        gate_group = []
        for l in g:
            gate_group.append(gmodel.get_layer(l2g[l]))
        groups.append(gate_group)

    # Compute normalization score
    if enable_norm:
        norm = {
            t.name: 0.0
            for t in targets
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
            t.name: 1.0
            for t in targets
        }

    #joints = parser.get_joints()
    positions = compute_positions(model, ordered_groups, torder, parser, position_mode, num_blocks)

    # ready for collecting
    for layer in gmodel.layers:
        if layer.__class__ == SimplePruningGate:
            layer.grad_holder = []
            layer.collecting = False

    def callback_after_deletion_(num_removed):
        if num_removed % save_steps == 0:
            assert save_dir is not None
            assert save_prefix is not None
            cmodel = parser.cut(gmodel)
            tf.keras.models.save_model(cmodel, save_dir+"/"+save_prefix+"_"+str(num_removed)+".h5")

    if save_steps == -1:
        cbk = None
    else:
        cbk = callback_after_deletion_

    return gmodel, model, parser, positions, PruningCallback(
        norm,
        targets,
        gate_groups=groups,
        period=period,
        target_ratio=target_ratio,
        l2g=l2g,
        num_remove=num_remove,
        fully_random=fully_random,
        callback_after_deletion=cbk,
        logging=logging)


def prune_step(X, model, teacher_logits, y, pc):

    for layer in model.layers:
        if layer.__class__ == SimplePruningGate:
            layer.trainable = True
            layer.collecting = True

    tape, loss = train_step(X, model, teacher_logits, y)
    _ = tape.gradient(loss, model.trainable_variables)
    pruned = pc.on_train_batch_end(None)

    for layer in model.layers:
        if layer.__class__ == SimplePruningGate:
            layer.trainable = False
            layer.collecting = False
