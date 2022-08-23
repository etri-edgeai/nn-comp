import math
import tensorflow as tf

from numba import jit
import numpy as np
from tqdm import tqdm

from prep import add_augmentation, change_dtype
from group_fisher import is_simple_group, flatten
from nncompress.backend.tensorflow_.transformation.pruning_parser import has_intersection

@jit
def find_min(score, gates_info, n_channels_group, n_removed_group, ngates):
    idx = 0
    min_score = score[0]
    local_base = 0
    for gidx in range(len(gates_info)):

        if n_channels_group[gidx] - n_removed_group[gidx] < 2.0: # min channels.
            local_base += gates_info[gidx]
            continue

        for lidx in range(gates_info[gidx]):
            if score[local_base + lidx] < -100.0:
                continue

            if min_score > score[local_base+lidx]:
                min_score = score[local_base+lidx]
                idx = local_base + lidx

        local_base += gates_info[gidx]

    return idx

# get inchannel dependency
def get_vindices(lidx, group_struct):
    vindices = set([lidx])
    last = len(vindices)
    initial_run = True
    visited = set()
    while initial_run or (not last == len(vindices)):
        initial_run = False
        last = len(vindices)

        for key, val in group_struct[0].items():
            if type(key) == str and key not in visited:
                val = sorted(val, key=lambda x:x[0])
                for vidx in list(vindices):
                    found = False
                    for v in val:
                        if v[0] <= vidx and vidx < v[1]:
                            found = True
                            relative = vidx - v[0]
                            break
                    if found:
                        visited.add(key)
                        for v in val:
                            vindices.add(v[0]+relative)
    return vindices


def apply_curl(train_data_generator, teacher, gated_model, groups, l2g, parser, target_ratio, save_dir, save_prefix, save_steps):

    print("collecting...")
    data = []
    ty = []
    record_id = 1
    for X, y in train_data_generator:
        if record_id > 1:
            break
        data.append(X)
        ty.append(teacher(X, training=False)[0]) # the first output of teacher is the output logit.
        record_id += 1

    print("scoring...")
    n_channels = 0
    n_channels_group = [
        0 for _ in range(len(groups))
    ]
    gates_weights = {}
    gates_info = []
    for gidx, (g, _) in enumerate(groups):
        if is_simple_group(g):
            gates_info.append(gated_model.get_layer(l2g[g[0]]).gates.numpy().shape[0])
            n_channels_group[gidx] = gates_info[-1]
            n_channels += gates_info[-1]
            for g_ in g:
                gate = gated_model.get_layer(l2g[g_])
                gates_weights[l2g[g_]] = gate.gates.numpy()
                break
        else:
            g_ = flatten(g)
            _, group_struct = parser.get_group_topology(g_)

            assert len(group_struct) == 1
            items = []
            max_ = 0
            for key, val in group_struct[0].items():
                if type(key) == str:
                    val = sorted(val, key=lambda x:x[0])
                    items.append((key, val))
                    for v in val:
                        if v[1] > max_:
                            max_ = v[1]
            mask = np.zeros((max_,))
            for key, val in items:
                gate = gated_model.get_layer(l2g[key])
                for v in val:
                    mask[v[0]:v[1]] += gate.gates.numpy()
            gates_ = (mask >= 0.5).astype(np.float32)
            assert np.sum(gates_) == gates_.shape[-1]
            gates_weights[gidx] = gates_

            gates_info.append(mask.shape[0])
            n_channels_group[gidx] = gates_info[-1]
            n_channels += gates_info[-1]
        
    score = [0.0 for _ in range(n_channels)]

    import json
    with open("score_debug.json", "w") as f:
        json.dump(score, f)

    local_base = 0
    for gidx, (g, _) in enumerate(tqdm(groups, ncols=80)):

        if is_simple_group(g):
            gate = gated_model.get_layer(l2g[g[0]]).gates.numpy()
        else:
            g_ = flatten(g)
            _, group_struct = parser.get_group_topology(g_)

            items = []
            max_ = 0
            for key, val in group_struct[0].items():
                if type(key) == str:
                    val = sorted(val, key=lambda x:x[0])
                    items.append((key, val))
                    for v in val:
                        if v[1] > max_:
                            max_ = v[1]
            mask = np.zeros((max_,))
            for key, val in items:
                gate = gated_model.get_layer(l2g[key])
                for v in val:
                    mask[v[0]:v[1]] += gate.gates.numpy()
            gate = (mask >= 0.5).astype(np.float32)

        for lidx in range(gate.shape[0]):

            gate[lidx] = 0.0
            if is_simple_group(g):
                for layer in g:
                    gate_ = gated_model.get_layer(l2g[layer])
                    gate_.gates.assign(gate)
            else:
                g_ = flatten(g)
                _, group_struct = parser.get_group_topology(g_)

                vindices = get_vindices(lidx, group_struct) # expand to inner channels depends on lidx
                for vidx in vindices:
                    for key, val in group_struct[0].items():
                        if type(key) == str:
                            val = sorted(val, key=lambda x:x[0])
                            gate_ = gated_model.get_layer(l2g[key])
                            for v in val:
                                if v[0] <= vidx and vidx < v[1]:
                                    gates_ = gate_.gates.numpy()
                                    gates_[vidx - v[0]] = gate[vidx]
                                    gate_.gates.assign(gates_)
                                    break # it is okay to see once.

            sum_ = 0.0
            for X, ty_output in zip(data, ty):
                student_logits = gated_model(X, training=False)
                sum_ += tf.math.reduce_mean(tf.keras.losses.kl_divergence(student_logits[0], ty_output))
            score[local_base + lidx] = float(sum_)

            gate[lidx] = 1.0
            if is_simple_group(g):
                for layer in g:
                    gate_ = gated_model.get_layer(l2g[layer])
                    gate_.gates.assign(gate)

            else:
                g_ = flatten(g)
                _, group_struct = parser.get_group_topology(g_)

                vindices = get_vindices(lidx, group_struct) # expand to inner channels depends on lidx
                for vidx in vindices:
                    for key, val in group_struct[0].items():
                        if type(key) == str:
                            val = sorted(val, key=lambda x:x[0])
                            gate_ = gated_model.get_layer(l2g[key])
                            for v in val:
                                if v[0] <= vidx and vidx < v[1]:
                                    gates_ = gate_.gates.numpy()
                                    gates_[vidx - v[0]] = gate[vidx]
                                    gate_.gates.assign(gates_)
                                    break

        local_base += gate.shape[0]

    print("pruning...")
    # pruning
    n_removed = 0
    n_removed_group = [
        0 for _ in range(len(groups))
    ]

    total_ = math.ceil(n_channels * target_ratio)
    with tqdm(total=total_, ncols=80) as pbar:
        while float(n_removed) / n_channels < target_ratio:
            if save_steps != -1 and n_removed % save_steps == 0:
                for key in gates_weights:
                    if type(key) == str:
                        layer = gated_model.get_layer(key)
                        layer.gates.assign(gates_weights[key])
                    else:
                        g, _ = groups[key]
                        g_ = flatten(g)
                        _, group_struct = parser.get_group_topology(g_)

                        assert len(group_struct) == 1
                        for key_, val in group_struct[0].items():
                            if type(key_) == str:
                                val = sorted(val, key=lambda x:x[0]) 
                                gate_ = gated_model.get_layer(l2g[key_])
                                gates_ = gate_.gates.numpy()
                                for vidx, v in enumerate(val):
                                    gates_[:] = gates_weights[key][v[0]:v[1]]
                                    assert np.sum(gates_) > 0.0
                                    break
                                gate_.gates.assign(gates_)

                cmodel = parser.cut(gated_model)
                print(cmodel.count_params())
                tf.keras.models.save_model(cmodel, save_dir+"/"+save_prefix+"_"+str(n_removed)+".h5")
                tf.keras.models.save_model(gated_model, save_dir+"/"+save_prefix+"_"+str(n_removed)+"_gated_model.h5")

            val = find_min(score, gates_info, n_channels_group, n_removed_group, len(gates_info))
            local_base = 0
            min_gidx = -1
            min_lidx = -1
            hit_base = -1
            for gidx, len_ in enumerate(gates_info):
                if val - local_base <= len_-1: # hit
                    min_gidx = gidx
                    min_lidx = val - local_base
                    hit_base = local_base
                    break
                else:
                    local_base += len_

            #print(min_gidx, min_lidx, val, local_base, score[val])
            assert min_gidx != -1
            min_group, _ = groups[min_gidx]
            if is_simple_group(min_group):
                vindices = set([min_lidx])
                for min_layer in min_group:
                    gates_weights[l2g[min_layer]][min_lidx] = 0.0
            else:
                g_ = flatten(min_group)
                _, group_struct = parser.get_group_topology(g_)

                vindices = get_vindices(min_lidx, group_struct)
                for vidx in vindices:
                    gates_weights[min_gidx][vidx] = 0.0

            for vidx in vindices:
                score[hit_base + vidx] = -999.0

            # zero filter test
            if is_simple_group(min_group):
                if np.sum(gates_weights[l2g[min_group[0]]]) <= 5.0: # avoid zero filter case
                    for min_layer in min_group:
                        gates_weights[l2g[min_layer]][min_lidx] = 1.0
                    continue
            else:
                g, _ = groups[min_gidx]
                g_ = flatten(g)
                _, group_struct = parser.get_group_topology(g_)

                assert len(group_struct) == 1
                min_filter_violation = False
                for key_, val in group_struct[0].items():
                    if type(key_) == str:
                        val = sorted(val, key=lambda x:x[0]) 
                        for vidx, v in enumerate(val):
                            if np.sum(gates_weights[key][v[0]:v[1]]) <= 5.0:
                                min_filter_violation = True
                                break

                if min_filter_violation: # avoid zero filter case
                    for vidx in vindices:
                        gates_weights[min_gidx][vidx] = 1.0
                    continue

            n_removed += len(vindices)
            n_removed_group[gidx] += len(vindices)
            pbar.update(1)

    for key in gates_weights:
        if type(key) == str:
            layer = gated_model.get_layer(key)
            layer.gates.assign(gates_weights[key])
        else:
            g, _ = groups[key]
            g_ = flatten(g)
            _, group_struct = parser.get_group_topology(g_)

            for key_, val in group_struct[0].items():
                if type(key_) == str:
                    val = sorted(val, key=lambda x:x[0]) 
                    gate_ = gated_model.get_layer(l2g[key_])
                    gates_ = gate_.gates.numpy()
                    for v in val:
                        gates_[:] = gates_weights[key][v[0]:v[1]]
                        break
                    gate_.gates.assign(gates_)
