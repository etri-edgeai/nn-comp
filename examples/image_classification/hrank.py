import math
import tensorflow as tf
import numpy as np
import json

from numba import jit
from tqdm import tqdm

from group_fisher import is_simple_group, flatten

def apply_hrank(train_data_generator, teacher, gated_model, groups, l2g, parser, target_ratio, gf_model):

    groups_ = []
    for g, _ in groups:
        if is_simple_group(g):
            groups_.append((g, _))
        else:
            g_ = flatten(g)
            for _g in g_:
                groups_.append((_g, None))
    groups = groups_

    convs = [
        layer for layer in teacher.layers if layer.__class__.__name__ == "Conv2D" and parser.get_first_activation(layer.name) != None and\
            teacher.get_layer(parser.get_first_activation(layer.name)).output.shape[-1] == layer.filters
    ]

    acts = {
        layer.name:teacher.get_layer(parser.get_first_activation(layer.name)) for layer in convs
    }
    
    num_splits = 1
    offset = int(float(len(convs)) / num_splits)
    rest = len(convs) % num_splits
    score = {}
    for i in range(num_splits):
        data_holders = []
        start_idx = offset * i
        end_idx = offset * (i + 1)
        if i == num_splits - 1:
            end_idx += rest
        print(start_idx, end_idx)

        targets_ = convs[start_idx:end_idx]
        model = tf.keras.Model(teacher.input, [acts[layer.name].output for layer in targets_])
        for t in targets_:
            score[t.name] = [ 0 for _ in range(t.filters) ]

        record_id = 1
        for X, y in train_data_generator:
            if record_id > 1:
                break
            data_holders.append(model(X)) # the first output of teacher is the output logit.
            record_id += 1

        for bidx, batch_data in enumerate(data_holders):
            for layer, feat_map in zip(targets_, batch_data):
                for channel_idx in range(int(feat_map.shape[-1])):
                    # compute rank.
                    feat_mat = feat_map[:,:,:,channel_idx]
                    print(channel_idx, len(score[layer.name]), feat_map.shape)
                    score[layer.name][channel_idx] += tf.math.reduce_sum(
                        tf.linalg.matrix_rank(
                            feat_mat, tol=None
                            )
                    )

    # filter meaningless groups
    removal = set()
    for gidx, (g, _) in enumerate(groups):
        for layer in g:
            if layer not in score:
                removal.add(gidx)
                break
    groups_ = []
    for gidx, (g, _) in enumerate(groups):
        if gidx not in removal:
            groups_.append((g, _))
    groups = groups_
        
    n_channels = 0
    gscore = [
        None for _ in groups
    ]
    gates_weights = {}
    for gidx, (g, _) in enumerate(groups):
        channels = gated_model.get_layer(l2g[g[0]]).gates.numpy().shape[0]
        n_channels += channels 
        gscore[gidx] = [
            [_, 0] for _ in range(channels)
        ]
  
        for layer in g:
            for idx in range(channels):
                gscore[gidx][idx][1] += float(score[layer][idx])
            gate = gated_model.get_layer(l2g[layer])
            gates_weights[l2g[layer]] = gate.gates.numpy()

        for idx in range(channels):
            gscore[gidx][idx][1] /= len(g)

    for gidx in range(len(groups)):
        gscore[gidx] = sorted(gscore[gidx], key=lambda x: x[1])

    for idx in range(len(gscore)):
        print(gscore[idx][0], gscore[idx][-1])
 
    tsparsity =  {}
    for layer in gf_model.layers:
        if layer.__class__.__name__ == "Conv2D":
            w = layer.get_weights()[0]
            dim = w.shape[-1]
            odim = teacher.get_layer(layer.name).filters
            tsparsity[layer.name] = 1 - float(dim) / odim

    removed = set()
    for gidx, (g, _) in enumerate(groups):
        sparsity = tsparsity[g[0]]
        nchannels = len(gscore[gidx])
        idx = 0
        while float(idx) / nchannels < sparsity:
            item = gscore[gidx][idx]
            cidx = item[0]
            removed.add((gidx, cidx)) 
            idx += 1

    for (min_gidx, min_channel_idx) in removed:
        g, _ = groups[min_gidx]
        for layer in g:
            gates_weights[l2g[layer]][min_channel_idx] = 0.0

    for key in gates_weights:
        layer = gated_model.get_layer(key)
        layer.gates.assign(gates_weights[key])

    cmodel = parser.cut(gated_model)

    return cmodel
