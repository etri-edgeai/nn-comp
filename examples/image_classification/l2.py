import math
import tensorflow as tf
import numpy as np

from numba import jit
from tqdm import tqdm

def apply_l2prune(train_data_generator, teacher, gated_model, groups, l2g, parser, target_ratio, gf_model):

    score = {}
    convs = [
        layer for layer in teacher.layers if layer.__class__.__name__ == "Conv2D"
    ]

    for t in convs:
        w = np.abs(t.get_weights()[0])
        sum_ = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))
        score[t.name] = list(sum_)

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
