import math
import tensorflow as tf

from numba import jit
from tqdm import tqdm

def find_min(group_score, removed):
    min_gidx = -1
    min_channel_idx = -1
    min_score = -1
    for gidx, gscore in enumerate(group_score):
        for idx, val in enumerate(gscore):
            channel_idx = val[0]
            if (gidx, channel_idx) in removed:
                continue
            
            if idx == len(gscore) - 1: # keep the last channel to avoid 0-channel problem.
                continue

            if min_gidx == -1 or min_score > val[1]:
                min_score = val[1]
                min_channel_idx = channel_idx
                min_gidx = gidx
                break
    return min_gidx, min_channel_idx

def apply_hrank(train_data_generator, teacher, gated_model, groups, l2g, parser, target_ratio):

    convs = [
        layer for layer in teacher.layers if layer.__class__.__name__ == "Conv2D"
    ]

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
        model = tf.keras.Model(teacher.input, [layer.output for layer in targets_])
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
                    score[layer.name][channel_idx] += tf.math.reduce_sum(
                        tf.linalg.matrix_rank(
                            feat_mat, tol=None
                            )
                    )

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

    removed = set()
    while float(len(removed)) / n_channels < target_ratio:
        if len(removed) % 1000 == 0:
            print(float(len(removed)) / n_channels, target_ratio)
        min_gidx, min_channel_idx = find_min(gscore, removed)
        removed.add((min_gidx, min_channel_idx))

    for (min_gidx, min_channel_idx) in removed:
        g, _ = groups[min_gidx]
        for layer in g:
            gates_weights[l2g[layer]][min_channel_idx] = 0.0

    for key in gates_weights:
        layer = gated_model.get_layer(key)
        layer.gates.assign(gates_weights[key])

    cmodel = parser.cut(gated_model)
    return cmodel

