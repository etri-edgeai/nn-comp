import math
import tensorflow as tf

from numba import jit
from tqdm import tqdm

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
            if score[local_base + lidx] == -999.0:
                continue

            if min_score > score[local_base+lidx]:
                min_score = score[local_base+lidx]
                idx = local_base + lidx

        local_base += gates_info[gidx]

    return idx

def apply_curl(train_data_generator, teacher, gated_model, groups, l2g, parser, target_ratio, save_dir, save_prefix, save_steps):

    print("collecting...")
    data = []
    ty = []
    record_id = 1
    for X, y in train_data_generator:
        if record_id > 1:
            break
        data.append(X)
        ty.append(teacher(X)[0]) # the first output of teacher is the output logit.
        record_id += 1

    print("scoring...")
    n_channels = 0
    n_channels_group = [
        0 for _ in range(len(groups))
    ]
    gates_weights = {}
    gates_info = []
    for gidx, (g, _) in enumerate(groups):
        gates_info.append(gated_model.get_layer(l2g[g[0]]).gates.numpy().shape[0])
        n_channels_group[gidx] = gates_info[-1]
        n_channels += gates_info[-1]
        for g_ in g:
            gate = gated_model.get_layer(l2g[g_])
            gates_weights[l2g[g_]] = gate.gates.numpy()
    score = [0.0 for _ in range(n_channels)]
  
    local_base = 0
    for gidx, (g, _) in enumerate(tqdm(groups, ncols=80)):
        gate = gated_model.get_layer(l2g[g[0]]).gates.numpy()
        for lidx in range(gate.shape[0]):

            gate[lidx] = 0.0
            for layer in g:
                gate_ = gated_model.get_layer(l2g[layer])
                gate_.gates.assign(gate)

            sum_ = 0.0
            for X, ty_output in zip(data, ty):
                student_logits = gated_model(X)
                sum_ += tf.math.reduce_mean(tf.keras.losses.kl_divergence(student_logits[0], ty_output))
            score[local_base + lidx] = float(sum_)

            gate[lidx] = 1.0
            for layer in g:
                gate_ = gated_model.get_layer(l2g[layer])
                gate_.gates.assign(gate)

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
            if n_removed % save_steps == 0:
                for key in gates_weights:
                    layer = gated_model.get_layer(key)
                    layer.gates.assign(gates_weights[key])

                cmodel = parser.cut(gated_model)
                tf.keras.models.save_model(cmodel, save_dir+"/"+save_prefix+"_"+str(n_removed)+".h5")

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

            assert min_gidx != -1

            min_group, _ = groups[min_gidx]
            for min_layer in min_group:
                gates_weights[l2g[min_layer]][min_lidx] = 0.0
            score[hit_base + min_lidx] = -999.0

            n_removed += 1
            n_removed_group[gidx] += 1
            pbar.update(1)

    for key in gates_weights:
        layer = gated_model.get_layer(key)
        layer.gates.assign(gates_weights[key])

