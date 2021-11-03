import tensorflow as tf
from tensorflow import keras
from scipy.linalg import circulant
import numpy as np

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser
from nncompress.backend.tensorflow_.transformation import parse, inject, cut
from nncompress.backend.tensorflow_ import SimplePruningGate, DifferentiableGate

from group_fisher import find_all, get_layers

def build_db_matrix(conv_weight):

    # (H, W, Input, Output)
    height, width, idim, odim = conv_weight.shape
    val = np.zeros((idim, odim, height, width, width, width))
    conv_weight_t = np.transpose(conv_weight, (2, 3, 0, 1))
    for i in range(idim):
        for o in range(odim):
            hcir = circulant([i for i in range(height)])
            K = conv_weight_t[i, o]
            cache = {}
            for h in range(height):
                for w in range(width):
                    h_ = hcir[h, w]
                    if h_ not in cache:            
                        vec = K[h]
                        assert len(vec.shape) == 1 # vector
                        cache[h_] = circulant(vec)
                    val[i, o, h, w] = cache[h_]
    return val

def compute_score(model, parsers):
    pass

def request(db_mat, layer):
    if layer.name not in db_mat:
        db_mat[layer.name] = build_db_matrix(layer.get_weights()[0])
    return db_mat[layer.name]

def make_gates(model, sparsity=0.5, custom_objects=None, avoid=None):
    parsers = parse(model, PruningNNParser, custom_objects=custom_objects, gate_class=SimplePruningGate)
    gmodel, gate_mapping = inject(parsers, avoid=avoid)

    l2g = {}
    for layer, flow in gate_mapping:
        l2g[layer] = gate_mapping[(layer, flow)][0]["config"]["name"]

    conv_layers = find_all(model, tf.keras.layers.Conv2D)
    layers = get_layers(model)
    glayers = get_layers(gmodel)

    db_mat = {}

    torder = {}
    for _, parser in parsers.items():
        v = parser.traverse()
        torder_ = {
            name:idx
            for idx, (name, _) in enumerate(v)
        }
        torder.update(torder_)

    convs = set()
    for _, p in parsers.items():
        groups_ = p.get_sharing_groups()
        for g in groups_:
            for layer in g:
                if layer in l2g:
                    convs.add(layer)

    affected_layers = {}
    affecting_layers = {}
    all_affecting_layers = {}
    for _, p in parsers.items():
        adict = p.get_affecting_layers()
        all_affecting_layers.update(adict)
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
                if layer not in affecting_layers:
                    affecting_layers[layer] = set()
                affecting_layers[layer].add(l)

    all_affected_layers = {}
    for (layer, _), g_ in all_affecting_layers.items():
        for l, level, tensor_idx in g_:
            if l not in all_affected_layers:
                all_affected_layers[l] = set()
            all_affected_layers[l].add(layer)

    for layer in conv_layers:

        if "project_conv" in layer.name:
            continue

        if "se_" in layer.name:
            continue
        print(layer.name)
       
        if layer.name not in affecting_layers:
            continue

        if layer.name not in affected_layers:
            continue

        left = list(affecting_layers[layer.name])
        right = list(affected_layers[layer.name])

        sum_left = None
        for left_conv in left:
            affected_ = all_affected_layers[left_conv]
            affected_ = sorted(affected_, key= lambda x: torder[x])
            muls = []
            for affected in affected_:
                if layers[affected].__class__ in [keras.layers.DepthwiseConv2D,\
                    keras.layers.BatchNormalization]:
                    muls.append(layers[affected])

            left_db = request(db_mat, layers[left_conv])
            sum_left_ = np.zeros(left_db.shape[1],)
            for j in range(left_db.shape[1]):
                left_db_j = left_db[:,j]
                sum_left_[j] = np.linalg.norm(left_db_j.reshape((left_db.shape[0], -1)), ord="fro")
               
                for mul in muls:
                    if mul.__class__ == keras.layers.DepthwiseConv2D:
                        s = np.linalg.norm(mul.get_weights()[0][:,:,j,0], ord="fro")
                        sum_left_[j] *= s
                    else:
                        a = mul.get_weights()[0]
                        sum_left_[j] *= a[j]
            if sum_left is None:
                sum_left = np.zeros(left_db.shape[1],)
            sum_left += sum_left_

        sum_right = None
        for right_conv in right:
            affected_ = all_affected_layers[layer.name]
            affected_ = sorted(affected_, key= lambda x: torder[x])
            muls = []
            for affected in affected_:
                if layers[affected].__class__ in [keras.layers.DepthwiseConv2D,\
                    keras.layers.BatchNormalization]:
                    muls.append(layers[affected])

            right_db = request(db_mat, layers[right_conv])
            sum_right_ = np.zeros(right_db.shape[0],)
            for k in range(right_db.shape[0]):
                right_db_k = right_db[k,:]
                sum_right_[k] = np.linalg.norm(right_db_k.reshape((right_db.shape[1], -1)), ord="fro")
               
                for mul in muls:
                    if mul.__class__ == keras.layers.DepthwiseConv2D:
                        s = np.linalg.norm(mul.get_weights()[0][:,:,j,0], ord="fro")
                        sum_right_[k] *= s
                    else:
                        a = mul.get_weights()[0]
                        sum_right_[k] *= a[k]
            if sum_right is None:
                sum_right = np.zeros(right_db.shape[0],)
            sum_right += sum_right_

        db = request(db_mat, layer)
        L = np.zeros((sum_left.shape[0], sum_right.shape[0]))
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                L[i,j] = np.sum(np.abs(db[i,j])) * (sum_left[i] / len(left)) * (sum_right[j] / len(right))

        out_w = np.zeros(L.shape[1],)
        for j in range(L.shape[1]):
            out_w[j] = np.sum(np.abs(L[:,j]))
            #w = np.abs(layer.get_weights()[0])
            #out_w = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))

        sorted_ = np.sort(out_w, axis=None)
        val = sorted_[int((len(sorted_)-1)*sparsity)]
        mask = (out_w >= val).astype(np.float32)

        gate = glayers[l2g[layer.name]]
        gate.gates.assign(mask)

        #print(all_affected_layers[layer.name])
        #print(left, right, layer.name)

    cmodel = cut(parsers, gmodel)
    return cmodel, gmodel
