import math
import json
import os
import copy

import tensorflow as tf
from tensorflow import keras
import numpy as np
from numba import njit
from numpy import dot
from numpy.linalg import norm

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, NNParser, serialize
from nncompress.backend.tensorflow_ import SimplePruningGate, DifferentiableGate
from nncompress import backend as M
from group_fisher import make_group_fisher, add_gates, compute_positions, flatten

from prep import add_augmentation, change_dtype

from train import iteration_based_train, train_step
from rewiring import decode, get_add_inputs, replace_input

max_iters = 25
lr_mode = 0
with_label = True
with_distillation = False
augment = False
period = 25
num_remove = 500
alpha = 5.0
num_picks = 1

@njit
def find_min(cscore, gates, min_val, min_idx, lidx, ncol):
    for i in range(ncol):
        if (min_idx[0] == -1 or min_val > cscore[i]) and gates[i] == 1.0:
            min_val = cscore[i]
            min_idx = (lidx, i)
    return min_val, min_idx

def prune_step(X, model, teacher_logits, y, num_iter, groups, l2g, norm, inv_groups, period, score_info, pbar=None):

    for layer in model.layers:
        if layer.__class__ == SimplePruningGate:
            layer.trainable = True
            layer.collecting = True

    tape, loss, position_output = train_step(X, model, teacher_logits, y, ret_last_tensor=True)
    _ = tape.gradient(loss, model.trainable_variables)

    if (num_iter+1) % period == 0:
        grads = {}
        for key, gate in l2g.items():

            sum_ = None
            for grad in model.get_layer(gate).grad_holder:
                grad = grad = pow(grad, 2)
                grad = tf.reduce_sum(grad, axis=0)
                if sum_ is None:
                    sum_ = grad
                else:
                    sum_ += grad
            if key not in inv_groups:
                grads[key] = sum_ / norm[inv_groups[l2g[key]]]
            else:
                grads[key] = sum_ / norm[inv_groups[key]]

        cscore_ = {}
        for gidx, group in enumerate(groups):

            if type(group) == dict:
                for l in group:
                    if type(l) == str:
                        num_batches = len(model.get_layer(l2g[l]).grad_holder)
                        break

                items = []
                max_ = 0
                for key, val in group.items():
                    if type(key) == str:
                        val = sorted(val, key=lambda x:x[0])
                        items.append((key, val))
                        for v in val:
                            if v[1] > max_:
                                max_ = v[1]
                sum_ = np.zeros((max_,))
                for bidx in range(num_batches):
                    for key, val in items:
                        gate = model.get_layer(l2g[key])
                        grad = gate.grad_holder[bidx]
                        if type(grad) == int and grad == 0:
                            continue

                        grad = pow(grad, 2)
                        grad = tf.reduce_sum(grad, axis=0)
                        for v in val:
                            sum_[v[0]:v[1]] += grad

            else:
                # compute grad based si
                num_batches = len(group[0].grad_holder)

                sum_ = 0
                for bidx in range(num_batches):
                    grad = 0
                    for lidx, layer in enumerate(group):
                        grad += layer.grad_holder[bidx]

                    grad = pow(grad, 2)
                    sum_ += tf.reduce_sum(grad, axis=0)

            # compute normalization
            cscore = sum_ / norm[gidx]
            cscore_[gidx] = cscore

        score_info["return"] = cscore_, grads

        for layer in model.layers:
            if layer.__class__ == SimplePruningGate:
                layer.grad_holder = []

    for layer in model.layers:
        if layer.__class__ == SimplePruningGate:
            layer.trainable = False
            layer.collecting = False

def get_gmodel(dataset, model, model_handler, gates_info=None):
    gmodel, _, parser, ordered_groups, _, pc = make_group_fisher(
        model,
        model_handler,
        model_handler.get_batch_size(dataset),
        period=period,
        target_ratio=0.5,
        enable_norm=True,
        num_remove=num_remove,
        enable_distortion_detect=False,
        norm_update=False,
        fully_random=False,
        custom_objects=model_handler.get_custom_objects(),
        save_steps=-1)

    for layer in gmodel.layers:
        if layer.__class__ == SimplePruningGate:
            layer.trainable = False
            layer.collecting = False

    if gates_info is not None and len(gates_info) > 0:
        for layer in gmodel.layers:
            if layer.__class__.__name__ == "Conv2D" and layer.name in parser.torder:
                
                if layer.name in gates_info:
                    print(layer.name, np.sum(gates_info[layer.name]))
                    gates = gates_info[layer.name]
                elif "copied" in layer.name:
                    oname = cname2name(layer.name)
                    gates = gates_info[oname]
                gmodel.get_layer(pc.l2g[layer.name]).gates.assign(gates)

    return gmodel, parser, ordered_groups, pc
   

def prune(dataset, model, model_handler, target_ratio=0.5, continue_info=None, gates_info=None, dump=False):
    
    if dataset == "imagenet2012":
        n_classes = 1000
    else:
        n_classes = 100

    if continue_info is None:
            
        gmodel, parser, ordered_groups, pc = get_gmodel(dataset, model, model_handler, gates_info=gates_info)

        if gates_info is not None:
            iteration_based_train(
                dataset,
                gmodel,
                model_handler,
                max_iters=period*alpha,
                lr_mode=lr_mode,
                teacher=None,
                with_label=with_label,
                with_distillation=with_distillation,
                callback_before_update=None,
                stopping_callback=None,
                augment=augment,
                n_classes=n_classes)

        continue_info = (gmodel, pc.l2g, pc.inv_groups, ordered_groups, parser, pc)
    else:
        gmodel, l2g, inv_groups, ordered_groups, parser, pc = continue_info

    if dump:
        tf.keras.utils.plot_model(model, "model.pdf", show_shapes=True)
        tf.keras.utils.plot_model(gmodel, "gmodel.pdf", show_shapes=True)
        cmodel = parser.cut(gmodel)
    else:
        cmodel = None

    norm = pc.norm
    l2g = pc.l2g
    groups = pc.gate_groups
    inv_groups = pc.inv_groups
    score_info = {}

    def callback_before_update(idx, global_step, X, model_, teacher_logits, y, pbar):
        teacher_logits = None
        return prune_step(X, model_, teacher_logits, y, global_step, groups, l2g, norm, inv_groups, period, score_info, pbar)

    def stopping_callback(idx, global_step):
        if global_step >= period: # one time pruning
            return True
        else:
            return False

    iteration_based_train(
        dataset,
        gmodel,
        model_handler,
        max_iters=period,
        lr_mode=lr_mode,
        teacher=None,
        with_label=with_label,
        with_distillation=with_distillation,
        callback_before_update=callback_before_update,
        stopping_callback=stopping_callback,
        augment=augment,
        n_classes=n_classes)

    return score_info["return"], continue_info, cmodel

def inject_input(target_dict, name, flow_idx=0, tensor_idx=0, val=None):
    if val is None:
        val = {}
    assert target_dict["class_name"] == "Add"
    inbound = target_dict["inbound_nodes"]
    for flow in inbound:
        flag = False
        for ib in flow:
            if ib[0] == name:
                # do nothing
                return
        flow.append(
            [name, flow_idx, tensor_idx, val]
        )
        break

def remove_input(target_dict, name):
    assert target_dict["class_name"] == "Add"
    inbound = target_dict["inbound_nodes"]
    removal = None
    for flow in inbound:
        for ib in flow:
            if ib[0] == name:
                removal = ib
        if removal is not None:
            flow.remove(removal)

def rewire_copied_body(idx, layers, input_layer, new_input):

    map_ = {} # name change
    for layer in layers:
        layer["config"]["name"] = str(idx)+"_copied_"+layer["config"]["name"]
        if "name" in layer:
            layer["name"] = str(idx)+"_copied_"+layer["name"]

    for layer in layers:
        inbound = layer["inbound_nodes"]
        if layer["config"]["name"] != str(idx)+"_copied_"+input_layer:
            for flow in inbound:
                for ib in flow:
                    assert type(ib[0]) == str
                    ib[0] = str(idx)+"_copied_"+ib[0]
        else:
            if new_input is not None:
                for flow in inbound:
                    for ib in flow:
                        assert type(ib[0]) == str
                        ib[0] = new_input

def extract_body(idx, next_group, parser, olayer_dict, new_input=None):

    left = next_group[2][1]
    right = next_group[2][0]
    assert left[1] < right[1]
    layers = []
    input_layer = None
    for layer in olayer_dict:
        if left[1] < parser.torder[layer] and parser.torder[layer] <= right[1]:
            layers.append(copy.deepcopy(olayer_dict[layer]))
            if input_layer is None:
                found = False
                inbound = olayer_dict[layer]["inbound_nodes"]
                for flow in inbound:
                    for ib in flow:
                        if ib[0] == left[0]:
                            found = True
                            break
                    if found:
                        break
                if found:
                    input_layer = layer
    assert input_layer is not None 

    rewire_copied_body(idx, layers, input_layer, new_input)

    return layers, str(idx)+"_copied_"+right[0]

def get_conn_to(name, parser):
    ret = []
    node = parser.get_nodes([name])[0]
    neighbors = parser._graph.out_edges(node[0], data=True)
    for n in neighbors:
        ret.append(n[1])
    return ret

def cname2name(cname):
    return "_".join(cname.split("_")[2:])

def select_submodel(
    curr_model,
    model_handler,
    dataset,
    gates_info,
    gidx,
    idx,
    mask,
    groups,
    first_masked_idx,
    num_flow,
    olayer_dict,
    model_dict,
    parser,
    conn_to,
    tidx,
    data_holder,
    last_add=None,
    prev_last_add=None):

    print("Selection %d %d" % (gidx, idx))

    submask = mask[idx]

    _added_layers = None
    _removed_layers = None
    out_layer_dict = None
    for pick in range(num_picks):

        min_val = None
        min_iidx = -1
        min_last_add = None
        min_layer_info = None

        for iidx in range(len(mask)+1):
            if iidx <= idx:
                continue

            if submask[iidx] == 1:
                continue

            if iidx <= len(mask)-1:
                num_flow[iidx] += 1
            submask[iidx] = 1

            print("---- %d" % iidx)

            model_dict_ = copy.deepcopy(model_dict)
            layer_dict_ = {}
            for layer in model_dict_["config"]["layers"]:
                layer_dict_[layer["name"]] = layer

            _added_layers, _removed_layers, last_add_ = create_submodel(
                gidx, idx, mask, submask, groups, first_masked_idx, num_flow, olayer_dict, layer_dict_, parser, conn_to, tidx, last_add=last_add, prev_last_add=prev_last_add)
            submask[iidx] = 0
            if iidx <= len(mask)-1:
                num_flow[iidx] -= 1

            for r in _removed_layers:
                if layer_dict_[r] in model_dict_["config"]["layers"] :
                    model_dict_["config"]["layers"].remove(layer_dict_[r])

            model_dict_["config"]["layers"].extend(_added_layers)
            model_json = json.dumps(model_dict_)
            model_ = tf.keras.models.model_from_json(model_json, custom_objects=parser.custom_objects)

            tf.keras.utils.plot_model(model_, "temp_%d_%d.pdf" % (idx, iidx), show_shapes=True)

            for layer in model_.layers:
                if len(layer.get_weights()) > 0:
                    if "copied" in layer.name:
                        layer.set_weights(curr_model.get_layer(cname2name(layer.name)).get_weights())
                    else:
                        layer.set_weights(curr_model.get_layer(layer.name).get_weights())

            gmodel_ = get_gmodel(dataset, model_, model_handler, gates_info=gates_info)[0]
            sum_ = 0
            for data, y in data_holder:
                output2 = gmodel_(data)
                sum_ += tf.keras.metrics.kl_divergence(y, output2)
            sum_ /= len(data_holder)
            sum_ = np.average(sum_)
            print(iidx, sum_)
            if min_val is None or sum_ < min_val:
                min_val = sum_
                min_iidx = iidx
                min_last_add = last_add_
                min_layer_info = (_added_layers, _removed_layers, layer_dict_)

        if min_val is None: # one case.
            break

        print("%d is selected." % min_iidx)
        if min_iidx <= len(mask)-1:
            num_flow[min_iidx] += 1
        submask[min_iidx] = 1

        _added_layers = min_layer_info[0]
        _removed_layers = min_layer_info[1]
        out_last_add = min_last_add
        out_layer_dict = min_layer_info[2]

    layer_dict = {}
    for layer in model_dict["config"]["layers"]:
        layer_dict[layer["name"]] = layer

    for r in _removed_layers:
        if layer_dict[r] in model_dict["config"]["layers"] :
            model_dict["config"]["layers"].remove(layer_dict[r])

    for layer in model_dict["config"]["layers"]:
        layer["inbound_nodes"] = out_layer_dict[layer["name"]]["inbound_nodes"]

    model_dict["config"]["layers"].extend(_added_layers)

    return out_last_add

def create_submodel(gidx, idx, mask, submask, groups, first_masked_idx, num_flow, olayer_dict, layer_dict, parser, conn_to, tidx, last_add=None, prev_last_add=None):

    group = groups[gidx]
    add_name = group[idx][0]
    inputs = get_add_inputs(group, idx, olayer_dict, parser)
    target_dicts = [layer_dict[conn] for conn in conn_to[add_name]]

    _removed_layers = []
    added_layers = []

    # check which one is pointed to this.
    if num_flow[idx] == 0:
        _removed_layers.append(add_name)
        if first_masked_idx is not None:
            if prev_last_add is None:
                leftmost_inputs = get_add_inputs(group, first_masked_idx, olayer_dict, parser)
            else:
                leftmost_inputs = [prev_last_add["name"]]
        else:
            assert False
            leftmost_inputs = inputs
    else:
        if prev_last_add is None:
            leftmost_inputs = get_add_inputs(group, first_masked_idx, olayer_dict, parser)
        else:
            leftmost_inputs = [prev_last_add["name"]]
        remove_input(layer_dict[add_name], leftmost_inputs[0])

    for target_dict in target_dicts:
        replace_input(target_dict, add_name, leftmost_inputs[0]) # inputs[0] is the output from the previous module.

    for iidx, v in enumerate(submask):
        if iidx <= idx:
            continue

        if v == 1: # use idx - lidx path

            if iidx <= len(mask) - 1: # since submask includes the outer conv, we use mask to compute the length of group.

                if num_flow[idx] == 0 or idx == len(mask)-1:
                    subnet, output_name = extract_body(idx, group[iidx], parser, olayer_dict, inputs[1]) # inputs[1]: body
                else:
                    subnet, output_name = extract_body(idx, group[iidx], parser, olayer_dict, add_name)

                next_add_dict = layer_dict[group[iidx][0]]
                inject_input(next_add_dict, output_name)
                added_layers.extend(subnet)

            else:

                add_name_at_last = group[len(mask)-1][0]

                if gidx != len(groups)-1:
                    next_inputs = get_add_inputs(groups[gidx+1], 0, olayer_dict, parser)
                    first_act_after_group = next_inputs[0]
                else:
                    first_act_after_group = "block7a_project_bn"

                subnet = []
                for j in range(parser.torder[add_name_at_last]+1, len(olayer_dict)):
                    subnet.append(copy.deepcopy(olayer_dict[tidx[j]]))
                    if olayer_dict[tidx[j]]["name"] == first_act_after_group:
                        break

                input_layer = tidx[parser.torder[add_name_at_last]+1] 

                if num_flow[idx] == 0:
                    rewire_copied_body(idx, subnet, input_layer, inputs[1])
                else:
                    rewire_copied_body(idx, subnet, input_layer, add_name_at_last)

                added_layers.extend(subnet)

                if last_add is None:

                    new_add_name = first_act_after_group + "_add_%d_%d" % (gidx, iidx)

                    # create new_add when the last module's residual connection is removed.
                    new_layer = None
                    for layer in olayer_dict:
                        if olayer_dict[layer]["class_name"] == "Add":
                            new_layer = copy.deepcopy(olayer_dict[layer])
                            break
                    last_add = new_layer
                    new_layer["name"] = new_add_name
                    new_layer["config"]["name"] = new_add_name
                    flow = [
                        [str(idx)+"_copied_"+first_act_after_group, 0, 0, {}],
                        [first_act_after_group, 0, 0, {}]
                    ]
                    new_layer["inbound_nodes"] = [flow]

                    # replace
                    next_act = get_conn_to(first_act_after_group, parser)
                    for next_ in next_act:
                        replace_input(layer_dict[next_], first_act_after_group, new_add_name)
                
                    added_layers.extend([new_layer])

                else:
                  
                    # last_add obj may not be in layer_dict, so we need to find a corresponding obj in layer_dict and inject.
                    inject_input(layer_dict[last_add["name"]], str(idx)+"_copied_"+first_act_after_group)

    return added_layers, _removed_layers, last_add

def remove_skip_edge(basemodel, curr_model, parser, groups, remove_masks, model_handler, dataset, gates_info, data_holder, gmasked, weight_copy=False):

    affecting_layers = parser.get_affecting_layers()
    affected_by = {}
    for a in affecting_layers:
        affected_by[a[0]] = [x[0] for x in affecting_layers[a]]
    sharing_groups = parser.get_sharing_groups() # unorderd
    invg = {}
    for g in sharing_groups:
        g_ = [(x, parser.torder[x]) for x in g]
        g_.sort(key=lambda x:x[1])
        g_ = [x[0] for x in g_]
        for layer in g:
            invg[layer] = g_

    model_dict = json.loads(basemodel.to_json())
    layer_dict = {}
    for layer in model_dict["config"]["layers"]:
        layer_dict[layer["name"]] = layer
    olayer_dict = copy.deepcopy(layer_dict)
    tidx = {}
    for layer in olayer_dict:
        tidx[parser.torder[layer]] = layer

    conn_to = {}
    for g in groups:
        for item in g:
            add_name = item[0]
            conn_to[add_name] = get_conn_to(add_name, parser)

    prev_last_add = None
    for gidx, (group, mask) in enumerate(zip(groups, remove_masks)):

        num_flow = [0 for _ in range(len(mask))]
        for idx, submask in enumerate(mask):
            num_flow[idx] = np.sum([
                    mask[idx_][idx] for idx_, v in enumerate(mask) if idx >= idx_ and mask[idx_][idx_] == 0 # mask[x][x] -> 0. mask[x][x'] -> data flow
                ])

        last_add = None
        first_masked_idx = None
        for idx, submask in enumerate(mask):
            if submask[idx] == 0: # it is masked.

                if first_masked_idx is None:
                    first_masked_idx = idx

                if (gidx, idx) not in gmasked:
                    last_add = select_submodel(
                        curr_model,
                        model_handler,
                        dataset,
                        gates_info,
                        gidx, idx, mask, groups, first_masked_idx, num_flow, olayer_dict, model_dict, parser, conn_to, tidx, data_holder, last_add=last_add, prev_last_add=prev_last_add)
                else:

                    layer_dict = {}
                    for layer in model_dict["config"]["layers"]:
                        layer_dict[layer["name"]] = layer

                    _added_layers, _removed_layers, last_add = create_submodel(
                        gidx, idx, mask, submask, groups, first_masked_idx, num_flow, olayer_dict, layer_dict, parser, conn_to, tidx, last_add=last_add, prev_last_add=prev_last_add)

                    for r in _removed_layers:
                        if layer_dict[r] in model_dict["config"]["layers"] :
                            model_dict["config"]["layers"].remove(layer_dict[r])
                    model_dict["config"]["layers"].extend(_added_layers)
        prev_last_add = last_add


    #print(model_dict)
    model_json = json.dumps(model_dict)
    cmodel = tf.keras.models.model_from_json(model_json, custom_objects=parser.custom_objects)

    if weight_copy:
        for layer in cmodel.layers:
            try:
                if "copied_" in layer.name:
                    layer.set_weights(curr_model.get_layer(cname2name(layer.name)).get_weights())
                else:
                    layer.set_weights(curr_model.get_layer(layer.name).get_weights())
            except Exception as e:
                pass # ignore
    return cmodel

def name2gidx(name, l2g, inv_groups):
    if name in inv_groups:
        gidx = inv_groups[name]
    else:
        gidx = inv_groups[l2g[name]]
    return gidx

def satisfy_max_depth(masks, rindex, max_depth=1):
    if max_depth == -1:
        return True
    right = rindex[1]
    mask = masks[rindex[0]]
    cnt = 1
    for i in range(right-1, -1, -1):
        if mask[i] != 0:
            break
        cnt += 1
    for i in range(right+1, len(mask)):
        if mask[i] != 0:
            break
        cnt += 1
    return cnt <= len(mask) * max_depth

def find_residual_group(sharing_group, layer_name, parser_, groups, model):
    # avoid the last residual group
    def filter_(node_data):
        return node_data["layer_dict"]["class_name"] == "Add"
    joints = parser_.get_joints(start=sharing_group[0])
    first = parser_.first_common_descendant(sharing_group, joints=joints, is_transforming=False)
    if model.get_layer(first).__class__.__name__ == "Add":

        joints_ = parser_.get_joints(start=sharing_group[0], filter_=filter_)
        nearest = parser_.first_common_descendant([layer_name], joints=joints_, is_transforming=False)
        
        rindex = None
        for ridx, g in enumerate(groups):
            for _lidx, item in enumerate(g):
                if item[0] == nearest: # hit
                    #if _lidx == len(g)-1: # avoid last.
                    #    break
                    rindex = (ridx, _lidx)
                    break
        return rindex
    else:
        return None


def evaluate(model, model_handler, groups, subnets, parser, datagen, train_func, gmode=False, dataset="imagenet2012", custom_objects=None):

    window_size = 500

    parsers = [
        PruningNNParser(subnet, custom_objects=custom_objects) for subnet in subnets
    ]
    for p in parsers:
        p.parse()

    """
    feats = []
    for g in groups:
        bottom = g[0][2][0][0]
        if g[0][2][0][1] > g[0][2][1][1]:
            bottom = g[0][2][1][0]
        feats.append(model.get_layer(bottom).output)
    feat_model = tf.keras.Model(model.inputs, feats)

    data_keep = []
    feat_data = [[] for _ in range(len(groups))]
    for k, data in enumerate(datagen):
        feat = feat_model(data[0])
        feat = [
            f.numpy() for f in feat
        ]
        data_keep.append(data[0])
        for gidx, f in enumerate(feat):
            feat_data[gidx].append(f)
        if k == 256:
            break
    """

    affecting_layers = parser.get_affecting_layers()
    model_backup = model
    continue_info = None
    history = set()

    masks = [[] for _ in range(len(groups))]
    for i, g in enumerate(groups):
        for item in g:
            masks[i].append(1)

    # removing skip edges debugging
    masksnn = []
    for mask in masks:
        masknn = []
        for idx, v in enumerate(mask):
            submask = []
            for idx_, u in enumerate(mask):
                if idx_ == idx:
                    submask.append(v)
                else:
                    submask.append(0)
            submask.append(0)
            masknn.append(submask)
        masksnn.append(masknn)

    """
    masksnn = []
    masks = [[1], [1], [1, 0], [0, 0], [0, 0, 0]]
    for gidx, mask in enumerate(masks):
        masknn = []
        for idx, v in enumerate(mask):
            submask = []
            for idx_, u in enumerate(mask):
                if idx_ == idx:
                    submask.append(v)
                elif gidx == 4 and idx == 1 and idx_ == 2:
                    submask.append(1)
                elif gidx == 3 and idx == 0 and idx_ == 1:
                    submask.append(1)
                else:
                    submask.append(0)
            if gidx == len(masks)-1 and idx in [0, 2]:
                submask.append(1)
            elif gidx == 2 and idx == 1:
                submask.append(1)
            elif gidx == 3 and idx == 1:
                submask.append(1)
            else:
                submask.append(0)
            print(submask)
            masknn.append(submask)
        masksnn.append(masknn)

    gates_info = None
    data_holder = []
    for k, data in enumerate(datagen):
        y = model(data[0])
        data_holder.append((data[0], y))
        if k == 256:
            break
    gmasked = set()
    gmasked.add((4,0))
    gmasked.add((4,1))
    gmasked.add((4,2))
    gmasked.add((2,1))
    gmasked.add((3,1))
    gmasked.add((3,0))
    model = remove_skip_edge(model_backup, model_backup, parser, groups, masksnn, model_handler, dataset, gates_info, data_holder, gmasked)

    tf.keras.utils.plot_model(model, "temp.pdf")
    xxx
    """

    residual_convs = set()

    gates_info = {}
    removed_layers = set()
    sharing_groups = None
    l2g = None
    inv_groups = None
    recon_mode = True
    gmasked = set()
    for it in range(100):

        # conduct pruning
        (cscore, grads), continue_info, temp_output = prune(dataset, model, model_handler, target_ratio=0.5, continue_info=continue_info, gates_info=gates_info, dump=it>1)
        gmodel, l2g_, inv_groups_, sharing_groups_, parser_, _ = continue_info # sharing groups (different from `groups`)

        data_holder = []
        for k, data in enumerate(datagen):
            y = gmodel(data[0])
            data_holder.append((data[0], y))
            if k == 256:
                break

        # one-time update
        if l2g is None:
            l2g = l2g_
        if inv_groups is None:
            inv_groups = inv_groups_
        if sharing_groups is None:
            sharing_groups = sharing_groups_

        exists = set()
        for layer in model.layers:
            exists.add(layer.name)
        
        if temp_output is not None:
            if not os.path.exists("saved"):
                os.mkdir("saved")
            tf.keras.models.save_model(temp_output, "saved/"+str(it-1)+".h5")

        # init
        if len(gates_info) == 0:
            for lidx, layer in enumerate(model.layers):
                if layer.__class__.__name__ == "Conv2D":
                    gates = gmodel.get_layer(l2g[layer.name]).gates.numpy()
                    gates_info[layer.name] = gates

            for lidx, layer in enumerate(model.layers):
                if layer.__class__.__name__ == "Conv2D":
                    gidx = name2gidx(layer.name, l2g, inv_groups) # gidx on the original model
                    if len(sharing_groups[gidx][0]) > 1:
                        rindex = find_residual_group(sharing_groups[gidx][0], layer.name, parser, groups, model_backup)
                        if rindex is not None:
                            residual_convs.add(layer.name)

        total_ = 0
        remaining = 0
        for gidx in gates_info:
            gates = gates_info[gidx]
            total_ += gates.shape[0]
            remaining += np.sum(gates)
        print(remaining / total_)

        if remaining / total_ < 0.5 and False:
            recon_mode = False
        print("RECON MODE: ", recon_mode)

        removing_idx = []
        count = {}
        residual_removal = {}
        flag = False
        masked = []
        for __ in range(window_size):
            min_val = -1
            min_idx = (-1, -1)
            for lidx, layer in enumerate(model.layers):
           
                if layer.__class__.__name__ == "Conv2D":
                    gidx_ = name2gidx(layer.name, l2g_, inv_groups_) # gidx on current model
                    residual_removal[layer.name] = None
                    if len(sharing_groups_[gidx_][0]) > 1:
                        
                        if layer.name in l2g:
                            gidx = name2gidx(layer.name, l2g, inv_groups) # gidx on the original model
                            rindex = find_residual_group(sharing_groups[gidx][0], layer.name, parser, groups, model_backup)

                            if rindex is not None:
                                if "copied_" in layer.name:
                                    continue

                                score = grads[layer.name].numpy()
                                if masks[rindex[0]][rindex[1]] != 0:

                                    if satisfy_max_depth(masks, rindex, max_depth=0.75) and recon_mode:
                                        residual_removal[layer.name] = rindex
                                    else:
                                        score = cscore[gidx_]
                                    #if "copied_" + layer.name in exists: # candidate to remove
                                    #    score += grads["copied_"+layer.name].numpy()
                                    #    score /= 2.0
                                else:
                                    exist_test = False
                                    for x in exists:
                                        if "copied_" in x and layer.name in x:
                                            exist_test = True
                                            break
                                    if exist_test:
                                        score = cscore[gidx_]
                            else:
                                score = cscore[gidx_]
                        else: # copied + sharing
                            assert "copied" in layer.name
                            if cname2name(layer.name) in residual_convs: # avoid duplicate removal
                                continue
                            score = cscore[gidx_] # copied + sharing + non-residual-conv
                    else:
                        score = grads[layer.name].numpy()

                    if layer.name not in gates_info: # copied + sharing case
                        gates_info[layer.name] = gmodel.get_layer(l2g_[layer.name]).gates.numpy()
                    gates = gates_info[layer.name]
                    if np.sum(gates) < 3.0:
                        continue
                    min_val_, min_idx_ = find_min(
                        score,
                        gates,
                        min_val,
                        min_idx,
                        lidx,
                        int(gmodel.get_layer(l2g_[layer.name]).gates.shape[0]))

                    if min_idx != min_idx_:
                        min_idx = min_idx_
                        min_val = min_val_
            if min_val != -1:
                layer_name = model.layers[min_idx[0]].name
                rindex = residual_removal[layer_name]
                if rindex is not None:
                    flag = True
                    masks[rindex[0]][rindex[1]] = 0
                    masked.append(rindex)
                    gates = gates_info[layer_name]
                    gates[min_idx[1]] = 0.0
                else:
                    gidx_ = name2gidx(layer_name, l2g_, inv_groups_)
                    for name in sharing_groups_[gidx_][0]:
                        if model.get_layer(name).__class__.__name__ == "Conv2D":
                            if name not in gates_info:
                                continue
                            gates = gates_info[name]
                            gates[min_idx[1]] = 0.0

            removing_idx.append((min_idx, min_val))

        for idx, val in removing_idx:
            layer_name = model.layers[idx[0]].name
            if layer_name in l2g: # residual conv only
                gidx = name2gidx(layer_name, l2g, inv_groups)

                if len(sharing_groups[gidx][0]) > 1:

                    rindex = find_residual_group(sharing_groups[gidx][0], layer.name, parser, groups, model_backup)
                    if rindex is not None:
                        assert rindex is not None
                        if rindex not in count:
                            count[rindex] = 0
                        count[rindex] += 1

        print(count)
        # compute a plan (masks)

        print(masks, flag)
        # rewire
        if flag:
            model_ = model
            for rindex in masked:
                masksnn[rindex[0]][rindex[1]][rindex[1]] = 0 # masking
            model = remove_skip_edge(model_backup, model, parser, groups, masksnn, model_handler, dataset, gates_info, data_holder, gmasked)
            for rindex in masked:
                gmasked.add(rindex)

            for layer in model.layers:
                if len(layer.get_weights()) > 0:
                    if "copied" in layer.name:
                        layer.set_weights(model_.get_layer(cname2name(layer.name)).get_weights())
                    else:
                        layer.set_weights(model_.get_layer(layer.name).get_weights())

            continue_info = None
            tf.keras.utils.plot_model(model, "temp.pdf", show_shapes=True)

        for layer in gmodel.layers:
            if layer.__class__.__name__ == "Conv2D":
                if layer.name in gates_info:
                    gates = gates_info[layer.name]
                elif "copied" in layer.name:
                    oname = cname2name(layer.name)
                    gates = gates_info[oname]
                gmodel.get_layer(l2g_[layer.name]).gates.assign(gates)

    # cutting
    xxx

    cmodel_groups = [[] for _ in range(len(groups))]
    for k, (group, subnet, local_parser, feat, local_groups) in enumerate(zip(groups, subnets, parsers, feat_data, local_group_info)):
        local_include = [curr_include[k]]
        psets = construct_pathset(local_groups, local_include)
        cmodel = psets2model(subnet, local_groups, psets, local_parser, custom_objects)
        cmodel_groups[k].append(cmodel)
    make_train_model(model, groups, cmodel_groups, scale=1.0, teacher_freeze=True)

    psets = construct_pathset(groups, curr_include)
    cmodel = psets2model(model, groups, psets, parser, custom_objects)

    return cmodel

def parse(model, parser):

    max_len = 4
    model_dict = json.loads(model.to_json())
    groups = []
    group = None
    for layer in model_dict["config"]["layers"]:
        if layer["class_name"] == "Add":
            left = None
            right = None
            for ib in layer["inbound_nodes"]:
                left = model.get_layer(ib[0][0])
                right = model.get_layer(ib[1][0])

                if (left.__class__.__name__ == "BatchNormalization" and right.__class__.__name__ == "Activation"): # Resnet
                    if group is None or len(group) > max_len:
                        group = []
                        groups.append(group)
                elif (left.__class__.__name__ == "Activation" and right.__class__.__name__ == "BatchNormalization"): # Resnet
                    if group is None or len(group) > max_len:
                        group = []
                        groups.append(group)
                elif (left.__class__.__name__ == "BatchNormalization" and right.__class__.__name__ == "BatchNormalization" and "resnet" in model_type): # Resnet
                    if group is not None:
                        group = None
                elif (left.__class__.__name__ != "Add" and right.__class__.__name__ != "Add") or len(group) > max_len: # start group
                    group = [] # assign new group
                    groups.append(group)

                pair = [(left.name, parser.torder[left.name]), (right.name, parser.torder[right.name])]

                assert len(ib) == 2
            assert len(layer["inbound_nodes"]) == 1
            if group is not None:
                group.append((layer["name"], parser.torder[layer["name"]], pair))
    return groups

def rewire(datagen, model, model_handler, parser, train_func, gmode=True, model_type="efnet", dataset="imagenet2012", custom_objects=None):

    model = change_dtype(model, "float32", custom_objects=custom_objects)
    tf.keras.utils.plot_model(model, "omodel.pdf", show_shapes=True)

    groups = parse(model, parser)

    subnets = []
    new_groups = []
    for i, g in enumerate(groups):
        bottom = g[0][2][0][0]
        if g[0][2][0][1] > g[0][2][1][1]:
                bottom = g[0][2][1][0]
        top = g[-1][0]
       
        layers = [
            layer for layer in model.layers if parser.torder[bottom] <= parser.torder[layer.name] and\
                parser.torder[layer.name] <= parser.torder[top]
        ]
        
        subnet, _, _ = parser.get_subnet(layers, model, custom_objects=custom_objects)

        if len(subnet.inputs) > 1:
            continue
        new_groups.append(g)

        subnet = change_dtype(subnet, "float32", custom_objects=custom_objects)

        tf.keras.utils.plot_model(subnet, "%d.pdf"%i)
        subnets.append(subnet)
    
    cmodel = evaluate(model, model_handler, new_groups, subnets, parser, datagen, train_func, gmode, dataset, custom_objects)

    tf.keras.utils.plot_model(cmodel, "model.pdf", show_shapes=True)
    tf.keras.models.save_model(cmodel, "%s_75.h5" % model_type)

    print(model.summary())
    print(cmodel.summary())
    
    return cmodel

def apply_rewiring(train_data_generator, teacher, model_handler, gated_model, groups, l2g, parser, target_ratio, save_dir, save_prefix, save_steps, train_func, model_type="efnet", dataset="imagenet2012", custom_objects=None):

    rewire(train_data_generator, teacher, model_handler, parser, train_func, True, model_type, dataset, custom_objects)


    xxx


