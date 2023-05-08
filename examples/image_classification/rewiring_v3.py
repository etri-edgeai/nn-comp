import math
import json
import os
import copy

import yaml
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

from train import iteration_based_train, train_step, load_dataset
from rewiring import decode, get_add_inputs, replace_input


max_iters = 25
lr_mode = 0
with_label = True
with_distillation = False
augment = False
period = 25
num_remove = 505
num_masks = 1
pick_ratio = 1.0
window_size = 505
droprate = 0.25
pre_epochs = 0
min_channels = 2 # min channels + 1
save_path = "saved_grad_%d" % (window_size)

@njit
def find_min(cscore, gates, min_val, min_idx, lidx, ncol):
    for i in range(ncol):
        if (min_idx[0] == -1 or min_val > cscore[i]) and gates[i] == 1.0:
            min_val = cscore[i]
            min_idx = (lidx, i)
    return min_val, min_idx

def prune_step(X, model, teacher_logits, y, num_iter, groups, l2g, norm, inv_groups, period, score_info, g2l, pbar=None):

    for layer in model.layers:
        if layer.__class__ == SimplePruningGate:
            layer.trainable = True
            layer.collecting = True

    tape, loss, position_output = train_step(X, model, teacher_logits, y, ret_last_tensor=True)
    _ = tape.gradient(loss, model.trainable_variables)

    if (num_iter+1) % period == 0:
        grads = {}
        raw_grads = {}
        for key, gate in l2g.items():
            raw_grads[key] = model.get_layer(gate).grad_holder 

            sum_ = None
            for grad in model.get_layer(gate).grad_holder:
                grad = pow(grad, 2)
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
                        """
                        flag = False # copyflag
                        for l in g2l[layer.name]:
                            if "_copied_" in l:
                                flag = True
                                break
                        if not flag:
                        """
                        grad += layer.grad_holder[bidx]
                    grad = pow(grad, 2)
                    sum_ += tf.reduce_sum(grad, axis=0)

            # compute normalization
            cscore = sum_ / norm[gidx]
            cscore_[gidx] = cscore

        score_info["return"] = cscore_, grads, raw_grads

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
                    #print(layer.name, np.sum(gates_info[layer.name]))
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

    g2l = {}
    for key, value in l2g.items():
        if value not in g2l:
            g2l[value] = []
        g2l[value].append(key)

    def callback_before_update(idx, global_step, X, model_, teacher_logits, y, pbar):
        teacher_logits = None
        return prune_step(X, model_, teacher_logits, y, global_step, groups, l2g, norm, inv_groups, period, score_info, g2l, pbar)

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

    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            layer.set_weights(gmodel.get_layer(layer.name).get_weights())

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

        if layer["class_name"] == "Conv2D":
            layer["config"]["kernel_size"] = [1, 1]
            layer["config"]["use_bias"] = False

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

    left = next_group[1]
    right = next_group[0]

    if left[1] >= right[1]:
        temp = left
        left = right
        right = temp

    assert left[1] < right[1]
    layers_ = [(olayer_dict[layer]["name"], parser.torder[layer]) for layer in olayer_dict]
    input_layer = None
    layers_.sort(key=lambda x: x[1])

    layers = []
    holder = None
    for layer in layers_:
        layer = layer[0]
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

            if olayer_dict[layer]["class_name"] == "Conv2D": # Transform
                holder = layer
                break

    assert input_layer is not None 

    if olayer_dict[left[0]]["class_name"] == "Activation":
        if olayer_dict[olayer_dict[left[0]]["inbound_nodes"][0][0][0]]["class_name"] == "Add": # hard coding for resnet
            input_layer = left[0]

    if olayer_dict[left[0]]["class_name"] == "Activation":
        layers.append(copy.deepcopy(olayer_dict[left[0]]))
    rewire_copied_body(idx, layers, input_layer, new_input)
    return layers, str(idx)+"_copied_"+holder, holder

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
    basemodel,
    curr_model,
    model_handler,
    dataset,
    gates_info,
    gidx,
    idx,
    remove_masks,
    groups,
    parser,
    data_holder):

    print("Selection %d %d" % (gidx, idx))

    mask = remove_masks[gidx]
    submask = remove_masks[gidx][idx]
    ret = []
    max_picks = 0
    for iidx in range(len(mask)+1):
        if iidx <= idx:
            continue
        max_picks += 1
    num_picks = max(int(max_picks * pick_ratio), 1)
    assert num_picks >= 1

    if num_picks == max_picks:
        print("100% submask")
        for iidx in range(len(mask)+1):
            if iidx <= idx:
                continue

            submask[iidx] = 1
    else:
        for pick in range(num_picks):

            min_val = None
            min_iidx = -1
            for iidx in range(len(mask)+1):
                if iidx <= idx:
                    continue

                if submask[iidx] == 1:
                    continue

                submask[iidx] = 1
                model_ = remove_skip_edge(basemodel, curr_model, parser, groups, remove_masks)
                submask[iidx] = 0

                #tf.keras.utils.plot_model(model_, "temp_%d_%d.pdf" % (idx, iidx), show_shapes=True)

                for layer in model_.layers:
                    if len(layer.get_weights()) > 0:
                        if "copied" in layer.name:
                            w = curr_model.get_layer(cname2name(layer.name)).get_weights()
                            w[0] = np.expand_dims(np.average(w[0], axis=(0,1)), axis=(0,1))
                            if len(layer.get_weights()) == len(w):
                                layer.set_weights(w)
                            else:
                                layer.set_weights([w[0]])
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

            if min_val is None: # one case.
                break

            if min_iidx != -1:
                print(min_iidx, " is selected...")
                submask[min_iidx] = 1
            else:
                print("min_iidx is -1.")

def create_submodel(gidx, idx, mask, submask, groups, first_masked_idx, num_flow, olayer_dict, layer_dict, parser, conn_to, tidx, new_add=None):

    group = groups[gidx]
    add_name = group[idx][0]
    inputs = get_add_inputs(group, idx, olayer_dict, parser)
    target_dicts = [layer_dict[conn] for conn in conn_to[add_name]]

    _removed_layers = []
    added_layers = []

    if first_masked_idx is not None:
        leftmost_inputs = get_add_inputs(group, first_masked_idx, olayer_dict, parser)
    else:
        leftmost_inputs = inputs

    # check which one is pointed to this.
    if num_flow[idx] == 0: # add_index
        _removed_layers.append(add_name)
    else:
        remove_input(layer_dict[add_name], leftmost_inputs[0])

    for target_dict in target_dicts:
        replace_input(target_dict, add_name, leftmost_inputs[0]) # inputs[0] is the output from the previous module.

    act_dict = None
    for key in layer_dict:
        if layer_dict[key]["class_name"] == "Activation":
            act_dict = copy.deepcopy(layer_dict[key])
            break
        elif layer_dict[key]["class_name"] == "ReLU":
            act_dict = copy.deepcopy(layer_dict[key])
            break
    assert act_dict is not None

    for iidx, v in enumerate(submask):

        if iidx <= idx:
            continue

        if iidx == 0:
            continue

        if v == 1: # use idx - lidx path

            if iidx <= len(mask) - 1: 
                next_group = group[iidx][2]
            else:
                temp = parser.torder[group[len(mask)-1][0]]
                next_group = [(tidx[len(tidx)-1], len(tidx)-1), (tidx[temp], temp)] # [(right, tid), (left, tid)]
    
            subnet, output_name, holder = extract_body(idx, next_group, parser, olayer_dict, inputs[1]) # inputs[1]: body
            act_dict_ = copy.deepcopy(act_dict)
            act_dict_["name"] = output_name+"_act"
            act_dict_["config"]["name"] = act_dict_["name"]
            if act_dict["class_name"] != "ReLU":
                act_dict_["config"]["activation"] = "relu"
            act_dict_["inbound_nodes"] = [[[output_name, 0, 0, {}]]]
            subnet.append(act_dict_)
            output_name = act_dict_["name"]
            added_layers.extend(subnet)

            holder_follower = None # the follower of the holder in olayer_dict
            for j in range(parser.torder[holder]+1, len(olayer_dict)):
                for flow in olayer_dict[tidx[j]]["inbound_nodes"]:
                    for ib in flow:
                        if ib[0] == holder:
                            holder_follower = tidx[j]
                            break
            assert holder_follower is not None

            # last condition is for select_submodel for multiple zero mask situation.
            if num_flow[iidx-1] > 0 and mask[iidx-1][iidx-1] == 0 and np.sum(mask[iidx-1]) > 0:

                target_add_name = group[iidx-1][0]
                is_already = False

                for flow in layer_dict[target_add_name]["inbound_nodes"]:
                    for ib in flow:
                        if ib[0] == holder:
                            is_already = True
                            break

                if not is_already:
                    flow = [
                        [output_name, 0, 0, {}],
                        [holder, 0, 0, {}]
                    ]
                    layer_dict[target_add_name]["inbound_nodes"] = [flow]
                        
                    for flow in layer_dict[holder_follower]["inbound_nodes"]:
                        for ib in flow:
                            if ib[0] == holder:
                                ib[0] = target_add_name

                else:

                    inject_input(layer_dict[target_add_name], output_name)

            elif (num_flow[iidx-1] > 0 and mask[iidx-1][iidx-1] == 1) or np.sum(mask[iidx-1]) == 0:

                if new_add[iidx] is None:
                    new_layer = None
                    for layer in olayer_dict:
                        if olayer_dict[layer]["class_name"] == "Add":
                            new_layer = copy.deepcopy(olayer_dict[layer])
                            break
                    new_layer["name"] = str(iidx)+"_"+holder+"_add"
                    new_layer["config"]["name"] = new_layer["name"]
                    new_add[iidx] = new_layer["name"]

                    flow = [
                        [output_name, 0, 0, {}],
                        [holder, 0, 0, {}]
                    ]
                    new_layer["inbound_nodes"] = [flow]

                    for flow in layer_dict[holder_follower]["inbound_nodes"]:
                        for ib in flow:
                            if ib[0] == holder:
                                ib[0] = new_layer["name"]

                    added_layers.extend([new_layer])

                else:
                    new_layer = layer_dict[new_add[iidx]]
                    inject_input(new_layer, output_name)
            else:
                raise ValueError()
                                       
    return added_layers, _removed_layers, new_add

def remove_skip_edge(basemodel, curr_model, parser, groups, remove_masks, weight_copy=False):

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

    for gidx, (group, mask) in enumerate(zip(groups, remove_masks)):

        num_flow = [0 for _ in range(len(mask))]
        for idx, submask in enumerate(mask):
            num_flow[idx] = np.sum([
                    mask[idx_][idx+1] for idx_, v in enumerate(mask) if idx >= idx_ and mask[idx_][idx_] == 0 # mask[x][x] -> 0. mask[x][x'] -> data flow
                ])

        first_masked_idx = None
        new_add = [None for _ in range(len(mask)+1)]
        for idx, submask in enumerate(mask):
            if submask[idx] == 0: # it is masked.

                if first_masked_idx is None:
                    first_masked_idx = idx

                layer_dict = {}
                for layer in model_dict["config"]["layers"]:
                    layer_dict[layer["name"]] = layer

                _added_layers, _removed_layers, new_add = create_submodel(
                    gidx, idx, mask, submask, groups, first_masked_idx, num_flow, olayer_dict, layer_dict, parser, conn_to, tidx, new_add=new_add)

                for r in _removed_layers:
                    if layer_dict[r] in model_dict["config"]["layers"] :
                        model_dict["config"]["layers"].remove(layer_dict[r])
                model_dict["config"]["layers"].extend(_added_layers)

    #print(model_dict)
    model_json = json.dumps(model_dict)
    cmodel = tf.keras.models.model_from_json(model_json, custom_objects=parser.custom_objects)

    if weight_copy:
        for layer in cmodel.layers:
            try:
                if "copied_" in layer.name:
                    w = curr_model.get_layer(cname2name(layer.name)).get_weights()
                    w[0] = np.expand_dims(np.average(w[0], axis=(0,1)), axis=(0,1))
                    if len(layer.get_weights()) == len(w):
                        layer.set_weights(w)
                    else:
                        layer.set_weights([w[0]])

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
    return None
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


def evaluate(model, model_handler, groups, subnets, parser, datagen, train_func, num_iters=100, gmode=False, dataset="imagenet2012", sub_path=None, masking=None, custom_objects=None):

    if sub_path is not None:
        save_path_ = os.path.join(save_path, sub_path)
        if not os.path.exists(save_path_):
            os.mkdir(save_path_)

    parsers = [
        PruningNNParser(subnet, custom_objects=custom_objects) for subnet in subnets
    ]
    for p in parsers:
        p.parse()

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

    # determine masks
    gates_info = {}
    gmodel, parser, ordered_groups, pc = get_gmodel(dataset, model, model_handler, gates_info=gates_info)
    data_holder = []
    for k, data in enumerate(datagen):
        y = gmodel(data[0])
        data_holder.append((data[0], y))
        if k == 512:
            break

    test_model = model
    mask_history = set()
    if masking is None:
        for _ in range(num_masks):
            
            max_value = 0
            max_mask = None
            max_pair = None
            for gidx, mask in enumerate(masks):
                if len(mask) <= 1:
                    continue

                for idx, v in enumerate(mask):

                    if (gidx, idx) in mask_history:
                        continue
                    
                    masksnn[gidx][idx][idx] = 0 # masking
                    masks[gidx][idx] = 0

                    masksnn_ = copy.deepcopy(masksnn)
                    masks_ = copy.deepcopy(masks)

                    select_submodel(
                        model_backup,
                        model,
                        model_handler,
                        dataset,
                        gates_info,
                        gidx,
                        idx,
                        masksnn_,
                        groups,
                        parser,
                        data_holder)

                    test_model = remove_skip_edge(model_backup, model, parser, groups, masksnn_)

                    tf.keras.utils.plot_model(test_model, "temp_test_model.pdf", show_shapes=True)

                    target = None
                    for layer in test_model.layers:
                        if len(layer.get_weights()) > 0:
                            if "copied" in layer.name:
                                w = model.get_layer(cname2name(layer.name)).get_weights()
                                w[0] = np.expand_dims(np.average(w[0], axis=(0,1)), axis=(0,1))
                                if len(layer.get_weights()) == len(w):
                                    layer.set_weights(w)
                                else:
                                    layer.set_weights([w[0]])
                            else:
                                layer.set_weights(model.get_layer(layer.name).get_weights())

                        if "copied" in layer.name:
                            target = layer.name
                    
                    test_gmodel, test_parser, _, test_pc = get_gmodel(dataset, test_model, model_handler, gates_info=gates_info)

                    gate = test_gmodel.get_layer(test_pc.l2g[groups[gidx][idx][2][0][0]])
                    num_gates = gate.gates.shape[0]
                    gate.gates.assign(np.zeros(num_gates,))

                    print(target)
                    print(gate.name)
                    model_handler.compile(test_gmodel, run_eagerly=False)
                    (_, _, test_data_gen), (iters, iters_val) = load_dataset(dataset, model_handler, n_classes=100)
                    value = test_gmodel.evaluate(test_data_gen, verbose=1)[1]
                    print(value)
                    if max_mask is None or value > max_value:
                        max_value = value
                        max_mask = (gidx, idx)
                        max_pair = (masks_, masksnn_)

                    masksnn[gidx][idx][idx] = 1 # restore
                    masks[gidx][idx] = 1

            assert max_mask is not None
            masks, masksnn = max_pair
            mask_history.add(max_mask)
    else:
        masks, masksnn = masking

        # complete masksnn
        for gidx, mask in enumerate(masks):
            for idx, v in enumerate(mask):
                if mask[idx] == 0:
                    select_submodel(
                        model_backup,
                        model,
                        model_handler,
                        dataset,
                        gates_info,
                        gidx,
                        idx,
                        masksnn,
                        groups,
                        parser,
                        data_holder)

        mask_history.add(None) # dummy: do nothing

    if len(mask_history) > 0:
        test_model = remove_skip_edge(model_backup, model, parser, groups, masksnn)
        for layer in test_model.layers:
            if len(layer.get_weights()) > 0:
                if "copied" in layer.name:
                    w = model.get_layer(cname2name(layer.name)).get_weights()
                    w[0] = np.expand_dims(np.average(w[0], axis=(0,1)), axis=(0,1))
                    if len(layer.get_weights()) == len(w):
                        layer.set_weights(w)
                    else:
                        layer.set_weights([w[0]])
                else:
                    layer.set_weights(model.get_layer(layer.name).get_weights())
        model = test_model

    if pre_epochs > 0:
        train_func(model, pre_epochs, None)

    gates_info = {}
    removed_layers = set()
    sharing_groups = None
    l2g = None
    inv_groups = None
    recon_mode = True
    idx2rgroups = {}
    headers = {}
    masked = []
    split_flag = {}
    for it in range(num_iters):

        # conduct pruning
        (cscore, grads, raw_grads), continue_info, temp_output = prune(dataset, model, model_handler, target_ratio=0.5, continue_info=continue_info, gates_info=gates_info, dump=False)
        gmodel, l2g_, inv_groups_, sharing_groups_, parser_, pc_ = continue_info # sharing groups (different from `groups`)

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
            if not os.path.exists(save_path_):
                os.mkdir(save_path_)
            tf.keras.models.save_model(temp_output, save_path_+"/"+str(it-1)+"_temp.h5")

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
                            if rindex[1] == 0 and parser.torder[layer.name] < groups[rindex[0]][0][2][1][1]: # header finding
                                headers[rindex[0]] = layer.name
                            else:
                                if rindex not in idx2rgroups:
                                    idx2rgroups[rindex] = []
                                idx2rgroups[rindex].append(layer.name)

        total_ = 0
        remaining = 0
        for gidx in gates_info:
            gates = gates_info[gidx]
            total_ += gates.shape[0]
            remaining += np.sum(gates)

        ratio = remaining / total_ 
        print(remaining / total_)

        if remaining / total_ < 0.5 and False:
            recon_mode = False
        print("RECON MODE: ", recon_mode)

        count = {}
        residual_removal = {}
        for __ in range(window_size):
            min_val = -1
            min_idx = (-1, -1)
            for lidx, layer in enumerate(model.layers):
           
                if layer.__class__.__name__ == "Conv2D":
                    gidx_ = name2gidx(layer.name, l2g_, inv_groups_) # gidx on current model
                    residual_removal[layer.name] = None
                    if len(sharing_groups_[gidx_][0]) > 1:
                        score = cscore[gidx_] # copied + sharing + non-residual-conv
                    else:
                        score = grads[layer.name].numpy()

                    if layer.name not in gates_info: # copied + sharing case
                        gates_info[layer.name] = gmodel.get_layer(l2g_[layer.name]).gates.numpy()
                    gates = gates_info[layer.name]
                    if np.sum(gates) < min_channels:
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
                gidx_ = name2gidx(layer_name, l2g_, inv_groups_)
                for name in sharing_groups_[gidx_][0]:
                    if model.get_layer(name).__class__.__name__ == "Conv2D":
                        if name not in gates_info:
                            continue
                        gates = gates_info[name]
                        gates[min_idx[1]] = 0.0

        dump_model = model
         
        continue_info = None

        temp_gmodel, temp_parser, _, _ = get_gmodel(dataset, dump_model, model_handler, gates_info)
        ccmodel = temp_parser.cut(temp_gmodel)

        if not os.path.exists(save_path_):
            os.mkdir(save_path_)
        tf.keras.models.save_model(ccmodel, save_path_+"/"+str(it-1)+".h5")
        tf.keras.utils.plot_model(ccmodel, "temp.pdf", show_shapes=True)

        if droprate > 0.0:

            from droppath import DropPath
            if custom_objects is None:
                custom_objects = {"DropPath":DropPath}
            if "DropPath" not in custom_objects:
                custom_objects["DropPath"] = DropPath
            ccparser = NNParser(ccmodel, custom_objects=custom_objects)
            ccparser.parse()

            drlayers = []
            drpositions = []
            for layer in ccmodel.layers:
                if "_copied_" in layer.name and layer.__class__.__name__ == "Conv2D":
                    dr = DropPath(droprate)
                    drdict = {
                        "name":layer.name+"_dr",
                        "class_name":"DropPath",
                        "config":dr.get_config()
                    }
                    drdict["config"]["name"] = drdict["name"]
                    drlayers.append(drdict)
                    drpositions.append(layer.name)

            ccmodel = ccparser.insert_layers(drlayers, drpositions)
     
            if not os.path.exists(save_path_):
                os.mkdir(save_path_)
            tf.keras.models.save_model(ccmodel, save_path_+"/"+str(it-1)+"_dr.h5")

    return ccmodel

def parse(model, parser, model_type="efnet"):

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

    global num_masks, pick_ratio, window_size, num_remove, min_channels, droprate, pre_epochs
    gidx = -1
    idx = -1
    if os.path.exists("config.yaml"):
        with open("config.yaml", 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        num_rep = config["num_rep"]
        num_masks_ = config["num_masks"]
        if type(num_masks_) != list:
            num_masks_ = [num_masks_]

        pick_ratio_ = config["pick_ratio"]
        if type(pick_ratio_) != list:
            pick_ratio_ = [pick_ratio_]

        window_size = config["window_size"]
        num_remove = config["num_remove"]

        min_channels_ = config["min_channels"]
        if type(min_channels_) != list:
            min_channels_ = [min_channels_]

        droprate_ = config["droprate"]
        if type(droprate_) != list:
            droprate_ = [droprate_]

        pre_epochs_ = config["pre_epochs"]
        if type(pre_epochs_) != list:
            pre_epochs_ = [pre_epochs_]

        gidx = config["gidx"]
        idx = config["idx"]
        num_iters = config["num_iters"]

    groups = parse(model, parser, model_type)

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


    # prototype
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

    for _ in range(num_rep):
        for i in num_masks_:
            num_masks = i
            for j in pick_ratio_:
                pick_ratio = j
                for k in min_channels_:
                    min_channels = k
                    for q in droprate_:
                        droprate = q
                        for a in pre_epochs_:
                            pre_epochs = a

                            print(num_masks, pick_ratio, min_channels, droprate, pre_epochs)
                            if gidx != -1 and idx != -1:

                                masksnn[gidx][idx][idx] = 0 # masking
                                masks[gidx][idx] = 0

                                masksnn_ = copy.deepcopy(masksnn) # masksnn will be changed in evaluate().
                                masks_ = copy.deepcopy(masks)

                                masking = (masks_, masksnn_)
                                sub_path = "masking_targeted_%d_%d_%d" % (_, gidx, idx)
                                cmodel = evaluate(model, model_handler, new_groups, subnets, parser, datagen, train_func, num_iters=num_iters, gmode=gmode, dataset=dataset, sub_path=sub_path, masking=masking, custom_objects=custom_objects)

                            else:

                                for gidx, mask in enumerate(masks):
                                    if len(mask) <= 1:
                                        continue

                                    for idx, v in enumerate(mask):

                                        masksnn[gidx][idx][idx] = 0 # masking
                                        masks[gidx][idx] = 0

                                        masksnn_ = copy.deepcopy(masksnn) # masksnn will be changed in evaluate().
                                        masks_ = copy.deepcopy(masks)

                                        masking = (masks_, masksnn_)
                                        sub_path = "masking_%d_%d_%f_%d_%d_%d_%f_%d" % (_, num_masks, pick_ratio, min_channels, gidx, idx, droprate, pre_epochs)

                                        cmodel = evaluate(model, model_handler, new_groups, subnets, parser, datagen, train_func, num_iters=num_iters, gmode=gmode, dataset=dataset, sub_path=sub_path, masking=masking, custom_objects=custom_objects)

                                        masksnn[gidx][idx][idx] = 1 # restore
                                        masks[gidx][idx] = 1

    return cmodel


def apply_rewiring(train_data_generator, teacher, model_handler, gated_model, groups, l2g, parser, target_ratio, save_dir, save_prefix, save_steps, train_func, model_type="efnet", dataset="imagenet2012", custom_objects=None):

    global save_path
    save_path = save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    rewire(train_data_generator, teacher, model_handler, parser, train_func, True, model_type, dataset, custom_objects)

    xxx
