import math
import json
import os
import copy
import shutil

import yaml
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numba import njit
from numpy import dot
from numpy.linalg import norm

import horovod.tensorflow.keras as hvd
import horovod
hvd.init()

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, NNParser, serialize
from nncompress.backend.tensorflow_ import SimplePruningGate, DifferentiableGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection
from nncompress import backend as M
from group_fisher import make_group_fisher, add_gates, compute_positions, flatten

from scipy import spatial
from scipy import stats

import random

from prep import add_augmentation, change_dtype

from train import iteration_based_train, train_step, load_dataset, train
from rewiring import decode, get_add_inputs, replace_input
from loader import get_model_handler
from curl import apply_curl

import reg as reg_
from utils import optimizer_factory

max_iters = 25
lr_mode = 0
with_label = True
with_distillation = False
augment = False
period = 25
num_remove = 505
num_masks = 1
num_hold = 0
pick_ratio = 1.0
window_size = 505
droprate = 0.25
pre_epochs = 0
min_channels = 2 # min channels + 1
pruning_masked_only = False
reg_factor = 0.0
pruning_method = "gfp"
activation = "relu"
reg_opt = "Custom/ortho"
reg_mode = "masked"
reg_dim_mode = "rows"
max_len = 1000
save_path = "saved_grad_%d" % (window_size)
dropblock = False
config_path = None
custom_object_scope = {
    "SimplePruningGate":SimplePruningGate, "StopGradientLayer":StopGradientLayer, "HvdMovingAverage":optimizer_factory.HvdMovingAverage, "Custom/ortho":reg_.OrthoRegularizer
}


def pretrain_(model_path, model_name, config_path_, epochs=1, lr=0.1): 

    custom_objects = {
        "SimplePruningGate":SimplePruningGate,
        "StopGradientLayer":StopGradientLayer,
        "HvdMovingAverage":optimizer_factory.HvdMovingAverage
    }

    silence_tensorflow()
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(c) for c in list(range(1,num_gpus+1))])
    hvd.init()
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for i, p in enumerate(physical_devices):
            tf.config.experimental.set_memory_growth(
                physical_devices[i], True
                )
        tf.config.set_visible_devices(physical_devices[hvd.local_rank()+1], 'GPU')
    tf.random.set_seed(2)
    random.seed(1234)
    np.random.seed(1234)
    model_handler = get_model_handler(model_name)

    dirname = os.path.dirname(model_path)

    with open(config_path_, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    batch_size = model_handler.get_batch_size(config["dataset"])
    model = tf.keras.models.load_model(model_path, custom_objects)

    if config["use_amp"]:
        tf.keras.backend.set_floatx("float16")
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        model = change_dtype(model, mixed_precision.global_policy(), custom_objects=custom_object_scope)

    model = add_augmentation(model, model_handler.width, train_batch_size=batch_size, do_mixup=config["mixup_alpha"] > 0, do_cutmix=config["cutmix_alpha"] > 0, custom_objects=custom_objects, update_batch_size=True)

    tf.keras.backend.set_floatx("float64")
    model = change_dtype(model, "float64", custom_objects=custom_object_scope)

    config["mode"] = "finetune"
    train(config["dataset"], model, model_handler.get_name()+"ooo", model_handler, n_classes=config["num_classes"], save_dir=dirname, conf=config, epochs_=epochs, sampling_ratio=config["sampling_ratio"], save_all=False, use_unique_name=True)


    if hvd.size() > 1 and hvd.local_rank() == 0:
        model_path = dirname+"/"+model_handler.get_name()+"ooo_"+config["dataset"]+"_model.best.h5"
        shutil.copy(model_path, dirname+"/output.h5")

def pretrain(model, epochs, model_handler):

    model_handler.compile(model, run_eagerly=False)

    #with tempfile.TemporaryDirectory() as dirpath:
    dirpath = "test"
    tf.keras.models.save_model(model, os.path.join(dirpath, "model.h5"))

    horovod.run(pretrain_, (os.path.join(dirpath, 'model.h5'), model_handler.get_name(), config_path, epochs), np=len(tf.config.list_physical_devices('GPU'))-1, use_mpi=True)

    model_path = dirpath+"/output.h5"
    if not os.path.exists(model_path):
        raise Exception("err")
    else:
        # load and transfer model.
        ret_model = tf.keras.models.load_model(model_path, custom_objects=custom_object_scope)
        """
        model = ret_model
        for layer in model.layers:
            if len(layer.get_weights()) > 0:
                layer.set_weights(ret_model.get_layer(layer.name).get_weights())
        """
    return ret_model

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
                        flag = False # copyflag
                        for l in g2l[layer.name]:
                            if "_copied_" in l and model.get_layer(l).__class__.__name__ in ["Conv2D", "Dense", "MultiHeadAttention"]:
                                flag = True
                                break
                        if not flag:
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

    last_ = parser.get_last_transformers()

    for layer in gmodel.layers:
        if layer.__class__ == SimplePruningGate:
            layer.trainable = False
            layer.collecting = False

    if gates_info is not None and len(gates_info) > 0:
        for layer in gmodel.layers:
            if layer.__class__.__name__ in ["Conv2D", "Dense", "MultiHeadAttention"] and layer.name in parser.torder:
                
                if layer.name in gates_info:
                    #print(layer.name, np.sum(gates_info[layer.name]))
                    gates = gates_info[layer.name]
                elif "copied" in layer.name:
                    oname = cname2name(layer.name)
                    if oname in last_:
                        continue # might be the last transformers
                    gates = gates_info[oname]
                if layer.name in last_:
                    continue
                gmodel.get_layer(pc.l2g[layer.name]).gates.assign(gates)

    ordered_groups_ = []
    for g in ordered_groups:
        if g[0][0] in last_:
            continue
        ordered_groups_.append(g)

    return gmodel, parser, ordered_groups_, pc
   

def prune(dataset, model, model_handler, target_ratio=0.5, continue_info=None, gates_info=None, dump=False):
    
    if dataset == "imagenet2012":
        n_classes = 1000
    else:
        n_classes = 100

    if continue_info is None:
        gmodel, parser, ordered_groups, pc = get_gmodel(dataset, model, model_handler, gates_info=gates_info)
        tf.keras.utils.plot_model(gmodel, "gmodel_vit.pdf")
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

def is_masked(layer, groups, masks, torder):

    if layer not in torder:
        return False

    for gidx, mask in enumerate(masks):
        for idx, v in enumerate(mask):
            if v == 0: # masked
                left = groups[gidx][idx][2][0][1]
                right = groups[gidx][idx][2][1][1]
                
                if left > right:
                    temp = left
                    left = right
                    right = temp
                
                if left <= torder[layer] and torder[layer] <= right:
                    return True
    return False

def add_regularizer(model, is_masked_func=None, mode="copied", custom_objects=None):
    
    if mode is None:
        return model
    elif mode == "masked":
        assert is_masked_func is not None

    model_dict = json.loads(model.to_json())
    for layer in model_dict["config"]["layers"]:
        if "Conv2D" in layer["class_name"]:
            if mode == "all" or (mode == "copied" and "copied" in layer["config"]["name"]) or (mode == "masked" and is_masked_func(layer["config"]["name"])):
                w = model.get_layer(layer["config"]["name"]).get_weights()[0]
                flag = 0
                for x in [int(v) for v in w.shape]:
                    if x > 1:
                        flag += 1
                if flag < 2:
                    continue
                if "ortho" in reg_opt:
                    reg_dict = {"class_name": reg_opt, "config": {"factor": reg_factor, "mode": reg_dim_mode}}
                elif "l2" in reg_opt:
                    reg_dict = {"class_name": reg_opt, "config": {"l2": reg_factor}}
                else:
                    raise NotImplementedError()
                layer["config"]["kernel_regularizer"] = reg_dict
    model_json = json.dumps(model_dict)
    if custom_objects is None:
        custom_objects = {}
    custom_objects += custom_object_scope
    model_ = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)

    for layer in model_.layers:
        layer.set_weights(model.get_layer(layer.name).get_weights())

    return model_

def remove_regularizer_if_one(model, is_masked_func=None, mode="copied", custom_objects=None):
    
    if mode is None:
        return model
    elif mode == "masked":
        assert is_masked_func is not None

    model_dict = json.loads(model.to_json())
    for layer in model_dict["config"]["layers"]:
        if "Conv2D" in layer["class_name"]:
            if mode == "all" or (mode == "copied" and "copied" in layer["config"]["name"]) or (mode == "masked" and is_masked_func(layer["config"]["name"])):
                w = model.get_layer(layer["config"]["name"]).get_weights()[0]
                flag = 0
                for x in [int(v) for v in w.shape]:
                    if x > 1:
                        flag += 1
                if flag < 2: # removal condition
                    layer["config"]["kernel_regularizer"] = None
    model_json = json.dumps(model_dict)
    if custom_objects is None:
        custom_objects = {}
    custom_objects += custom_object_scope
    model_ = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)

    for layer in model_.layers:
        layer.set_weights(model.get_layer(layer.name).get_weights())

    return model_

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
                if type(flow[0]) == str:
                    flow[0] = str(idx)+"_copied_"+flow[0]
                else:
                    for ib in flow:
                        assert type(ib[0]) == str
                        if "value" in ib[-1]:
                            assert ib[0] == ib[-1]["value"][0]
                        ib[0] = str(idx)+"_copied_"+ib[0]
                        if "value" in ib[-1]:
                            ib[-1]["value"][0] = ib[0]
        else:
            if new_input is not None:
                for flow in inbound:
                    for ib in flow:
                        assert type(ib[0]) == str
                        if "value" in ib[-1]:
                            assert ib[0] == ib[-1]["value"][0]
                        ib[0] = new_input
                        if "value" in ib[-1]:
                            ib[-1]["value"][0] = ib[0]


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

            if olayer_dict[layer]["class_name"] in ["Conv2D", "Dense", "MultiHeadAttention"]: # Transform
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

            max_val = None
            max_iidx = -1
            for iidx in range(len(mask)+1):
                if iidx <= idx:
                    continue

                if submask[iidx] == 1:
                    continue

                """
                # non-linearity check
                nonlinear = False
                left = groups[gidx][iidx][2][0][1]
                right = groups[gidx][iidx][2][1][1]
                if left > right:
                    temp = right
                    right = left
                    left = temp

                if iidx < 8:
                    continue

                for layer in basemodel.layers:
                    if left <= parser.torder[layer.name] and parser.torder[layer.name] <= right:
                        if "Dense" == layer.__class__.__name__:
                            nonlinear = True

                if nonlinear:
                    continue
                """

                submask[iidx] = 1
                model_ = remove_skip_edge(basemodel, curr_model, parser, groups, remove_masks)
                submask[iidx] = 0

                #tf.keras.utils.plot_model(model_, "temp_%d_%d.pdf" % (idx, iidx), show_shapes=True)

                for layer in model_.layers:
                    if len(layer.get_weights()) > 0:
                        if "copied" in layer.name:
                            w = curr_model.get_layer(cname2name(layer.name)).get_weights()
                            if layer.__class__.__name__ == "Conv2D":
                                w[0] = np.expand_dims(np.average(w[0], axis=(0,1)), axis=(0,1))
                            if len(layer.get_weights()) == len(w):
                                layer.set_weights(w)
                            else:
                                layer.set_weights([w[0]])
                        else:
                            layer.set_weights(curr_model.get_layer(layer.name).get_weights())

                print("+++++++ %d -> %d +++++++" % (idx, iidx))
                model_handler.compile(model_, run_eagerly=False)
                if dataset == "imagenet2012":
                    n_classes = 1000
                else:
                    n_classes = 100

                (_, _, test_data_gen), (iters, iters_val) = load_dataset(dataset, model_handler, n_classes=n_classes)
                value = model_.evaluate(test_data_gen, verbose=1)[1]

                """
                gmodel_ = get_gmodel(dataset, model_, model_handler, gates_info=gates_info)[0]
                sum_ = 0
                for data, y in data_holder:
                    output2 = gmodel_(data)
                    sum_ += tf.keras.metrics.kl_divergence(y, output2)
                sum_ /= len(data_holder)
                sum_ = np.average(sum_)
                """
                print(iidx, value)
                if max_val is None or value > max_val:
                    max_val = value
                    max_iidx = iidx

            if max_val is None: # one case.
                break

            if max_iidx != -1:
                print(max_iidx, " is selected...")
                submask[max_iidx] = 1
            else:
                print("max_iidx is -1.")

def create_submodel(gidx, idx, mask, submask, groups, first_masked_idx, num_flow, olayer_dict, layer_dict, parser, conn_to, tidx, new_add=None):

    group = groups[gidx]
    add_name = group[idx][0]
    #inputs = get_add_inputs(group, idx, olayer_dict, parser)
    inputs = [group[idx][2][0][0], group[idx][2][1][0]]

    target_dicts = [layer_dict[conn] for conn in conn_to[add_name]]

    _removed_layers = []
    added_layers = []

    new_model_output = None

    if first_masked_idx is not None:
        #leftmost_inputs = get_add_inputs(group, first_masked_idx, olayer_dict, parser)
        leftmost_inputs = [group[first_masked_idx][2][0][0], group[first_masked_idx][2][1][0]]
    else:
        leftmost_inputs = inputs

    # check which one is pointed to this.
    if num_flow[idx] == 0: # add_index
        _removed_layers.append(add_name)
    else:
        remove_input(layer_dict[add_name], leftmost_inputs[0])

    for target_dict in target_dicts:
        replace_input(target_dict, add_name, leftmost_inputs[0]) # inputs[0] is the output from the previous module.

    act_dict_template = {
        'class_name': 'Activation',
        'config': {
            'name': 'name', 'trainable': True, 'dtype': 'float32', 'activation': 'swish'},
        'name': 'name',
        'inbound_nodes': []
    }

    act_dict = act_dict_template
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
            act_dict_["config"]["activation"] = activation
            act_dict_["inbound_nodes"] = [[[output_name, 0, 0, {}]]]
            subnet.append(act_dict_)
            output_name = act_dict_["name"]
            added_layers.extend(subnet)

            holder_follower = None # the follower of the holder in olayer_dict
            for j in range(parser.torder[holder]+1, len(olayer_dict)):
                for flow in olayer_dict[tidx[j]]["inbound_nodes"]:
                    if type(flow[0]) == str:
                        if flow[0] == holder:
                            holder_follower = tidx[j]
                            break
                    else:
                        for ib in flow:
                            if ib[0] == holder:
                                holder_follower = tidx[j]
                                break

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

                    if holder_follower is not None:
                        for flow in layer_dict[holder_follower]["inbound_nodes"]:
                            if type(flow[0]) == str:
                                if flow[0] == holder:
                                    flow[0] = target_add_name
                            else:
                                for ib in flow:
                                    if ib[0] == holder:
                                        ib[0] = target_add_name
                    else:   # holder_follower is None -> last layer
                        new_model_output = target_add_name
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

                    if holder_follower is not None:
                        for flow in layer_dict[holder_follower]["inbound_nodes"]:
                            if type(flow[0]) == str:
                                if flow[0] == holder:
                                    flow[0] = new_layer["name"]
                            else:
                                for ib in flow:
                                    if ib[0] == holder:
                                        ib[0] = new_layer["name"]
                    else: # holder_follwer is None -> last layer
                        new_model_output = new_layer["name"]

                    added_layers.extend([new_layer])

                else:
                    new_layer = layer_dict[new_add[iidx]]
                    inject_input(new_layer, output_name)
            else:
                raise ValueError()
                                       
    return added_layers, _removed_layers, new_add, new_model_output

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

                _added_layers, _removed_layers, new_add, new_model_output = create_submodel(
                    gidx, idx, mask, submask, groups, first_masked_idx, num_flow, olayer_dict, layer_dict, parser, conn_to, tidx, new_add=new_add)

                for r in _removed_layers:
                    if layer_dict[r] in model_dict["config"]["layers"] :
                        model_dict["config"]["layers"].remove(layer_dict[r])
                model_dict["config"]["layers"].extend(_added_layers)

                if new_model_output is not None: # for vit
                    model_dict["config"]["output_layers"][0][0] = new_model_output

    #print(model_dict)
    model_json = json.dumps(model_dict)
    cmodel = tf.keras.models.model_from_json(model_json, custom_objects=parser.custom_objects)

    tf.keras.utils.plot_model(cmodel, "cmodel.pdf", show_shapes=True)

    if weight_copy:
        for layer in cmodel.layers:
            try:
                if "copied_" in layer.name:
                    w = curr_model.get_layer(cname2name(layer.name)).get_weights()
                    if layer.__class__.__name__ == "Conv2D":
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

def remove_group(model, parser, groups, masks):

    model_dict = json.loads(model.to_json())
    layer_dict = {}
    for layer in model_dict["config"]["layers"]:
        layer_dict[layer["name"]] = layer

    for group, mask in zip(groups, masks):

        for idx, (g, m) in enumerate(zip(group, mask)):
            if m == 0:
                left = g[2][0][1]
                right = g[2][1][1]
                if left > right:
                    temp = right
                    right = left
                    left = temp

                removed_layers = [
                    layer_dict[layer.name] for layer in model.layers if left < parser.torder[layer.name] and\
                        parser.torder[layer.name] <= right
                ]
                add_name = g[0]
                removed_layers.append(layer_dict[add_name])

                alter = g[2][0][0]
                for layer in model_dict["config"]["layers"]:
                    for flow in layer["inbound_nodes"]:
                        if type(flow[0]) == str:
                            if flow[0] == add_name:
                                flow[0] = alter
                        else:
                            for ib in flow:
                                if ib[0] == add_name:
                                    ib[0] = alter

                                if "value" in ib[-1]:
                                    ib[-1]["value"][0] = ib[0]
    
    for layer in removed_layers:
        model_dict["config"]["layers"].remove(layer)

    model_json = json.dumps(model_dict)
    custom_objects = parser.custom_objects
    custom_objects.update(custom_object_scope)
    model_ = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
    return model_

def get_left_right(g):
    
    left = g[2][0][1]
    left_ = g[2][0][0]
    right = g[2][1][1]
    right_ = g[2][1][0]
    if left > right:
        temp = right
        right = left
        left = temp

        temp = right_
        right_ = left_
        left_ = temp

    return left_, right_

def evaluate(model, model_handler, groups, subnets, parser, datagen, train_func, num_iters=100, gmode=False, dataset="imagenet2012", sub_path=None, masking=None, custom_objects=None, greedy_filter=None):

    if sub_path is not None:
        save_path_ = os.path.join(save_path, sub_path)
        if not os.path.exists(save_path_):
            os.mkdir(save_path_)

    shutil.copy("config.yaml", save_path_+"/config.yaml")

    if dataset == "imagenet2012":
        n_classes = 1000
    else:
        n_classes = 100

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
            
            if greedy_filter is not None and greedy_filter == "score":

                # compute score
                (_, _grads, _raw_grads), info, _ = prune(dataset, test_model, model_handler, target_ratio=0.5, continue_info=None, gates_info={}, dump=False)
                _, l2g_, inv_groups_, sharing_groups_, _, _ = info # sharing groups (different from `groups`)

                max_value = 0
                max_mask = None
                for gidx, (group, mask) in enumerate(zip(groups, masks)):

                    for idx, (g, m) in enumerate(zip(group, mask)):
                        if masks[gidx][idx] == 0:
                            continue

                        print(gidx, idx)
                        left_, right_ = get_left_right(g)
                        target = right_

                        sgidx = name2gidx(target, l2g_, inv_groups_)
                        if len(sharing_groups_[sgidx][0]) == 1:
                            continue

                        dist_ = 0
                        for layer_name in sharing_groups_[sgidx][0]:
                            if l2g_[layer_name] == l2g_[target]: # same gate
                                continue

                            dist__ = 1 - spatial.distance.correlation(_grads[target].numpy().argsort(), _grads[layer_name].numpy().argsort())
                            #dist__ = 1 - stats.kendalltau(_grads[target].numpy().argsort(), _grads[layer_name].numpy().argsort()).correlation
                            dist_ += dist__
                            print(dist__, target, layer_name)

                        print(dist_)
                        dist_ = dist_ / (len(sharing_groups_[sgidx][0])-1) 
                        print(dist_)

                        if max_mask is None or dist_ > max_value:
                            max_value = dist_
                            max_mask = (gidx, idx)

                print("MAX DIST:", max_value)
                assert max_mask is not None
                gidx, idx = max_mask
                masksnn[gidx][idx][idx] = 0 # masking
                masks[gidx][idx] = 0

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

                test_model = remove_skip_edge(model_backup, model, parser, groups, masksnn)

                for layer in test_model.layers:
                    if len(layer.get_weights()) > 0:
                        if "copied" in layer.name:
                            w = model.get_layer(cname2name(layer.name)).get_weights()
                            if layer.__class__.__name__ == "Conv2D":
                                w[0] = np.expand_dims(np.average(w[0], axis=(0,1)), axis=(0,1))
                            if len(layer.get_weights()) == len(w):
                                layer.set_weights(w)
                            else:
                                layer.set_weights([w[0]])
                        else:
                            layer.set_weights(model.get_layer(layer.name).get_weights())
                            
            else:
                max_value = 0
                max_mask = None
                max_pair = None
                for gidx, mask in enumerate(masks):

                    for idx, v in enumerate(mask):

                        if (gidx, idx) in mask_history:
                            continue
                       
                        if greedy_filter is not None:
                            if greedy_filter == "first":
                                if idx != 0 and masks[gidx][idx-1] != 0:
                                    continue
                            elif greedy_filter == "last":
                                idx = len(mask)-1 - idx
                                if idx != len(mask)-1 and masks[gidx][idx+1] != 0:
                                    continue
                            else:
                                pass

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
                                    if layer.__class__.__name__ == "Conv2D":
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
                        gate.gates.assign(np.ones(num_gates,))

                        print(target)
                        print(gate.name)
                        model_handler.compile(test_gmodel, run_eagerly=False)
                        (_, _, test_data_gen), (iters, iters_val) = load_dataset(dataset, model_handler, n_classes=n_classes)
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

        if dropblock:
            pass
        else:
            # complete masksnn
            for gidx, mask in enumerate(masks):
                for idx, v in enumerate(mask):
                    if mask[idx] == 0:
                        if np.sum(masksnn[gidx][idx]) == 0:
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
        if dropblock:
            test_model = remove_group(model, parser, groups, masks)
            tf.keras.utils.plot_model(test_model, "dropblock.pdf")
        else:
            test_model = remove_skip_edge(model_backup, model, parser, groups, masksnn)
        for layer in test_model.layers:
            if len(layer.get_weights()) > 0:
                if "copied" in layer.name:
                    w = model.get_layer(cname2name(layer.name)).get_weights()
                    if layer.__class__.__name__ == "Conv2D":
                        w[0] = np.expand_dims(np.average(w[0], axis=(0,1)), axis=(0,1))
                    if len(layer.get_weights()) == len(w):
                        layer.set_weights(w)
                    else:
                        layer.set_weights([w[0]])
                else:
                    layer.set_weights(model.get_layer(layer.name).get_weights())

    model = test_model

    """
    model_handler.compile(model, run_eagerly=False)
    if dataset == "imagenet2012":
        n_classes = 1000
    else:
        n_classes = 100
    (_, _, test_data_gen), (iters, iters_val) = load_dataset(dataset, model_handler, n_classes=n_classes)
    value = model.evaluate(test_data_gen, verbose=1)[1]
    """

    if reg_factor > 0.0:
        is_masked_func = lambda x: is_masked(x, groups, masks, parser.torder)
        model = add_regularizer(model, is_masked_func=is_masked_func, mode=reg_mode, custom_objects=parser.custom_objects)

    if pre_epochs > 0:
        #train_func(model, pre_epochs, None)
        model = pretrain(model, pre_epochs, model_handler)

        model_handler.compile(model, run_eagerly=False)
        if dataset == "imagenet2012":
            n_classes = 1000
        else:
            n_classes = 100
        (_, _, test_data_gen), (iters, iters_val) = load_dataset(dataset, model_handler, n_classes=n_classes)
        model.evaluate(test_data_gen, verbose=1)[1]

    if pruning_method == "curl":

        if dataset == "imagenet2012":
            n_classes = 1000
        else:
            n_classes = 100

        gates_info = {}
        gmodel, parser, ordered_groups, pc = get_gmodel(dataset, model, model_handler, gates_info=gates_info)
        (train_data_gen, _, test_data_gen), (iters, iters_val) = load_dataset(dataset, model_handler, n_classes=n_classes)
        apply_curl(train_data_gen, model, gmodel, ordered_groups, pc.l2g, parser, 0.7, save_path_, model_handler.get_name()+"_curl", save_steps=200)
        return None

    gates_info = {}
    removed_layers = set()
    recon_mode = True
    masked = []
    split_flag = {}
    for it in range(num_iters):

        # conduct pruning
        (cscore, grads, raw_grads), continue_info, temp_output = prune(dataset, model, model_handler, target_ratio=0.5, continue_info=continue_info, gates_info=gates_info, dump=False)
        gmodel, l2g_, inv_groups_, sharing_groups_, parser_, pc_ = continue_info # sharing groups (different from `groups`)

        last_ = parser_.get_last_transformers()

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
                if layer.__class__.__name__ in ["Conv2D", "Dense", "MultiHeadAttention"]:
                    if layer.name in last_: # last transformer
                        continue
                    gates = gmodel.get_layer(l2g_[layer.name]).gates.numpy()
                    gates_info[layer.name] = gates

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
        for __ in range(window_size):
            min_val = -1
            min_idx = (-1, -1)
            for lidx, layer in enumerate(model.layers):
           
                if layer.__class__.__name__ in ["Conv2D", "Dense", "MultiHeadAttention"]:
                    if layer.name in last_:
                        continue

                    gidx_ = name2gidx(layer.name, l2g_, inv_groups_) # gidx on current model
                    score = cscore[gidx_] # copied + sharing + non-residual-conv

                    if layer.name not in gates_info: # copied + sharing case
                        gates_info[layer.name] = gmodel.get_layer(l2g_[layer.name]).gates.numpy()
                    gates = gates_info[layer.name]

                    if pruning_masked_only and (not is_masked(layer.name, groups, masks, parser.torder) or "_copied_" in layer.name) and num_masks > 0:
                        continue

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
                gidx_ = name2gidx(layer_name, l2g_, inv_groups_)
                for name in sharing_groups_[gidx_][0]:
                    if model.get_layer(name).__class__.__name__ in ["Conv2D", "Dense", "MultiHeadAttention"]:
                        if name in last_:
                            continue
                        gates = gates_info[name]
                        gates[min_idx[1]] = 0.0

        continue_info = None

        dump_model = model
        temp_gmodel, temp_parser, _, _ = get_gmodel(dataset, dump_model, model_handler, gates_info)
        ccmodel = temp_parser.cut(temp_gmodel)

        if reg_factor > 0.0:
            ccmodel = remove_regularizer_if_one(ccmodel, is_masked_func=is_masked_func, mode=reg_mode, custom_objects=parser.custom_objects)

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
                if "_copied_" in layer.name and layer.__class__.__name__ in ["Conv2D", "Dense", "MultiHeadAttention"]:
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

    model_dict = json.loads(model.to_json())
    groups = []
    group = None
    lefts = []
    last_add = None
    for layer in model_dict["config"]["layers"]:
        if layer["class_name"] == "Add":
            last_add = layer["name"]
            left = None
            right = None
            pass_ = False
            for ib in layer["inbound_nodes"]:
                left = model.get_layer(ib[0][0])
                right = model.get_layer(ib[1][0])

                if parser.torder[left.name] > parser.torder[right.name]:
                    temp = left
                    left = right
                    right = temp

                lefts.append(left)
                if len(lefts) > num_hold:
                    left = lefts[-1*num_hold-1]
                    lefts = []
                else:
                    pass_ = True
                    break
                
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
                elif (left.__class__.__name__ != "Add" and right.__class__.__name__ != "Add") or group is None or len(group) > max_len: # start group
                        group = [] # assign new group
                        groups.append(group)

                pair = [(left.name, parser.torder[left.name]), (right.name, parser.torder[right.name])]

                assert len(ib) == 2
            assert len(layer["inbound_nodes"]) == 1
            if group is not None and not pass_:
                group.append((layer["name"], parser.torder[layer["name"]], pair))

    group = groups[-1]
    if last_add != group[-1][0]:
        group.append((last_add, parser.torder[last_add], pair))

    return groups

def rewire(datagen, model, model_handler, parser, train_func, gmode=True, model_type="efnet", dataset="imagenet2012", custom_objects=None):

    model = change_dtype(model, "float32", custom_objects=custom_objects)
    tf.keras.utils.plot_model(model, "omodel.pdf", show_shapes=True)

    global num_masks, pick_ratio, window_size, num_remove, min_channels, droprate, pre_epochs, pruning_masked_only, num_hold, config_path, dropblock, pruning_method, activation, max_len
    gidx = -1
    idx = -1
    if os.path.exists("config.yaml"):
        with open("config.yaml", 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        config_path = config["config_path"]

        dropblock = config["dropblock"]

        num_rep = config["num_rep"]
        num_masks_ = config["num_masks"]
        if type(num_masks_) != list:
            num_masks_ = [num_masks_]

        pick_ratio_ = config["pick_ratio"]
        if type(pick_ratio_) != list:
            pick_ratio_ = [pick_ratio_]

        num_hold = config["num_hold"]

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

        indices = config["indices"]
        num_iters = config["num_iters"]

        if "pruning_method" in config:
            pruning_method = config["pruning_method"]

        if "activation" in config:
            activation = config["activation"]

        if "max_len" in config:
            max_len = config["max_len"]

        pruning_masked_only = config["pruning_masked_only"]

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
        """
        subnet = change_dtype(subnet, "float32", custom_objects=custom_objects)
        tf.keras.utils.plot_model(subnet, "%d.pdf"%i)
        subnets.append(subnet)
        """

    groups = new_groups
    print(groups)

    masks = [[] for _ in range(len(groups))]
    for i, g in enumerate(groups):
        for item in g:
            masks[i].append(1)

    # removing skip edges debugging
    masksnn = []
    for mask in masks:
        masknn = []
        for _idx, v in enumerate(mask):
            submask = []
            for idx_, u in enumerate(mask):
                if idx_ == _idx:
                    submask.append(v)
                else:
                    submask.append(0)
            submask.append(0)
            masknn.append(submask)
        masksnn.append(masknn)

    print(len(masksnn))

    for _ in range(num_rep):
        
        if indices == "base":
            num_masks = 0
            sub_path = "baseline_%d" % _
            cmodel = evaluate(model, model_handler, new_groups, subnets, parser, datagen, train_func, num_iters=num_iters, gmode=gmode, dataset=dataset, sub_path=sub_path, custom_objects=custom_objects)
            continue
 
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

                            print(num_masks, pick_ratio, min_channels, droprate, pre_epochs, num_rep, indices)
                            if type(indices) == list:

                                for tup_ in indices:
                                    if len(tup_) == 3:
                                        gidx, idx, iidx = tup_
                                        masksnn[gidx][idx][iidx] = 1
                                    else:
                                        gidx, idx = tup_
                                        iidx = None
                                    masksnn[gidx][idx][idx] = 0 # masking
                                    masks[gidx][idx] = 0

                                masksnn_ = copy.deepcopy(masksnn) # masksnn will be changed in evaluate().
                                masks_ = copy.deepcopy(masks)

                                masking = (masks_, masksnn_)
                                sub_path = "masking_targeted_%d_%f_%d_%f_%d_indices" % (_, pick_ratio, min_channels, droprate, pre_epochs)
                                cmodel = evaluate(model, model_handler, new_groups, subnets, parser, datagen, train_func, num_iters=num_iters, gmode=gmode, dataset=dataset, sub_path=sub_path, masking=masking, custom_objects=custom_objects)

                                for tup_ in indices:
                                    if len(tup_) == 3:
                                        gidx, idx, iidx = tup_
                                        masksnn[gidx][idx][iidx] = 0
                                    else:
                                        gidx, idx = tup_
                                        iidx = None
                                    masksnn[gidx][idx][idx] = 1 # restore
                                    masks[gidx][idx] = 1

                            elif "greedy" in indices:

                                if "first" in indices:
                                    greedy_filter = "first"
                                elif "last" in indices:
                                    greedy_filter = "last"
                                elif "score" in indices:
                                    greedy_filter = "score"
                                else:
                                    greedy_filter = None

                                sub_path = "masking_%d_%d_%f_%d_greedy_%f_%d" % (_, num_masks, pick_ratio, min_channels, droprate, pre_epochs)
                                cmodel = evaluate(model, model_handler, new_groups, subnets, parser, datagen, train_func, num_iters=num_iters, gmode=gmode, dataset=dataset, sub_path=sub_path, custom_objects=custom_objects, greedy_filter=greedy_filter)

                            else:

                                stk = []
                                for gidx_, mask in enumerate(masks):
                                    #if len(mask) <= 1:
                                    #    continue
                                    for idx_, v in enumerate(mask):

                                        if "first" in indices:
                                            if idx_ != 0:
                                                continue

                                        elif "last" in indices:
                                            if idx_ != len(mask)-1:
                                                continue

                                        stk.append([(gidx_, idx_)])

                                cnt = 0
                                while len(stk) > 0:
                                    curr = stk.pop()
                                    if len(curr) < num_masks:
                                        for gidx_, mask in enumerate(masks):
                                            #if len(mask) <= 1:
                                            #    continue
                                            for idx_, v in enumerate(mask):
                                                if (gidx_, idx_) not in curr:
                                                    
                                                    if "first" in indices:
                                                        flag = False
                                                        for cgidx, cidx in curr:
                                                            if idx_ == 0 or (gidx_ == cgidx and cidx-1 == idx_):
                                                                flag = True
                                                                break
                                                        if not flag:
                                                            continue

                                                    elif "last" in indices:

                                                        flag = False
                                                        for cgidx, cidx in curr:
                                                            if idx_ == len(mask)-1 or (gidx_ == cgidx and cidx+1 == idx_):
                                                                flag = True
                                                                break
                                                        if not flag:
                                                            continue

                                                    curr_ = copy.deepcopy(curr)
                                                    curr_.append((gidx_, idx_))
                                                    stk.append(curr_)
                                        continue

                                    print("MASKS: ", cnt)
                                    for gidx_, idx_ in curr:
                                        masksnn[gidx_][idx_][idx_] = 0 # masking
                                        masks[gidx_][idx_] = 0
                                        print(gidx_, idx_)

                                    masksnn_ = copy.deepcopy(masksnn) # masksnn will be changed in evaluate().
                                    masks_ = copy.deepcopy(masks)

                                    masking = (masks_, masksnn_)
                                    sub_path = "masking_%d_%d_%f_%d_%f_%d_%d" % (_, num_masks, pick_ratio, min_channels, droprate, pre_epochs, cnt)
                                    cnt += 1

                                    cmodel = evaluate(model, model_handler, new_groups, subnets, parser, datagen, train_func, num_iters=num_iters, gmode=gmode, dataset=dataset, sub_path=sub_path, masking=masking, custom_objects=custom_objects)

                                    for gidx_, idx_ in curr:
                                        masksnn[gidx_][idx_][idx_] = 1 # restore
                                        masks[gidx_][idx_] = 1

    return cmodel


def apply_rewiring(train_data_generator, teacher, model_handler, gated_model, groups, l2g, parser, target_ratio, save_dir, save_prefix, save_steps, train_func, model_type="efnet", dataset="imagenet2012", custom_objects=None):

    global save_path
    save_path = save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    rewire(train_data_generator, teacher, model_handler, parser, train_func, True, model_type, dataset, custom_objects)

    xxx
