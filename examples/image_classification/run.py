# coding: utf-8

from __future__ import print_function

import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import horovod.tensorflow.keras as hvd
if "NO_HOROVOD" not in os.environ:
    hvd.init()

import tensorflow as tf
if "NO_HOROVOD" not in os.environ:
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for i, p in enumerate(physical_devices):
            tf.config.experimental.set_memory_growth(
                physical_devices[i], True
                )
        tf.config.set_visible_devices(physical_devices[hvd.local_rank()], 'GPU')

tf.random.set_seed(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
import imgaug as ia
ia.seed(1234)
import argparse
import cv2
import math
import time
import logging
import yaml
import copy

import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import model_profiler

from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import StopGradientLayer
from nncompress.compression import lowrank
from nncompress.backend.tensorflow_.transformation import parse, inject, cut, unfold
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, NNParser
from nncompress import backend as M

from curl import apply_curl
from hrank import apply_hrank
from l2 import apply_l2prune
from rewiring_v3 import apply_rewiring
#from quant import apply_quantize
from group_fisher import make_group_fisher, add_gates, prune_step, compute_positions, flatten
from loader import get_model_handler

from train import load_dataset, train, iteration_based_train
from prep import add_augmentation, change_dtype
from utils import optimizer_factory

#from ray.tune.integration.horovod import DistributedTrainableCreator
import reg as reg_
from droppath import DropPath
custom_object_scope = {
    "SimplePruningGate":SimplePruningGate, "StopGradientLayer":StopGradientLayer, "HvdMovingAverage":optimizer_factory.HvdMovingAverage, "DropPath":DropPath, "Custom/ortho":reg_.OrthoRegularizer
}

def get_total_channels(groups, model, parser):
    total = 0
    for g in groups:
        if type(g[0][0]) == str:
            layer = model.get_layer(g[0][0])
            total += layer.filters
        else:
            g_ = flatten(g[0]) # g[1] is an integer
            _, group_struct = parser.get_group_topology(g_)
            max_ = 0
            for key, val in group_struct[0].items():
                if type(key) == str:
                    val = sorted(val, key=lambda x:x[0])
                    for v in val:
                        if v[1] > max_:
                            max_ = v[1]
            total += max_
    return total

def model_path_based_load(dataset, model_path, model_handler):
    if model_handler.get_custom_objects() is not None:
        for key, val in model_handler.get_custom_objects().items():
            custom_object_scope[key] = val

    if dataset == "imagenet2012" and model_path is None:
        model = model_handler.get_model(dataset="imagenet2012")
    else:
        if hasattr(model_handler, "get_custom_objects"):
            model = tf.keras.models.load_model(model_path, custom_objects=custom_object_scope)
        else:
            model = tf.keras.models.load_model(model_path)

    """
    if "efnet" in model_handler.get_name() and dataset != "imagenet2012":
        mean, var = model_handler.fix_mean_variance()

        if model.layers[0].__class__.__name__ == "InputLayer":
            assert hasattr(model.layers[2], "mean")
            model.layers[2].mean = mean
            model.layers[2].variance = var
        else:
            model.layers[0].get_layer("normalization").mean = mean
            model.layers[0].get_layer("normalization").variance = var
    """

    return model


def prune(
    dataset,
    model,
    model_handler,
    position_mode,
    with_label=False,
    label_only=False,
    distillation=True,
    fully_random=False,
    num_remove=1,
    enable_distortion_detect=False,
    norm_update=False,
    print_by_pruning=False,
    target_ratio=0.5,
    min_steps=-1,
    method="gf",
    finetune=False,
    period=25,
    n_classes=100,
    num_blocks=3,
    save_dir="",
    save_steps=-1,
    model_path2=None,
    backup_args=None,
    ret_score=False,
    eval_steps=-1,
    lr_mode=0,
    config=None):

    start_time = time.time()
    custom_object_scope = {
        "SimplePruningGate":SimplePruningGate, "StopGradientLayer":StopGradientLayer
    }


    if type(position_mode) != int:
        pos_str = "custom"
    if type(position_mode) == list:
        pos_str = "search"
    elif type(position_mode) == str and os.path.splitext(position_mode)[1] == ".json":
        pos_str = "from_file"
    else:
        pos_str = str(position_mode)

    if label_only and method == "gf":
        postfix = "_"+pos_str+"_"+dataset+"_"+str(with_label)+"_gf_label_only_"+str(target_ratio)+"_"+str(num_remove)
    elif method == "curl":
        postfix = "_"+pos_str+"_"+dataset+"_"+str(with_label)+"_curl_"+str(target_ratio)
    elif method == "rewire":
        postfix = "_"+pos_str+"_"+dataset+"_"+str(with_label)+"_rewire_"+str(target_ratio)
    elif method == "quantize":
        postfix = "_"+pos_str+"_"+dataset+"_"+str(with_label)+"_quant_"+str(target_ratio)
    elif method == "hrank":
        postfix = "_"+pos_str+"_"+dataset+"_"+str(with_label)+"_hrank_"+str(target_ratio)
    elif method == "l2":
        postfix = "_"+pos_str+"_"+dataset+"_"+str(with_label)+"_l2_"+str(target_ratio)
    elif method == "gf":
        postfix = "_"+pos_str+"_"+dataset+"_"+str(with_label)+"_gf"+"_"+str(num_blocks)+"_"+str(target_ratio)+"_"+str(num_remove)
    elif method == "t-finetune":
        assert model_path2 is not None
        postfix = os.path.splitext(os.path.basename(model_path2))[0]
        postfix = "_".join(postfix.split("_")[1:])+"_finetuned" # remove name
    else:
        raise NotImplementedError("Method name error!")

    if backup_args is not None:
        import json
        with open(model_handler.get_name()+"_"+postfix+".log", "w") as file_:
            json.dump(backup_args, file_)

    (_, _, test_data_gen), (iters, iters_val) = load_dataset(dataset, model_handler, n_classes=n_classes)
    def validate(model_):
        model_handler.compile(model_, run_eagerly=True)
        if dataset == "imagenet":
            return model_.evaluate(test_data_gen, verbose=1, steps=int(math.ceil(50000.0/model_handler.batch_size)))[1]
        else:
            return model_.evaluate(test_data_gen, verbose=1)[1]

    if method == "curl":
        gmodel, copied_model, l2g, ordered_groups, torder, parser, _ = add_gates(model, custom_objects=model_handler.get_custom_objects())
        backup = model_handler.batch_size
        model_handler.batch_size = 256
        (train_data_generator_, _, _), (_, _) = load_dataset(dataset, model_handler, training_augment=False, n_classes=n_classes)
        copied_model = add_augmentation(copied_model, model_handler.width, train_batch_size=model_handler.batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)
        gmodel = add_augmentation(gmodel, model_handler.width, train_batch_size=model_handler.batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)
        #model_handler.batch_size = backup
        apply_curl(train_data_generator_, copied_model, gmodel, ordered_groups, l2g, parser, target_ratio, save_dir+"/pruning_steps", model_handler.get_name()+postfix, save_steps=save_steps)

    elif method == "rewire":
        gmodel, copied_model, l2g, ordered_groups, torder, parser, _ = add_gates(model, custom_objects=model_handler.get_custom_objects())
        backup = model_handler.batch_size
        (train_data_generator_, _, _), (_, _) = load_dataset(dataset, model_handler, training_augment=False, n_classes=n_classes)
        copied_model = add_augmentation(copied_model, model_handler.width, train_batch_size=model_handler.batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)
        gmodel = add_augmentation(gmodel, model_handler.width, train_batch_size=model_handler.batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)
        #model_handler.batch_size = backup

        config["mode"] = "finetune"
        train_func = lambda model_, epochs_, save_dir_: train(dataset, model_, model_handler.get_name()+"_rewire", model_handler, run_eagerly=False, n_classes=n_classes, save_dir=save_dir_, conf=config, epochs_=epochs_)

        apply_rewiring(train_data_generator_, copied_model, model_handler, gmodel, ordered_groups, l2g, parser, target_ratio, save_dir+"/rewiring", model_handler.get_name()+postfix, save_steps=save_steps, train_func=train_func, model_type=model_handler.get_name(), dataset=dataset)


    elif method == "quantize":
        gmodel, copied_model, l2g, ordered_groups, torder, parser, _ = add_gates(model, custom_objects=model_handler.get_custom_objects())
        backup = model_handler.batch_size
        model_handler.batch_size = 16
        (train_data_generator_, _, _), (_, _) = load_dataset(dataset, model_handler, training_augment=False, n_classes=n_classes)
        copied_model = add_augmentation(copied_model, model_handler.width, train_batch_size=model_handler.batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)
        gmodel = add_augmentation(gmodel, model_handler.width, train_batch_size=model_handler.batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)
        #model_handler.batch_size = backup
        apply_quantize(validate, train_data_generator_, copied_model, gmodel, ordered_groups, l2g, parser, target_ratio, save_dir+"/pruning_steps", model_handler.get_name()+postfix, save_steps=save_steps)

    elif method == "diff":

        gf_model= model_path_based_load(dataset, model_path2, model_handler)
        class SparsityCallback(tf.keras.callbacks.Callback):

            def __init__(self, model):
                self.set_model(model)

            def on_epoch_end(self, epoch, logs=None):

                print("\n------------------------")
                pruning_layers = []
                for layer in self.model.layers:
                    if type(layer).__name__ == "DifferentiableGate":
                        pruning_layers.append(layer)

                for layer in pruning_layers:
                    print(float(layer.get_sparsity()))
                print("\n")

        from nncompress.backend.tensorflow_ import DifferentiableGate
        model_ = unfold(model, custom_object_scope)
        #model_ = add_augmentation(model_, model_handler.width, train_batch_size=model_handler.batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)
        parser = PruningNNParser(model_, custom_objects=custom_object_scope, gate_class=DifferentiableGate)
        parser.parse()
        gmodel_, gate_mapping = parser.inject(with_mapping=True)
        dmodel = make_distiller(gmodel_, model_, [])

        gmodel, copied_model, l2g_, ordered_groups, torder, parser, _ = add_gates(model, custom_objects=model_handler.get_custom_objects())
        il2g_ = {}
        for key, value in l2g_.items():
            il2g_[value] = key

        sparsity_cbk = SparsityCallback(dmodel)
        l2g = {}
        il2g = {}
        for layer, flow in gate_mapping:
            l2g[layer] = gate_mapping[(layer, flow)][0]["config"]["name"]
            il2g[gate_mapping[(layer, flow)][0]["config"]["name"]] = layer

        for layer in dmodel.layers:
            if type(layer) == DifferentiableGate:
                target = il2g[layer.name]
                layer.sparsity = gf_model.get_layer(l2g_[target]).get_sparsity()
                print(layer.sparsity, layer.name)
                dmodel.add_loss(layer.get_sparsity_loss)

        config["mode"] = "distillation_label_free"
        train(dataset, dmodel, model_handler.get_name()+"_diff", model_handler, callbacks=[sparsity_cbk], run_eagerly=False, n_classes=n_classes, save_dir=save_dir, conf=config, epochs_=30)

        for layer in gmodel.layers:
            if type(layer) == SimplePruningGate:
                target = il2g_[layer.name]
                layer.gates.assign(dmodel.get_layer(l2g[target]).gates.numpy())
            else:
                layer.set_weights(dmodel.get_layer(layer.name).get_weights())
        
        tf.keras.models.save_model(gmodel, model_handler.get_name()+"_diff_gated.h5")
        return None

    elif method == "hrank":

        gmodel, copied_model, l2g, ordered_groups, torder, parser, _ = add_gates(model, custom_objects=model_handler.get_custom_objects())
        backup = model_handler.batch_size
        model_handler.batch_size = 256
        (train_data_generator_, _, _), (_, _)  = load_dataset(dataset, model_handler, training_augment=False, n_classes=n_classes)
        copied_model = add_augmentation(copied_model, model_handler.width, train_batch_size=model_handler.batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)
        gmodel = add_augmentation(gmodel, model_handler.width, train_batch_size=model_handler.batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)
        gf_model= model_path_based_load(dataset, model_path2, model_handler)
        apply_hrank(train_data_generator_, copied_model, gmodel, ordered_groups, l2g, parser, target_ratio, gf_model)
        cmodel = parser.cut(gmodel)
        tf.keras.models.save_model(cmodel, save_dir+"/"+model_handler.get_name()+postfix+"_untrained_.h5")
        tf.keras.models.save_model(gmodel, save_dir+"/"+model_handler.get_name()+postfix+"_untrained_gated.h5")

    elif method == "l2":

        gmodel, copied_model, l2g, ordered_groups, torder, parser, _ = add_gates(model, custom_objects=model_handler.get_custom_objects())
        backup = model_handler.batch_size
        model_handler.batch_size = 256
        (train_data_generator_, _, _), (_, _) = load_dataset(dataset, model_handler, training_augment=False, n_classes=n_classes)
        model_handler.batch_size = backup
        gf_model= model_path_based_load(dataset, model_path2, model_handler)
        apply_l2prune(train_data_generator_, copied_model, gmodel, ordered_groups, l2g, parser, target_ratio, gf_model)

    elif method == "gf":
        gmodel, copied_model, parser, ordered_groups, torder, pc = make_group_fisher(
            model,
            model_handler,
            model_handler.get_batch_size(dataset),
            period=period,
            target_ratio=target_ratio,
            enable_norm=True,
            num_remove=num_remove,
            enable_distortion_detect=enable_distortion_detect,
            norm_update=norm_update,
            fully_random=fully_random,
            custom_objects=model_handler.get_custom_objects(),
            save_steps=save_steps,
            save_prefix=model_handler.get_name()+postfix,
            save_dir=save_dir+"/pruning_steps",
            logging_=False)

    elif method == "t-finetune":
        assert model_path2 is not None
        gmodel_ = model_path_based_load(dataset, model_path2, model_handler)
        gmodel, copied_model, l2g, ordered_groups, torder, parser, _ = add_gates(model, custom_objects=model_handler.get_custom_objects())

        # gmodel_ should be identical to gmodel
        for layer in gmodel_.layers:
            glayer = gmodel.get_layer(layer.name)
            glayer.set_weights(layer.get_weights())

    else:
        raise NotImplementedError("unknown method")

    def callback_before_update(idx, global_step, X, model_, teacher_logits, y, pbar):
        if method in ["curl", "hrank", "t-finetune", "l2"]:
            return

        if distillation:
            assert teacher_logits is not None
        else:   
            assert teacher_logits is None

        if label_only:
            assert with_label
            teacher_logits = None
        
        # Do pruning
        if with_label:
            return prune_step(X, model_, teacher_logits, y, pc, print_by_pruning, pbar)
        else:
            assert distillation
            return prune_step(X, model_, teacher_logits, None, pc, print_by_pruning, pbar)

    def stopping_callback(idx, global_step):
        if min_steps != -1:
            if global_step >= min_steps and (method in ["curl", "hrank", "t-finetune", "l2"] or not pc.continue_pruning):
                return True
            else:
                return False
        else:
            if method not in ["curl", "hrank", "t-finetune", "l2"]:
                return not pc.continue_pruning
            else:
                return False

    if type(position_mode) == int:
        positions = compute_positions(
            copied_model, ordered_groups, torder, parser, position_mode, num_blocks, model_handler.get_heuristic_positions())
    elif type(position_mode) == str:
        import json
        with open(position_mode, "r") as f:
            positions = json.load(f)["data"]
    else:
        positions = position_mode

    logging.info(positions)
    print(positions)
    if not ret_score:
        profile = model_profiler.model_profiler(copied_model, 1)
        print(profile)
        logging.info(profile)
        cmodel = parser.cut(gmodel)
        print(cmodel.count_params())
        logging.info(cmodel.count_params())
        #print(validate(cmodel))

    if distillation:
        if model_handler.get_custom_objects() is not None:
            for key, val in model_handler.get_custom_objects().items():
                custom_object_scope[key] = val
        with keras.utils.custom_object_scope(custom_object_scope):
            t_model = M.add_prefix(copied_model, "t_", not_change_input=True)

        if method == "gf" and enable_distortion_detect:
            pc.build_subnets(positions, custom_objects=model_handler.get_custom_objects())

        t_outputs = []
        for p in positions:
            if type(p) == list:
                sub_p = []
                for l in p:
                    sub_p.append(t_model.get_layer("t_"+l).output)
                t_outputs.append(sub_p)
            else:
                t_outputs.append(t_model.get_layer("t_"+p).output)

        g_outputs = []
        for p in positions:
            if type(p) == list:
                sub_p = []
                for l in p:
                    sub_p.append(gmodel.get_layer(l).output)
                g_outputs.append(sub_p)
            else:
                g_outputs.append(gmodel.get_layer(p).output)

        if method == "gf" and enable_distortion_detect:
            subnet_outputs = []
            for subnet, inputs, outputs in pc.subnets:
                ins = []
                for in_ in inputs:
                    ins.append(gmodel.get_layer(in_).output)
 
                outs = []
                for out_ in outputs:
                    outs.append(gmodel.get_layer(out_).output)
                subnet_outputs.append((ins, outs))
            g_outputs.append(subnet_outputs)
       
        tt = tf.keras.Model(t_model.input, [t_model.output]+t_outputs)
        tt.trainable = False
        gg = tf.keras.Model(gmodel.input, [gmodel.output]+g_outputs)
    else:
        tt = None
        gg = gmodel
 
    total_channels = get_total_channels(ordered_groups, gmodel, parser)
    num_target_channels = math.ceil(total_channels * target_ratio)
    if not ret_score:
        print(total_channels, num_target_channels)
    if method == "gf":
        if print_by_pruning:
            max_iters = num_target_channels
        else:
            if enable_distortion_detect:
                max_iters = num_target_channels * period
            else:
                max_iters = (num_target_channels // num_remove + int(num_target_channels % num_remove > 0)) * period
        if not ret_score:
            print(max_iters, min_steps)
        if min_steps > max_iters:
            max_iters = min_steps
    else:
        if min_steps == -1:
            return
        else:
            max_iters = min_steps

    if not ret_score:
        print("FINAL max_iters:", max_iters)

    def validate_func(gmodel_, parser_):
        cmodel_ = parser.cut(gmodel_)
        val_score = validate(cmodel_)
        return val_score

    vfunc = lambda : validate_func(gmodel, parser)

    iteration_based_train(
        dataset,
        gg,
        model_handler,
        max_iters,
        lr_mode=lr_mode,
        teacher=tt,
        with_label=with_label,
        with_distillation=distillation and not label_only,
        callback_before_update=callback_before_update,
        stopping_callback=stopping_callback,
        augment=True,
        n_classes=n_classes,
        eval_steps=eval_steps,
        validate_func=vfunc)

    end_time = time.time()
    
    cmodel = parser.cut(gmodel)
    print(cmodel.count_params())
    val_score = validate(cmodel)
    if not ret_score:
        print("elapsed_time:", end_time - start_time)
        logging.info("elapsed_time: "+str(end_time - start_time))
        print(val_score)
        logging.info(val_score)
        profile = model_profiler.model_profiler(cmodel, 1)
        print(profile)
        logging.info(profile)
        tf.keras.models.save_model(cmodel, save_dir+"/"+model_handler.get_name()+postfix+".h5")
        tf.keras.models.save_model(gmodel, save_dir+"/"+model_handler.get_name()+postfix+"_gated.h5")
        with open(save_dir+"/"+model_handler.get_name()+postfix+".json", "w") as f:
            json.dump(positions, f)
        from keras_flops import get_flops
        flops = get_flops(cmodel, batch_size=1)
        print(f"FLOPS: {flops / 10 ** 9:.06} G")
        logging.info(f"FLOPS: {flops / 10 ** 9:.06} G")

        if finetune:
            train(dataset, cmodel, model_handler.get_name()+args.model_prefix+"_finetuned"+postfix, model_handler, run_eagerly=True, save_dir=save_dir)
        return cmodel
    else:
        return val_score


def make_distiller(model, teacher, positions, scale=0.1, model_builder=None):

    custom_object_scope = {
        "SimplePruningGate":SimplePruningGate, "StopGradientLayer":StopGradientLayer
    }
    with keras.utils.custom_object_scope(custom_object_scope):
        t_model = M.add_prefix(teacher, "t_", not_change_input=True)

    if len(positions) > 0:
        t_outputs = []
        for p in positions:
            if type(p) == list:
                sub_p = []
                for l in p:
                    sub_p.append(t_model.get_layer("t_"+l).output)
                t_outputs.append(sub_p)
            else:
                t_outputs.append(t_model.get_layer("t_"+p).output)

        g_outputs = []
        for p in positions:
            if type(p) == list:
                sub_p = []
                for l in p:
                    sub_p.append(model.get_layer(l).output)
                g_outputs.append(sub_p)
            else:
                g_outputs.append(model.get_layer(p).output)

        tt = tf.keras.Model(t_model.input, [t_model.output]+t_outputs)
    else:
        tt = t_model
    tt.trainable = False

    toutputs_ = tt(model.input)
    if type(toutputs_) != list:
        toutputs_ = [toutputs_]

    if model_builder is None:
        new_model = tf.keras.Model(model.input, [model.output]+toutputs_)
    else:
        new_model = model_builder(model.input, [model.output]+toutputs_)

    if len(positions) > 0:
        for s, t in zip(g_outputs, toutputs_[1:]):
            if type(s) == list:
                temp = None
                for s_, t_ in zip(s, t):
                    if temp is None:
                        temp = tf.reduce_mean(tf.keras.losses.mean_squared_error(t_, s_)*scale)
                    else:
                        temp += tf.reduce_mean(tf.keras.losses.mean_squared_error(t_, s_)*scale)
                temp /= len(s)
                new_model.add_loss(temp)
            else:
                new_model.add_loss(tf.reduce_mean(tf.keras.losses.mean_squared_error(t, s)*scale))

    new_model.add_loss(tf.reduce_mean(tf.keras.losses.kl_divergence(model.output, toutputs_[0])*scale))

    for layer in tt.layers:
        layer.trainable = False

    for layer in model.layers:
        layer.trainable = True
        if layer.__class__ == SimplePruningGate:
            #layer.trainable = False
            layer.collecting = False
            layer.data_collecting = False

    return new_model

def run():
    parser = argparse.ArgumentParser(description='CIFAR100 ', add_help=False)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None, help='model')
    parser.add_argument('--model_path', type=str, default=None, help='model')
    parser.add_argument('--model_path2', type=str, default=None, help='model')
    parser.add_argument('--model_name', type=str, default=None, help='model')
    parser.add_argument('--model_prefix', type=str, default="", help='model')
    parser.add_argument('--mode', type=str, default="test", help='model')
    parser.add_argument('--position_mode', type=str, help='model')
    parser.add_argument('--with_label', action='store_true')
    parser.add_argument('--label_only', action='store_true')
    parser.add_argument('--distillation', action='store_true')
    parser.add_argument('--fully_random', action='store_true')
    parser.add_argument('--method', type=str, default="gf", help="method")
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--num_remove', type=int, default=500, help='model')
    parser.add_argument('--enable_distortion_detect', action='store_true')
    parser.add_argument('--norm_update', action='store_true')
    parser.add_argument('--print_by_pruning', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=-1, help='model')
    parser.add_argument('--num_blocks', type=int, default=5, help='model')
    parser.add_argument('--epochs', type=int, default=None, help='model')
    parser.add_argument('--period', type=int, default=25, help='model')
    parser.add_argument('--min_steps', type=int, default=-1, help='model')
    parser.add_argument('--lr_mode', type=int, default=0, help='model')
    parser.add_argument('--droprate', type=float, default=0, help='model')
    parser.add_argument('--target_ratio', type=float, default=0.5, help='model')
    parser.add_argument('--save_steps', type=int, default=-1, help='model')
    parser.add_argument('--log_file', type=str, default=None, help="method")
    args = parser.parse_args()

    if args.position_mode.isdigit():
        args.position_mode = int(args.position_mode)

    import json
    with open("args.log", "w") as file_:
        json.dump(vars(args), file_)
        if "NO_HOROVOD" not in os.environ or hvd.rank() == 0:
            print(vars(args))

    if args.config is not None:
        with open(args.config, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        for key in config:
            if hasattr(args, key):
                if "NO_HOROVOD" not in os.environ or hvd.rank() == 0:
                    print("%s is loaded as %s." % (key, config[key]))
                if config[key] in ["true", "false"]: # boolean handling.
                    config[key] = config[key] == "true"
                setattr(args, key, config[key])

    if args.label_only:
        args.with_label = True

    if not args.distillation:
        args.with_label = True

    # Initialize the current folder for experiment
    save_dir = os.path.join(os.getcwd(), "saved_models")
    if os.path.exists(save_dir) and not args.overwrite:
        raise ValueError("`save_models` is not empty!")
    elif not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(asctime)s ] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.info(vars(args))

    model_handler = get_model_handler(args.model_name)

    if args.dataset == "cifar100":
        n_classes = 100
    elif args.dataset == "caltech_birds2011":
        n_classes = 200
    elif args.dataset == "imagenet2012":
        n_classes = 1000
    elif args.dataset == "oxford_iiit_pet":
        n_classes = 37
    elif args.dataset == "cars196":
        n_classes = 196
    elif args.dataset == "stanford_dogs":
        n_classes = 120

    batch_size = model_handler.get_batch_size(args.dataset)
    method = args.method
    dataset = args.dataset
    if args.mode == "test":

        model = model_path_based_load(args.dataset, args.model_path, model_handler)
        model = add_augmentation(model, model_handler.width, train_batch_size=batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope)
        tf.keras.utils.plot_model(model, "tested_model.pdf", expand_nested=True)
        model_handler.compile(model, run_eagerly=False)
        (_, _, test_data_gen), (iters, iters_val) = load_dataset(dataset, model_handler, n_classes=n_classes)
        acc = model.evaluate(test_data_gen, verbose=1)[1]

        from keras_flops import get_flops
        flops = get_flops(model, batch_size=1)
        print(f"FLOPS: {flops / 10 ** 9:.06} G")
        print(model.summary())

        from profile import measure

        tf.keras.backend.set_floatx("float32")
        model = change_dtype(model, "float32", custom_objects=custom_object_scope, distill_set=None)
        model_handler.compile(model, run_eagerly=False)

        x = measure(model, "onnx_cpu")
        print(x)

        tf.keras.utils.plot_model(model, "test.pdf", show_shapes=True)
        acc = model.evaluate(test_data_gen, verbose=1)[1]

    elif args.mode == "cut":
        model = model_path_based_load(args.dataset, args.model_path, model_handler)
        ref = model_path_based_load(args.dataset, args.model_path2, model_handler)
        ref = add_augmentation(ref, model_handler.width, train_batch_size=model_handler.batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)

        gmodel, copied_model, l2g, ordered_groups, torder, parser, _ = add_gates(ref, custom_objects=model_handler.get_custom_objects())
        #copied_model = add_augmentation(copied_model, model_handler.width, train_batch_size=model_handler.batch_size, do_mixup=True, do_cutmix=True, custom_objects=custom_object_scope, update_batch_size=True)
        #gmodel = add_augmentation(gmodel, model_handler.width, train_batch_size=model_handler.batch_size, do_mixup=True, do_cutmix=True, custom_objects=custom_object_scope, update_batch_size=True)

        for layer in model.layers:
            if type(layer) == SimplePruningGate:
                print(np.sum(layer.gates))

        model = parser.cut(model)

        from keras_flops import get_flops
        flops = get_flops(model, batch_size=1)
        print(f"FLOPS: {flops / 10 ** 9:.06} G")
        print(model.summary())

        model_handler.compile(model, run_eagerly=False)
        (_, _, test_data_gen), (iters, iters_val) = load_dataset(dataset, model_handler, n_classes=n_classes)
        print(model.evaluate(test_data_gen, verbose=1)[1])

    elif args.mode == "decompose": # decompose

        model = model_path_based_load(args.dataset, args.model_path, model_handler)
        model = unfold(model)
        sum_params = 0
        cnt = 0
        for layer in model.layers:
            if "Conv2D" == layer.__class__.__name__:
                sum_params += np.prod(layer.get_weights()[0].shape)
                cnt += 1
        avg_params = float(sum_params) / cnt
        targets = []
        for layer in model.layers:
            if "Conv2D" == layer.__class__.__name__:
                if np.prod(layer.get_weights()[0].shape) > avg_params:
                    targets.append((layer.name, 0.25))

        cmodel = lowrank.decompose(model, targets, custom_objects=custom_object_scope)[0]
        cmodel_filename = "decomposed_" + os.path.basename(args.model_path)
        tf.keras.models.save_model(cmodel, save_dir + "/" + cmodel_filename)
        
    elif args.mode == "train": # train
        model = model_handler.get_model(dataset, n_classes=n_classes)
        if config["use_amp"]:
            tf.keras.backend.set_floatx("float16")
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            model = change_dtype(model, mixed_precision.global_policy(), custom_objects=custom_object_scope, distill_set=None)
        model = add_augmentation(model, model_handler.width, train_batch_size=batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope)
        config["mode"] = "train"
        train(dataset, model, model_handler.get_name()+args.model_prefix, model_handler, run_eagerly=False, n_classes=n_classes, save_dir=save_dir, conf=config)

    elif args.mode == "hpo": # train
        model = model_handler.get_model(dataset, n_classes=n_classes)
        if config["use_amp"]:
            tf.keras.backend.set_floatx("float16")
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            model = change_dtype(model, mixed_precision.global_policy(), custom_objects=custom_object_scope, distill_set=None)
        model = add_augmentation(model, model_handler.width, train_batch_size=batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope)

        from ray import tune
        from ray.tune.suggest.bayesopt import BayesOptSearch
        from ray.tune.suggest import ConcurrencyLimiter

        algo = BayesOptSearch()
        algo = ConcurrencyLimiter(algo, max_concurrent=4)

        def training_function(config_):
            hvd.init()
            if "batch_size" in config_:
                model_handler.batch_size = config_["batch_size"]
            train(dataset, model, model_handler.get_name()+args.model_prefix, model_handler, run_eagerly=True, n_classes=n_classes, save_dir=save_dir, conf=config_)
            tune.report(test=1, rank=hvd.rank())

        trainable = DistributedTrainableCreator(
                training_function, num_workers=4, use_gpu=True)

        config_ray = copy.deepcopy(config)
        config_ray["initial_lr"] = tune.uniform(0.0001, 0.1)
        config_ray["decay_epcohs"] = tune.uniform(1.0, 4.0)
        config_ray["batch_size"] = tune.randint(2, 32)
        config_ray["t_mul"] = tune.uniform(0.5, 4.0)
        config_ray["m_mul"] = tune.uniform(0.5, 4.0)

        results = tune.run(trainable, config=config, name="horovod", metric="mean_loss", mode="min", search_alg=algo)
        print(results.best_config)

    elif args.mode == "finetune": # train
        save_dir = save_dir+"/finetuned_models"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if config is not None and config["grad_accum_steps"] > 1:
            model_builder = lambda x, y: GAModel(
                config["use_amp"],
                config["hvd_fp16_compression"],
                config["grad_clip_norm"],
                config["grad_accum_steps"],
                x, y
                )
        else:
            model_builder = None

        model = model_path_based_load(args.dataset, args.model_path, model_handler)

        tf.keras.utils.plot_model(model, "ftarget.pdf", show_shapes=True)

        if args.model_path2 is not None or args.distillation:
            # position_mode must be str
            import json
            with open(args.position_mode, "r") as f: 
                positions = json.load(f)
            position_set = set(positions)
        else:
            position_set = None

        if config["use_amp"]:
            tf.keras.backend.set_floatx("float16")
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print(model)
            model = change_dtype(model, mixed_precision.global_policy(), custom_objects=custom_object_scope, distill_set=position_set)

        if args.droprate > 0:
            print("droprate is changed to ", args.droprate)
            for layer in model.layers:
                if layer.__class__.__name__ == "DropPath":
                    layer.drop_path_prob = args.droprate

        if args.model_path2 is not None or args.distillation:
            print("distillation!")
            teacher = model_path_based_load(args.dataset, args.model_path2, model_handler)
            if config["use_amp"]:
                teacher = change_dtype(teacher, mixed_precision.global_policy(), custom_objects=custom_object_scope, distill_set=position_set)

            teacher = unfold(teacher)
            teacher = add_augmentation(teacher, model_handler.width, train_batch_size=batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)

            model = add_augmentation(model, model_handler.width, train_batch_size=batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)
            model = make_distiller(model, teacher, positions=positions, scale=0.1, model_builder=model_builder)

            if args.with_label:
                config["mode"] = "distillation"
            else:
                config["mode"] = "distillation_label_free"

        else:
            model = add_augmentation(model, model_handler.width, train_batch_size=model_handler.batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)
            config["mode"] = "finetune"

        #tf.keras.backend.set_floatx("float64")
        #model = change_dtype(model, "float64", custom_objects=custom_object_scope, distill_set=position_set)

        train(dataset, model, model_handler.get_name()+args.model_prefix, model_handler, run_eagerly=True, n_classes=n_classes, save_dir=save_dir, conf=config, epochs_=args.epochs)

        if hvd.rank() == 0:
            model_filename = "finetuned_" + os.path.basename(args.model_path)
            tf.keras.models.save_model(model, save_dir + "/" + model_filename, include_optimizer=False)

    elif args.mode == "cut": # train

        model = model_path_based_load(args.dataset, args.model_path, model_handler)
        if args.model_path2 is not None:
            teacher = model_path_based_load(args.dataset, args.model_path2, model_handler)
            # model must be a gated model.
            gmodel, copied_model, l2g, ordered_groups, torder, parser, _ = add_gates(teacher, custom_objects=model_handler.get_custom_objects())

        else:
            raise ValueError("model_path2 should be defined.")

        cmodel = parser.cut(model)
        save_dir = save_dir+"/finetuned_models"
        cmodel_filename = "cmodel_" + os.path.basename(args.model_path)
        tf.keras.models.save_model(cmodel, save_dir + "/" + cmodel_filename)

    elif args.mode == "prune":

        iter_dir = save_dir+"/pruning_steps"
        if not os.path.exists(iter_dir):
            os.mkdir(iter_dir)

        if args.model_path is None and args.dataset != "imagenet2012":
            from pathset import paths
            model_path = paths[args.dataset][args.model_name]
        else:
            model_path = args.model_path

        model = model_path_based_load(args.dataset, model_path, model_handler)
        if method not in ["curl", "hrank"]:
            model = add_augmentation(model, model_handler.width, train_batch_size=batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_object_scope, update_batch_size=True)

        prune(
            dataset,
            model,
            model_handler,
            position_mode=args.position_mode,
            with_label=args.with_label,
            label_only=args.label_only,
            distillation=args.distillation,
            fully_random=args.fully_random,
            num_remove=args.num_remove,
            enable_distortion_detect=args.enable_distortion_detect,
            norm_update=args.norm_update,
            print_by_pruning=args.print_by_pruning,
            target_ratio=args.target_ratio,
            method=method,
            finetune=args.finetune,
            period=args.period,
            n_classes=n_classes,
            num_blocks=args.num_blocks,
            save_dir=save_dir,
            min_steps=args.min_steps,
            save_steps=args.save_steps,
            model_path2=args.model_path2,
            backup_args=vars(args),
            eval_steps=args.eval_steps,
            lr_mode=args.lr_mode,
            config=config)

    elif args.mode == "find":

        if args.model_path is None and args.dataset != "imagenet2012":
            from pathset import paths
            model_path = paths[args.dataset][args.model_name]
        else:
            model_path = args.model_path

        model = model_path_based_load(args.dataset, model_path, model_handler)

        from search_positions import find_positions
        _pos, best_pos  = find_positions(
            prune,
            dataset,
            model,
            model_handler,
            position_mode=args.position_mode,
            with_label=args.with_label,
            label_only=args.label_only,
            distillation=args.distillation,
            num_remove=args.num_remove,
            enable_distortion_detect=args.enable_distortion_detect,
            target_ratio=args.target_ratio,
            n_classes=n_classes,
            period=args.period,
            num_blocks=args.num_blocks,
            min_steps=args.min_steps)

        import json
        pos_filename = model_handler.get_name()+"_"+dataset+"_"+str(args.with_label)+str(args.num_blocks)+"_"+str(args.target_ratio)+"_"+str(args.num_remove)+".json"
        print(best_pos)
        with open(pos_filename, "w") as f:
            json.dump({"data":best_pos}, f)

if __name__ == "__main__":
    run()
