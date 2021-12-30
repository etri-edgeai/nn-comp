# coding: utf-8

from __future__ import print_function
import tensorflow as tf
tf.random.set_seed(2)
import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
import imgaug as ia
ia.seed(1234)
import os
import argparse
import cv2
import math

import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import model_profiler

from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import StopGradientLayer
from nncompress import backend as M

from curl import apply_curl
from hrank import apply_hrank
from group_fisher import make_group_fisher, add_gates, prune_step, compute_positions, get_num_all_channels
from loader import get_model_handler

from train import load_data, train, iteration_based_train

def get_total_channels(groups, model):
    total = 0
    for g in groups:
        layer = model.get_layer(g[0][0])
        total += layer.filters
    return total

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
    print_by_pruning=False,
    target_ratio=0.5,
    min_steps=-1,
    method="gf",
    finetune=False,
    period=25,
    n_classes=100,
    num_blocks=3,
    save_dir="",
    save_steps=-1):

    if type(position_mode) != int:
        pos_str = "custom"
    else:
        pos_str = str(position_mode)

    if label_only:
        postfix = "_"+pos_str+"_"+dataset+"_"+str(with_label)+"_gf_label_only_"+str(target_ratio)
    elif method == "curl":
        postfix = "_"+pos_str+"_"+dataset+"_"+str(with_label)+"_curl_"+str(target_ratio)
    elif method == "hrank":
        postfix = "_"+pos_str+"_"+dataset+"_"+str(with_label)+"_hrank_"+str(target_ratio)
    elif method == "gf":
        postfix = "_"+pos_str+"_"+dataset+"_"+str(with_label)+"_gf"+"_"+str(num_blocks)+"_"+str(target_ratio)
    else:
        raise NotImplementedError("Method name error!")

    _, _, test_data_gen = load_data(dataset, model_handler, batch_size=model_handler.batch_size, n_classes=n_classes)
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
        train_data_generator_, _, _ = load_data(dataset, model_handler, training_augment=False, n_classes=n_classes)
        model_handler.batch_size = backup
        apply_curl(train_data_generator_, copied_model, gmodel, ordered_groups, l2g, parser, target_ratio, save_dir+"/pruning_steps", model_handler.get_name()+postfix, save_steps=save_steps)

    elif method == "hrank":

        gmodel, copied_model, l2g, ordered_groups, torder, parser, _ = add_gates(model, custom_objects=model_handler.get_custom_objects())
        backup = model_handler.batch_size
        model_handler.batch_size = 256
        train_data_generator_, _, _ = load_data(dataset, model_handler, training_augment=False, n_classes=n_classes)
        model_handler.batch_size = backup
        apply_hrank(train_data_generator_, copied_model, gmodel, ordered_groups, l2g, parser, target_ratio, save_dir+"/pruning_steps", model_handler.get_name()+postfix, save_steps=save_steps)

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
            fully_random=fully_random,
            custom_objects=model_handler.get_custom_objects(),
            save_steps=save_steps,
            save_prefix=model_handler.get_name()+postfix,
            save_dir=save_dir+"/pruning_steps",
            logging=False)

    def callback_before_update(idx, global_step, X, model_, teacher_logits, y, pbar):
        if method in ["curl", "hrank"]:
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
            if global_step >= min_steps and (method in ["curl", "hrank"] or not pc.continue_pruning):
                return True
            else:
                return False
        else:
            return not pc.continue_pruning

    if type(position_mode) == int:
        positions = compute_positions(
            copied_model, ordered_groups, torder, parser, position_mode, num_blocks)
    else:
        positions = position_mode

    profile = model_profiler.model_profiler(copied_model, 1)
    print(profile)
    cmodel = parser.cut(gmodel)
    print(cmodel.count_params())
    #print(validate(cmodel))

    if distillation:
        custom_object_scope = {
            "SimplePruningGate":SimplePruningGate, "StopGradientLayer":StopGradientLayer
        }
        if model_handler.get_custom_objects() is not None:
            for key, val in model_handler.get_custom_objects().items():
                custom_object_scope[key] = val
        with keras.utils.custom_object_scope(custom_object_scope):
            t_model = M.add_prefix(copied_model, "t_")

        if method == "gf":
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

        if method == "gf":
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
  
    #total_channels = get_num_all_channels(pc.gate_groups)
    total_channels = get_total_channels(ordered_groups, gmodel)
    num_target_channels = math.ceil(total_channels * target_ratio)
    if print_by_pruning:
        max_iters = num_target_channels
    else:
        if enable_distortion_detect:
            max_iters = num_target_channels * period
        else:
            max_iters = (num_target_channels // num_remove + int(num_target_channels % num_remove > 0)) * period

    if min_steps > max_iters:
        max_iters = min_steps
    iteration_based_train(
        dataset,
        gg,
        model_handler,
        max_iters,
        teacher=tt,
        with_label=with_label,
        with_distillation=distillation and not label_only,
        callback_before_update=callback_before_update,
        stopping_callback=stopping_callback,
        augment=True,
        n_classes=n_classes)

    cmodel = parser.cut(gmodel)
    print(cmodel.count_params())
    print(validate(cmodel))
    profile = model_profiler.model_profiler(cmodel, 1)
    print(profile)
    tf.keras.models.save_model(cmodel, save_dir+"/"+model_handler.get_name()+postfix+".h5")

    if finetune:
        train(dataset, cmodel, model_handler.get_name()+args.model_prefix+"_finetuned"+postfix, model_handler, run_eagerly=True, save_dir=save_dir)
    return cmodel


def run():
    parser = argparse.ArgumentParser(description='CIFAR100 ', add_help=False)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None, help='model')
    parser.add_argument('--model_path', type=str, default=None, help='model')
    parser.add_argument('--model_name', type=str, default=None, help='model')
    parser.add_argument('--model_prefix', type=str, default="", help='model')
    parser.add_argument('--mode', type=str, default="test", help='model')
    parser.add_argument('--position_mode', type=int, help='model')
    parser.add_argument('--with_label', action='store_true')
    parser.add_argument('--label_only', action='store_true')
    parser.add_argument('--distillation', action='store_true')
    parser.add_argument('--fully_random', action='store_true')
    parser.add_argument('--method', type=str, default="gf", help="method")
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--num_remove', type=int, default=500, help='model')
    parser.add_argument('--enable_distortion_detect', action='store_true')
    parser.add_argument('--print_by_pruning', action='store_true')
    parser.add_argument('--num_blocks', type=int, default=5, help='model')
    parser.add_argument('--period', type=int, default=25, help='model')
    parser.add_argument('--min_steps', type=int, default=-1, help='model')
    parser.add_argument('--target_ratio', type=float, default=0.5, help='model')
    parser.add_argument('--save_steps', type=int, default=-1, help='model')
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        for key in config:
            if hasattr(args, key):
                print("%s is loaded as %s." % (key, config[key]))
                if config[key] in ["true", "false"]: # boolean handling.
                    config[key] = config[key] == "true"
                setattr(args, key, config[key])
            else:
                raise NotImplementedError("Option Error. Check your config.")

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

    model_handler = get_model_handler(args.model_name)

    if args.dataset == "cifar100":
        n_classes = 100
    elif args.dataset == "caltech_birds2011":
        n_classes = 200
    elif args.dataset == "imagenet":
        n_classes = 1000
    elif args.dataset == "imagenet2012":
        n_classes = 1000
    elif args.dataset == "oxford_iiit_pet":
        n_classes = 37
    elif args.dataset == "cars196":
        n_classes = 196
    elif args.dataset == "stanford_dogs":
        n_classes = 120

    method = args.method
    dataset = args.dataset
    if args.mode == "test":

        if args.dataset == "imagenet2012" and args.model_path is None:
            model = model_handler.get_model(dataset="imagenet2012")
        else:
            if hasattr(model_handler, "get_custom_objects"):
                model = tf.keras.models.load_model(args.model_path, custom_objects=model_handler.get_custom_objects())
            else:
                model = tf.keras.models.load_model(args.model_path)

        model_handler.compile(model)
        _, _, test_data_gen = load_data(dataset, model_handler, n_classes=n_classes)
        print(model.evaluate(test_data_gen, verbose=1)[1])
    elif args.mode == "train": # train
        model = model_handler.get_model(dataset, n_classes=n_classes)
        train(dataset, model, model_handler.get_name()+args.model_prefix, model_handler, run_eagerly=True, n_classes=n_classes, save_dir=save_dir)

    elif args.mode == "finetune": # train
        save_dir = save_dir+"/finetuned_models"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if hasattr(model_handler, "get_custom_objects"):
            model = tf.keras.models.load_model(args.model_path, custom_objects=model_handler.get_custom_objects())
        else:
            model = tf.keras.models.load_model(args.model_path)
        train(dataset, model, model_handler.get_name()+args.model_prefix, model_handler, run_eagerly=True, n_classes=n_classes, save_dir=save_dir)
    elif args.mode == "prune":

        iter_dir = save_dir+"/pruning_steps"
        if not os.path.exists(iter_dir):
            os.mkdir(iter_dir)

        if args.model_path is None:
            from pathset import paths
            model_path = paths[args.dataset][args.model_name]
        else:
            model_path = args.model_path

        if args.dataset != "imagenet":
            model = tf.keras.models.load_model(model_path, custom_objects=model_handler.get_custom_objects())
        else:
            model = model_handler.get_model(dataset, n_classes=n_classes)

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
            print_by_pruning=args.print_by_pruning,
            target_ratio=args.target_ratio,
            method=method,
            finetune=args.finetune,
            period=args.period,
            n_classes=n_classes,
            num_blocks=args.num_blocks,
            save_dir=save_dir,
            min_steps=args.min_steps,
            save_steps=args.save_steps)


if __name__ == "__main__":
    run()
