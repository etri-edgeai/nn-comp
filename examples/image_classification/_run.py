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
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
import efficientnet.tfkeras
import os
import argparse
import cv2

import numpy as np
from tensorflow import keras

from keras_flops import get_flops

from sklearn.model_selection import StratifiedShuffleSplit

from datagen import DataGenerator
from numba import njit

# constants
n_classes = 100
epochs = 50

def load_npz(path):
    data = np.load(path)
    return (data["train"], data["trainy"]), (data["test"], data["testy"])

def load_data(dataset, model_handler, training_augment=True, batch_size=-1):

    dim = (model_handler.get_shape(dataset)[0], model_handler.get_shape(dataset)[1])
    preprocess_func = model_handler.preprocess_func
    if hasattr(model_handler, "batch_preprocess_func"):
        batch_pf = model_handler.batch_preprocess_func
    else:
        batch_pf = None

    if batch_size == -1:
        batch_size_ = model_handler.get_batch_size(dataset)
    else:
        batch_size_ = batch_size

    if dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    elif dataset == "cub":
        dir_ = "/ssd_data2/jongryul/CUB_200_2011/data.npz"
        (x_train, y_train), (x_test, y_test) = load_npz(dir_)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
    elif dataset == "imagenet":
        pass
    else:
        raise NotImplementedError("No Dataset")

    if dataset != "imagenet":
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)

    if dataset == "cifar100":

        num_classes = 100
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=123)

        for train_index, val_index in sss.split(x_train, y_train):
            x_train_data, x_val_data = x_train[train_index], x_train[val_index]
            y_train_data, y_val_data = y_train[train_index], y_train[val_index]

        print("Number of training samples: ", x_train_data.shape[0])
        print("Number of validation samples: ", x_val_data.shape[0])

        train_data_generator = DataGenerator(
            x_train_data,
            y_train_data,
            batch_size=batch_size_,
            augment=training_augment,
            dim=dim,
            n_classes=n_classes,
            preprocess_func=preprocess_func,
            batch_preprocess_func=batch_pf)
        valid_data_generator = DataGenerator(
            x_val_data,
            y_val_data,
            batch_size=batch_size_,
            augment=False,
            dim=dim,
            n_classes=n_classes,
            preprocess_func=preprocess_func,
            batch_preprocess_func=batch_pf)
        test_data_generator = DataGenerator(
            x_test,
            y_test,
            batch_size=batch_size_,
            augment=False,
            dim=dim,
            n_classes=n_classes,
            preprocess_func=preprocess_func,
            batch_preprocess_func=batch_pf)

    elif dataset == "imagenet":
        import sys
        sys.path.insert(0, "/home/jongryul/work/keras_imagenet")

        from utils.dataset import get_dataset
        dataset_dir = "/ssd_data2/jongryul/tf"
        train_data_generator= get_dataset(dataset_dir, 'train', batch_size_)
        valid_data_generator = get_dataset(dataset_dir, 'validation', batch_size_)
        test_data_generator = get_dataset(dataset_dir, 'validation', batch_size_)

    else:

        train_data_generator = DataGenerator(
            x_train,
            y_train,
            batch_size=batch_size_,
            augment=training_augment,
            dim=dim,
            n_classes=n_classes,
            preprocess_func=preprocess_func,
            batch_preprocess_func=batch_pf)
        valid_data_generator = DataGenerator(
            x_test,
            y_test,
            batch_size=batch_size_,
            augment=False,
            dim=dim,
            n_classes=n_classes,
            preprocess_func=preprocess_func,
            batch_preprocess_func=batch_pf)
        test_data_generator = DataGenerator(
            x_test,
            y_test,
            batch_size=batch_size_,
            augment=False,
            dim=dim,
            n_classes=n_classes,
            preprocess_func=preprocess_func,
            batch_preprocess_func=batch_pf)
 
    return train_data_generator, valid_data_generator, test_data_generator


def train(dataset, model, model_name, model_handler, run_eagerly=False, callbacks=None, is_training=True, augment=True, exclude_val=False, dir_="saved_models"):

    train_data_generator, valid_data_generator, test_data_generator = load_data(dataset, model_handler, training_augment=False)

    if is_training and hasattr(model_handler, "get_train_epochs"):
        epochs_ = model_handler.get_train_epochs()
    else:
        epochs_ = epochs

    if callbacks is None:   
        callbacks = []

    if dataset == "imagenet":
        iters = 1281167 // model_handler.batch_size
    else:
        iters = len(train_data_generator)
    callbacks_ = model_handler.get_callbacks(iters)
    model_handler.compile(model, run_eagerly=run_eagerly)

    print(model.count_params())

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), dir_)
    model_name_ = '%s_model.{epoch:03d}.h5' % model_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name_)

    mchk = keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_accuracy",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
    )

    if exclude_val:
        model_history = model.fit_generator(train_data_generator,
                                        callbacks=[mchk]+callbacks+callbacks_,
                                        verbose=1,
                                        epochs=epochs_,
                                        steps=iters)
    else:
        model_history = model.fit_generator(train_data_generator,
                                        validation_data=valid_data_generator,
                                        callbacks=[mchk]+callbacks+callbacks_,
                                        verbose=1,
                                        epochs=epochs_,
                                        steps=iters)

@njit
def find_min(score, gates_info, n_channels_group, n_removed_group, ngates):
    idx = 0
    min_score = score[0]

    local_base = 0
    for gidx in range(len(gates_info)):

        if n_channels_group[gidx] - n_removed_group[gidx] < 2.0: # min channels.
            local_base += gates_info[gidx]
            continue

        for lidx in range(gates_info[gidx]):
            if score[gidx * ngates + lidx] == -1.0:
                continue

            if min_score > score[local_base+lidx]:
                min_score = score[local_base+lidx]
                idx = local_base + lidx

        local_base += gates_info[gidx]

    return idx


def prune(dataset, model, model_handler, position_mode, with_label=False, label_only=False, fully_random=False, num_remove=1, target_ratio=0.5, curl=False, finetune=False):

    print(get_flops(model, batch_size=1))
    _, _, test_data_gen = load_data(dataset, model_handler, batch_size=model_handler.batch_size)
    def validate(model_):
        model_handler.compile(model_, run_eagerly=True)
        if dataset == "imagenet":
            return model_.evaluate(test_data_gen, verbose=1, steps=50000//model_handler.batch_size)[1]
        else:
            return model_.evaluate(test_data_gen, verbose=1)[1]

    from nncompress.backend.tensorflow_.transformation import parse, inject, cut
    from group_fisher import make_group_fisher
    gmodel, copied_model, blocks, ordered_groups, parser, pc = make_group_fisher(
        model,
        model_handler,
        model_handler.get_batch_size(dataset),
        target_ratio=0.5,
        target_gidx=-1,
        target_idx=-1,
        enable_norm=True,
        num_remove=num_remove,
        fully_random=fully_random,
        custom_objects=model_handler.get_custom_objects(),
        logging=False)

    from nncompress.backend.tensorflow_ import SimplePruningGate, DifferentiableGate
    from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer
    from nncompress import backend as M
    from dc import init_gate
    custom_object_scope = {
        "SimplePruningGate":SimplePruningGate, "StopGradientLayer":StopGradientLayer
    }
    if model_handler.get_custom_objects() is not None:
        for key, val in model_handler.get_custom_objects().items():
            custom_object_scope[key] = val
    with keras.utils.custom_object_scope(custom_object_scope):
        t_model = M.add_prefix(copied_model, "t_")

    convs = [
        layer.name for layer in copied_model.layers if "Conv2D" in layer.__class__.__name__
    ]

    all_ = [
        layer.name for layer in copied_model.layers
    ]


    if position_mode == 0: # all
        positions = convs
    elif position_mode == 4: # 1x1 conv
        positions = []
        for c in convs:
            if gmodel.get_layer(c).kernel_size == (1,1):
                positions.append(c)
    elif position_mode == 1: # joints
        positions = [
            b[-1][1] for b in blocks
        ]
    elif position_mode == 2: # random
        positions = [
            all_[int(random.random() * len(all_))] for i in range(len(blocks))
        ]

    elif position_mode == 3: # cut
        positions  = []
        for b in blocks:
            g = b[-1][0]
            des = parser.first_common_descendant(list(g), convs, False)

            des_g = None
            for g_, idx in ordered_groups:
                if des in g_:
                    des_g = g_
                    break
        
            if des_g is not None:
                sub_positions = []
                positions.append(sub_positions)
                for l in des_g:
                    sub_positions.append(l)
            else:
                positions.append(des) # maybe the last transforming layer.

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
   
    tt = tf.keras.Model(t_model.input, [t_model.output]+t_outputs)
    gg = tf.keras.Model(gmodel.input, [gmodel.output]+g_outputs)

    # CURL paper
    if curl:

        backup = model_handler.batch_size
        model_handler.batch_size = 256
        train_data_generator_, _, _ = load_data(dataset, model_handler, training_augment=False)
        model_handler.batch_size = backup

        print("collecting...")
        data = []
        ty = []
        record_id = 1
        for X, y in train_data_generator_:
            if record_id > 1:
                break
            data.append(X)
            ty.append(tt(X)[0])
            record_id += 1

        print("scoring...")
        n_channels = 0
        n_channels_group = [
            0 for _ in range(len(ordered_groups))
        ]
        gates_weights = {}
        gates_info = []
        for gidx, (g, _) in enumerate(ordered_groups):
            
            gates_info.append(gg.get_layer(pc.l2g[g[0]]).gates.numpy().shape[0])
            for g_ in g:
                gate = gg.get_layer(pc.l2g[g_])
                gates_weights[pc.l2g[g_]] = gate.gates.numpy()
                n_channels += gate.gates.shape[0]
        score = [0.0 for _ in range(n_channels)]
        
        for gidx, (g, _) in enumerate(ordered_groups):
            print(gidx)
            gate = gg.get_layer(pc.l2g[g[0]])
            n_channels += gate.gates.shape[0]
            n_channels_group[gidx] = gate.gates.shape[0]
            for lidx in range(gate.gates.shape[0]):
                sum_ = 0.0
                for X, ty_output in zip(data, ty):
                    student_logits = gg(X)
                    sum_ += tf.math.reduce_mean(tf.keras.losses.kl_divergence(student_logits[0], ty_output))
                score[gidx * len(ordered_groups) + lidx] = float(sum_)

        print("pruning...")
        # pruning
        n_removed = 0
        n_removed_group = [
            0 for _ in range(len(ordered_groups))
        ]

        while float(n_removed) / n_channels < target_ratio:
            if n_removed % 100 == 0:
                print(float(n_removed) / n_channels)

            val = find_min(score, gates_info, n_channels_group, n_removed_group, len(gates_info))

            local_base = 0
            min_gidx = -1
            min_lidx = -1
            for gidx, len_ in enumerate(gates_info):
                if val - local_base <= len_-1: # hit
                    min_gidx = gidx
                    min_lidx = val - local_base
                    break
                else:
                    local_base += len_

            assert min_gidx != -1

            min_group, _ = ordered_groups[min_gidx]
            for min_layer in min_group:
                gates_weights[pc.l2g[min_layer]][min_lidx] = 0.0
            score[min_gidx * len(ordered_groups) + min_lidx] = -1.0

            n_removed += 1
            n_removed_group[gidx] += 1

        for key in gates_weights:
            layer = gg.get_layer(key)
            layer.gates.assign(gates_weights[key])

        pruned = True

    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    for layer in gg.layers:
        if layer.__class__ == SimplePruningGate:
            layer.collecting = True
            layer.grad_holder = []

    train_data_generator, valid_data_generator, test_data_generator = load_data(dataset, model_handler, training_augment=True)
    min_steps = 20
    step = 0
    while True:
        done = False
        for X, y in train_data_generator:
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            teacher_logits = tt(X)

            if pc.continue_pruning and not curl:
                for layer in gg.layers:
                    if layer.__class__ == SimplePruningGate:
                        layer.trainable = True

                with tf.GradientTape() as tape:
                    student_logits = gg(X)
                    if not label_only:
                        loss = None
                        for i in range(len(t_outputs)):
                            if type(t_outputs[i]) == list:
                                temp = None
                                for j in range(len(t_outputs[i])):
                                    if temp is None:
                                        temp = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(student_logits[i+1][j], teacher_logits[i+1][j]))
                                    else:
                                        temp += tf.math.reduce_mean(tf.keras.losses.mean_squared_error(student_logits[i+1][j], teacher_logits[i+1][j]))
                                temp /= len(t_outputs[i])
                            else:
                                temp = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(student_logits[i+1], teacher_logits[i+1]))

                            if loss is None:
                                loss = temp
                            else:
                                loss += temp

                        loss += tf.math.reduce_mean(tf.keras.losses.kl_divergence(student_logits[0], teacher_logits[0]))

                        if with_label:
                            loss += tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(student_logits[0], y))

                    if label_only:
                        loss = tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(student_logits[0], y))

                gradients = tape.gradient(loss, gg.trainable_variables)
                pruned = pc.on_train_batch_end(None)

            for layer in gg.layers:
                if layer.__class__ == SimplePruningGate:
                    layer.trainable = False

            with tf.GradientTape() as tape:
                student_logits = gg(X)
                if not label_only:
                    loss = None
                    for i in range(len(t_outputs)):
                        if type(t_outputs[i]) == list:
                            temp = None
                            for j in range(len(t_outputs[i])):
                                if temp is None:
                                    temp = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(student_logits[i+1][j], teacher_logits[i+1][j]))
                                else:
                                    temp += tf.math.reduce_mean(tf.keras.losses.mean_squared_error(student_logits[i+1][j], teacher_logits[i+1][j]))
                            temp /= len(t_outputs[i])
                        else:
                            temp = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(student_logits[i+1], teacher_logits[i+1]))

                        if loss is None:
                            loss = temp
                        else:
                            loss += temp

                    loss += tf.math.reduce_mean(tf.keras.losses.kl_divergence(student_logits[0], teacher_logits[0]))

                    if with_label:
                        loss += tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(student_logits[0], y))

                if label_only:
                    loss = tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(student_logits[0], y))

            gradients = tape.gradient(loss, gg.trainable_variables)
            optimizer.apply_gradients(zip(gradients, gg.trainable_variables))

            if pruned:
                for layer in gg.layers:
                    if layer.__class__ == SimplePruningGate:
                        layer.grad_holder = []

            step += 1
            if not pc.continue_pruning and min_steps <= step:
                done = True
                break

        if curl and min_steps <= step:
            break
        elif done:
            break
        else:
            train_data_generator.on_epoch_end()

    cmodel = parser.cut(gmodel)

    print(get_flops(cmodel, batch_size=1))
    print(cmodel.count_params())
    print(validate(cmodel))
    import model_profiler
    profile = model_profiler.model_profiler(cmodel, 1)
    print(profile)
    postfix = "_"+str(position_mode)+"_"+dataset+"_"+str(with_label)+"_"+str(label_only)+"_"+str(curl)
    tf.keras.models.save_model(cmodel, "compressed_models/"+model_handler.get_name()+postfix+".h5")

    if finetune:
        train(dataset, cmodel, model_handler.get_name()+postfix+"_finetuned", model_handler, run_eagerly=True, dir_="finetuned_models")


def run():
    parser = argparse.ArgumentParser(description='CIFAR100 ', add_help=False)
    parser.add_argument('--dataset', type=str, default=None, help='model')
    parser.add_argument('--model_path', type=str, default=None, help='model')
    parser.add_argument('--model_name', type=str, default=None, help='model')
    parser.add_argument('--model_prefix', type=str, default="", help='model')
    parser.add_argument('--mode', type=str, default="test", help='model')
    parser.add_argument('--position_mode', type=int, help='model')
    parser.add_argument('--with_label', action='store_true')
    parser.add_argument('--label_only', action='store_true')
    parser.add_argument('--fully_random', action='store_true')
    parser.add_argument('--curl', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--num_remove', type=int, default=100, help='model')
    parser.add_argument('--target_ratio', type=float, default=0.5, help='model')
    args = parser.parse_args()

    from loader import get_model_handler
    model_handler = get_model_handler(args.model_name)

    if args.dataset == "cifar100":
        n_classes = 100
    elif args.dataset == "cub":
        n_classes = 200
    elif args.dataset == "imagenet":
        n_classes = 1000

    dataset = args.dataset
    if args.mode == "test":
        if hasattr(model_handler, "get_custom_objects"):
            model = tf.keras.models.load_model(args.model_path, custom_objects=model_handler.get_custom_objects())
        else:
            model = tf.keras.models.load_model(args.model_path)
        _, _, test_data_gen = load_data(dataset, model_handler)
        print(model.evaluate(test_data_gen, verbose=1)[1])
    elif args.mode == "train": # train
        model = model_handler.get_model(dataset, n_classes=n_classes)
        train(dataset, model, model_handler.get_name()+args.model_prefix, model_handler, run_eagerly=True)
    elif args.mode == "prune":
        model = tf.keras.models.load_model(args.model_path, custom_objects=model_handler.get_custom_objects())
        prune(dataset, model, model_handler, position_mode=args.position_mode, with_label=args.with_label, label_only=args.label_only, fully_random=args.fully_random, num_remove=args.num_remove, target_ratio=args.target_ratio, curl=args.curl, finetune=args.finetune)

if __name__ == "__main__":
    run()
