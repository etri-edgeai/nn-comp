import os
import json
import copy
import shutil
import time
from os import listdir
from os.path import isfile, join
from silence_tensorflow import silence_tensorflow
silence_tensorflow()


from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
tf.random.set_seed(2)
import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
import sys
import argparse

tf.config.experimental.set_synchronous_execution(True)
tf.config.experimental.enable_op_determinism()

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for i, p in enumerate(physical_devices):
        tf.config.experimental.set_memory_growth(
            physical_devices[i], True
            )
    tf.config.set_visible_devices(physical_devices[0], 'GPU')

from keras_flops import get_flops
import numpy as np
from tensorflow.keras.optimizers import Adam

from efficientnet.tfkeras import EfficientNetB0

import tf2onnx
import onnxruntime as rt

from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection
from prep import remove_augmentation, add_augmentation

from train import load_dataset
from droppath import DropPath
custom_objects = {
    "SimplePruningGate":SimplePruningGate,
    "StopGradientLayer":StopGradientLayer,
    "DropPath": DropPath
}


BATCH_SIZE_GPU = 1
BATCH_SIZE_ONNX_GPU = 1
BATCH_SIZE_CPU = 1

def tf_convert_onnx(model):
    input_shape = model.input.shape
    spec = (tf.TensorSpec(input_shape, tf.float32, name=model.input.name),)
    output_path = "/tmp/tmp_%d.onnx" % os.getpid()
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]
    return output_path, output_names

def tf_convert_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
    ]
    model_= converter.convert()
    return model_

def measure(model, mode="cpu", batch_size=-1, num_rounds=100):
    model = remove_augmentation(model, custom_objects)
    
    total_t = 0
    if type(model.input) == list:
        input_shape = model.input[0].shape
    else:
        input_shape = list(model.input.shape)
    if input_shape[1] is None:
        input_shape = [None, 224, 224, 3]
    if batch_size == -1:
        if mode == "gpu":
            input_shape[0] = BATCH_SIZE_GPU
        elif mode == "onnx_gpu":
            input_shape[0] = BATCH_SIZE_ONNX_GPU
        else:
            input_shape[0] = BATCH_SIZE_CPU
    else:
        input_shape[0] = batch_size

    if mode == "cpu" and batch_size == -1:
        assert input_shape[0] == BATCH_SIZE_CPU

    tf.keras.backend.clear_session()
    input_shape = tuple(input_shape)
    if "onnx" in mode:
        output_path, output_names = tf_convert_onnx(model)
        if mode == "onnx_cpu":
            providers = ['CPUExecutionProvider']
            DEVICE_NAME = "cpu"
            DEVICE_INDEX = 0
        elif mode == "onnx_gpu":
            providers = [('CUDAExecutionProvider', {"device_id":0})]
            DEVICE_NAME = "cuda"
            DEVICE_INDEX = 0
        else:
            raise NotImplementedError("check your mode: %s" % mode)
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        m = rt.InferenceSession(output_path, sess_options, providers=providers)

        input_data = np.array(np.random.rand(*input_shape), dtype=np.float32)
        x_ortvalue = rt.OrtValue.ortvalue_from_numpy(input_data, DEVICE_NAME, DEVICE_INDEX)
        io_binding = m.io_binding()
        io_binding.bind_input(name=model.input.name, device_type=x_ortvalue.device_name(), device_id=DEVICE_INDEX, element_type=input_data.dtype, shape=x_ortvalue.shape(), buffer_ptr=x_ortvalue.data_ptr())
        io_binding.bind_output(output_names[0])
        for i in range(10):
            #onnx_pred = m.run(output_names, {model.input.name: input_data})
            onnx_pred = m.run_with_iobinding(io_binding)

        for i in range(num_rounds):
            start = timer()
            #onnx_pred = m.run(output_names, {model.input.name: input_data})
            onnx_pred = m.run_with_iobinding(io_binding)
            time_ = float(timer() - start)
            total_t = total_t + time_
        avg_time = (total_t / float(num_rounds))

    elif mode == "gpu":
        # dummy run
        with tf.device("/gpu:0"):
            input_data = tf.convert_to_tensor(np.array(np.random.rand(*input_shape), dtype=np.float32), dtype=tf.float32)
            for i in range(10):
                model(input_data, training=False)

            for i in range(num_rounds):
                start = timer()
                model(input_data, training=False)
                time_ = float(timer() - start)
                total_t = total_t + time_
        avg_time = (total_t / float(num_rounds))

    elif mode == "cpu":
        with tf.device("/cpu:0"):
            input_data = tf.convert_to_tensor(np.array(np.random.rand(*input_shape), dtype=np.float32), dtype=tf.float32)
            for i in range(num_rounds):
                start = timer()
                model(input_data, training=False)
                time_ = float(timer() - start)
                total_t = total_t + time_
        avg_time = (total_t / float(num_rounds))
    elif mode == "tflite":
        tflite_model = tf_convert_tflite(model)
        # Save the model.
        with open('/tmp/tmp_%d.tflite' % os.getpid(), 'wb') as f:
          f.write(tflite_model)
        interpreter = tf.lite.Interpreter(model_path = "tmp.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        input_index = input_details[0]['index']

        total_t = 0
        input_data = np.array(np.random.rand(*input_shape), dtype=np.float32)
        for i in range(num_rounds):
            start = timer()
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            time_ = float(timer() - start)
            total_t = total_t + time_
        avg_time = (total_t / float(num_rounds))

    elif mode == "trt":
        from k2t import t2t_test
        avg_time = t2t_test(model, input_shape[0], num_rounds=num_rounds)
    else:

        raise NotImplementedError("!")

    del input_data
    return avg_time * 1000


def validate(model, model_handler, dataset, sampling_ratio=1.0):
    custom_objects = {
        "SimplePruningGate":SimplePruningGate,
        "StopGradientLayer":StopGradientLayer
    }
    if dataset == "imagenet2012":
        n_classes = 1000
    elif dataset == "cifar100":
        n_classes = 100

    batch_size = model_handler.batch_size
    model = add_augmentation(model, model_handler.width, train_batch_size=batch_size, do_mixup=False, do_cutmix=False, custom_objects=custom_objects)
    (_, _, test_data_generator), (iters, iters_val) = load_dataset(
        dataset,
        model_handler,
        training_augment=False,
        sampling_ratio=sampling_ratio,
        n_classes=n_classes)
    model_handler.compile(model, run_eagerly=False)

    return model.evaluate(test_data_generator, verbose=1)[1]


