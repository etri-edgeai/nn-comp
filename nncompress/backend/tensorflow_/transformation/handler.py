from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
from tensorflow.keras.layers import Lambda, Concatenate

def get_handler(class_name):
    if class_name in LAYER_HANDLERS:
        return LAYER_HANDLERS[class_name]
    else:
        return LayerHandler

def cut(w, in_gate, out_gate):
    if out_gate is not None:
        out_gate = np.array(out_gate, dtype=np.bool)
        if len(w.shape) == 4: # conv2d
            w = w[:,:,:,out_gate]
        elif len(w.shape) == 2: # fc
            w = w[:,out_gate]
        elif len(w.shape) == 1: # bias ... 
            w = w[out_gate]
    if in_gate is not None:
        in_gate = np.array(in_gate, dtype=np.bool)
        if len(w.shape) == 4: # conv2d
            w = w[:,:,in_gate,:]
        elif len(w.shape) == 2: # fc
            w = w[in_gate,:]
    return w

class LayerHandler(object):

    def __init__(self):
        pass

    @staticmethod
    def is_transformer(tensor_idx):
        return False

    @staticmethod
    def is_concat():
        return False

    @staticmethod
    def get_output_modifier(name):
        return None

    @staticmethod
    def get_gate_modifier(name):
        return None

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        return

    @staticmethod
    def update_gate(gates, input_shape):
        return None

    @staticmethod
    def cut_weights(W, in_gate, out_gate):
        ret = []
        for w in W:
            w_ = cut(copy.deepcopy(w), in_gate, out_gate)
            ret.append(w_)
        return ret

class Conv2DHandler(LayerHandler):

    @staticmethod
    def is_transformer(tensor_idx):
        return True

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        layer_dict["config"]["filters"] = new_weights[0].shape[-1]
        return

class DenseHandler(LayerHandler):
    
    @staticmethod
    def is_transformer(tensor_idx):
        return True

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        layer_dict["config"]["units"] = new_weights[0].shape[-1]
        return

class ShiftHandler(LayerHandler):

    @staticmethod
    def is_transformer(tensor_idx):
        return False

    @staticmethod
    def get_output_modifier(name):
        """
            x[0] -> data
            x[1] -> mask
        """
        return Lambda(lambda x: x[0] * x[1], name=name)

class DWConv2DHandler(LayerHandler):

    @staticmethod
    def is_transformer(tensor_idx):
        return False

    @staticmethod
    def get_output_modifier(name):
        """
            x[0] -> data
            x[1] -> mask
        """
        return Lambda(lambda x: x[0] * x[1], name=name)

    @staticmethod
    def cut_weights(W, in_gate, out_gate):
        ret = []
        for w in W:
            if len(w.shape) == 4:
                w_ = cut(copy.deepcopy(w), out_gate, None)
            else:
                w_ = cut(copy.deepcopy(w), in_gate, out_gate)
            ret.append(w_)
        return ret

class ConcatHandler(LayerHandler):

    @staticmethod
    def is_transformer(tensor_idx):
        return False

    @staticmethod
    def get_gate_modifier(name):
        return Concatenate(axis=-1, name=name)

    @staticmethod
    def is_concat():
        return True

    @staticmethod
    def update_gate(gates, input_shape):
        return np.concatenate(gates)

class FlattenHandler(LayerHandler):

    @staticmethod
    def is_transformer(tensor_idx):
        return False

    @staticmethod
    def update_gate(gates, input_shape):
        shape = list(input_shape[1:]) # data_shape
        shape[-1] = 1 # not repeated at the last dim.
        return np.tile(gates, tuple(shape)).flatten()

class ReshapeHandler(LayerHandler):

    @staticmethod
    def is_transformer(tensor_idx):
        return False

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        val = int(np.sum(output_gate))
        layer_dict["config"]["target_shape"][-1] = val
        return

LAYER_HANDLERS = {
    "Conv2D": Conv2DHandler,
    "Dense": DenseHandler,
    "BatchNormalization": ShiftHandler,
    "DepthwiseConv2D": DWConv2DHandler,
    "Concatenate": ConcatHandler,
    "Flatten": FlattenHandler,
    "Reshape": ReshapeHandler
}
