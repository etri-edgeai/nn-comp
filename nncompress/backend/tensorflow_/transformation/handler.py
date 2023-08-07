""" Layer-level Handler """


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
from tensorflow.keras.layers import Lambda, Concatenate

def get_handler(class_name):
    """ Get handler for class_name """
    if class_name in LAYER_HANDLERS:
        return LAYER_HANDLERS[class_name]
    else:
        return LayerHandler

def cut(w, in_gate, out_gate):
    """ Weight Slicing """
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
    """ Base LayerHandler """

    def __init__(self):
        pass

    @staticmethod
    def is_transformer(tensor_idx):
        """ Is transformer? """
        return False

    @staticmethod
    def is_concat():
        """ Is concatenation? """
        return False

    @staticmethod
    def get_output_modifier(name):
        """ Get output modifier """
        return None

    @staticmethod
    def get_gate_modifier(name):
        """ Get gate modifier """
        return None

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        """ Update layer schema """
        return

    @staticmethod
    def update_gate(gates, input_shape):
        """ Update gate """
        return None

    @staticmethod
    def cut_weights(W, in_gate, out_gate):
        """ Weight slicing """
        ret = []
        for w in W:
            w_ = cut(copy.deepcopy(w), in_gate, out_gate)
            ret.append(w_)
        return ret

class Conv2DHandler(LayerHandler):
    """ Conv2D Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ Is transformer? """
        return True

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        """ Update layer schema """
        layer_dict["config"]["filters"] = new_weights[0].shape[-1]
        return

class PatchingAndEmbeddingHandler(LayerHandler):
    """ Handling ViT """

    @staticmethod
    def is_transformer(tensor_idx):
        """ Is transformer? """
        return True

    @staticmethod
    def cut_weights(W, in_gate, out_gate):
        """ Cut weights """
        ret = []
        for w in W:
            if len(w.shape) == 4:
                w_ = cut(copy.deepcopy(w), in_gate, out_gate)
            else:
                w_ = cut(copy.deepcopy(w), None, out_gate)
            ret.append(w_)
        return ret

class WeightedSumHandler(LayerHandler):
    """ WeightedSum Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ Is transformer? """
        return False

    @staticmethod
    def cut_weights(W, in_gate, out_gate):
        """ Weight Slicing """
        ret = []
        for w in W:
            ret.append(w)
        return ret

class DenseHandler(LayerHandler):
    """ Dense Handler """
    
    @staticmethod
    def is_transformer(tensor_idx):
        """ Is transformer? """
        return True

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        """ Update layer schema """
        layer_dict["config"]["units"] = new_weights[0].shape[-1]
        return

class ShiftHandler(LayerHandler):
    """ Shift(by addition)Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ Is transformer? """
        return False

    @staticmethod
    def get_output_modifier(name):
        """
            x[0] -> data
            x[1] -> mask
        """
        return Lambda(lambda x: x[0] * x[1], name=name)

class DWConv2DHandler(LayerHandler):
    """ Depth-wise Conv2D Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ Is transformer? """
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
        """ Weight Slicing """
        ret = []
        for w in W:
            if len(w.shape) == 4:
                w_ = cut(copy.deepcopy(w), out_gate, None)
            else:
                w_ = cut(copy.deepcopy(w), in_gate, out_gate)
            ret.append(w_)
        return ret


class MultiHeadAttentionHandler(LayerHandler):
    """ MultiHeadAttentionHandler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ Is transformer? """
        return True

    @staticmethod
    def cut_weights(W, in_gate, out_gate): # self-attention
        """ Weights slicing """
        ret = []
        for idx, w in enumerate(W):
            if len(w.shape) == 3:
                w_ = copy.deepcopy(w)
                if idx < 6: # k,q,v
                    w_ = w_[in_gate,:,:]
                else: # o 
                    w_ = w_[:,:,out_gate]
            elif idx == len(W)-1: # last bias
                w_ = copy.deepcopy(w)
                w_ = w_[out_gate]
            else:
                w_ = copy.deepcopy(w)
            ret.append(w_)
        return ret

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        """ Update layer schema """
        print(np.sum(output_gate), output_gate.shape[0], np.sum(input_gate))
        if np.sum(output_gate) != output_gate.shape[0]:
            print(np.sum(output_gate), output_gate.shape[0])
        layer_dict["config"]['output_shape'] = [new_weights[-1].shape[-1],]
        return

class SeparableConv2DHandler(LayerHandler):
    """ SeparableConv2DHandler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ Is transformer? """
        return True

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        """ Update layer schema """
        layer_dict["config"]["filters"] = new_weights[1].shape[-1]
        return

    @staticmethod
    def cut_weights(W, in_gate, out_gate):
        """ Weights slicing """
        ret = []
        for idx, w in enumerate(W):
            if idx == 0: # Depth-wise
                w_ = cut(copy.deepcopy(w), in_gate, None)
            else: # Point-wise
                w_ = cut(copy.deepcopy(w), in_gate, out_gate)
            ret.append(w_)
        return ret

class ConcatHandler(LayerHandler):
    """ Conatenation Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ Is transformer """
        return False

    @staticmethod
    def get_gate_modifier(name):
        """ Gate gate modifier """
        return Concatenate(axis=-1, name=name)

    @staticmethod
    def is_concat():
        """ Is concatenation? """
        return True

    @staticmethod
    def update_gate(gates, input_shape):
        """ Update gate values """
        return np.concatenate(gates)

class FlattenHandler(LayerHandler):
    """ Handling Flatten """

    @staticmethod
    def is_transformer(tensor_idx):
        """ Is transformer? """
        return False

    @staticmethod
    def update_gate(gates, input_shape):
        """ Update gate for flatten op """
        shape = list(input_shape[1:]) # data_shape
        shape[-1] = 1 # not repeated at the last dim.
        return np.tile(gates, tuple(shape)).flatten()

class ReshapeHandler(LayerHandler):
    """ Reshape Handler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ Is transformer? """
        return False

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        """ Update layer schema """
        val = int(np.sum(output_gate))
        layer_dict["config"]["target_shape"][-1] = val
        return

class InputLayerHandler(LayerHandler):
    """ InputLayerHandler """

    @staticmethod
    def is_transformer(tensor_idx):
        """ Is transformer? """
        return False

    @staticmethod
    def update_layer_schema(layer_dict, new_weights, input_gate, output_gate):
        """ Update layer schema """
        layer_dict["config"]["batch_input_shape"][-1] = int(np.sum(input_gate))
        return


LAYER_HANDLERS = {
    "Conv2D": Conv2DHandler,
    "Dense": DenseHandler,
    "BatchNormalization": ShiftHandler,
    "DepthwiseConv2D": DWConv2DHandler,
    "Concatenate": ConcatHandler,
    "Flatten": FlattenHandler,
    "Reshape": ReshapeHandler,
    "SeparableConv2D": SeparableConv2DHandler,
    "WeightedSum":WeightedSumHandler,
    "InputLayer":InputLayerHandler,
    "MultiHeadAttention":MultiHeadAttentionHandler,
    "keras_cv>PatchingAndEmbedding":PatchingAndEmbeddingHandler
}
