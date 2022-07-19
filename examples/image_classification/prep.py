import json

import tensorflow as tf

from dataloader.dataset_factory import *
from models.custom import GAModel

from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection

def change_dtype_(model_dict, policy, distill_set=None):

    float32 = set()
    for layer in model_dict["config"]["output_layers"]:
        float32.add(layer[0])

    if distill_set is None:
        distill_set = set()

    for layer in model_dict["config"]["layers"]:
        if layer["class_name"] == "InputLayer":
            layer["config"]["dtype"] = "float16"
            continue
        if layer["config"]["name"] in float32 or layer["config"]["name"] in distill_set:
            continue
        elif layer["class_name"] == "Activation":
            continue
        elif layer["class_name"] == "Functional":
            change_dtype_(layer, policy, distill_set=distill_set)
        elif layer["class_name"] == "TFOpLambda" or layer["class_name"] == "AddLoss":
            continue
        else:
            layer["config"]["dtype"] = {'class_name': 'Policy', 'config': {'name': 'mixed_float16'}}


def change_dtype(model_, policy, distill_set=None, custom_objects=None):
    model_backup = model_
    model_dict = json.loads(model_.to_json())
    change_dtype_(model_dict, policy, distill_set=distill_set)
    model_json = json.dumps(model_dict)
    model_ = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
    for layer in model_backup.layers:
        model_.get_layer(layer.name).set_weights(layer.get_weights())
    return model_


def get_custom_objects():
    custom_objects = {
        "SimplePruningGate":SimplePruningGate,
        "StopGradientLayer":StopGradientLayer
    }
    return custom_objects


def add_augmentation(model, image_size, train_batch_size=32, do_mixup=False, do_cutmix=False, custom_objects=None):

    def cond_mixing(args):
      images,mixup_weights,cutmix_masks,is_tr_split = args
      return tf.cond(tf.keras.backend.equal(is_tr_split[0],0), 
                     lambda: images, # eval phase
                     lambda: mixing_lite(images,mixup_weights,cutmix_masks, train_batch_size, do_mixup, do_cutmix)) # tr phase

    input_shape = (image_size, image_size, 3)  # Should handle any image size
    image_input = tf.keras.layers.Input(shape=input_shape, name="image")
    mixup_input = tf.keras.layers.Input(shape=(1, 1, 1), name="mixup_weight")
    cutmix_input = tf.keras.layers.Input(shape=(None, None, 1), name="cutmix_mask")
    is_tr_split = tf.keras.layers.Input(shape=(1), name="is_tr_split") # indicates whether we use tr or eval data loader
    inputs = [image_input,mixup_input,cutmix_input,is_tr_split]

    mixup_weights = inputs[1]
    cutmix_masks = inputs[2]
    is_tr_split = inputs[3]
    x = tf.keras.layers.Lambda(cond_mixing, name="input_lambda")([image_input,mixup_weights,cutmix_masks,is_tr_split])
          
    temp_model = tf.keras.Model(inputs=inputs, outputs=x)
    temp_model_dict = json.loads(temp_model.to_json())
    model_dict = json.loads(model.to_json())

    to_removed_names = []
    to_removed = []
    for layer in model_dict["config"]["layers"]:
        if layer["class_name"] == "InputLayer":
            to_removed_names.append(layer["config"]["name"])
            to_removed.append(layer)

    for layer in model_dict["config"]["layers"]:
        for inbound in layer["inbound_nodes"]:
            if type(inbound[0]) == str:
                if inbound[0] in to_removed_names:
                    inbound[0] = "input_lambda"
            else:
                for ib in inbound:
                    if ib[0] in to_removed_names: # input
                        ib[0] = "input_lambda"

    for r in to_removed:
        model_dict["config"]["layers"].remove(r)

    model_dict["config"]["layers"] +=  temp_model_dict["config"]["layers"]
    model_dict["config"]["input_layers"] = [[layer.name, 0, 0] for layer in inputs]

    model_json = json.dumps(model_dict)
    if custom_objects is None:
        custom_objects = {}
    model_ = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)

    for layer in model.layers:
        if layer.name in to_removed_names:
            continue
        model_.get_layer(layer.name).set_weights(layer.get_weights())

    return model_
