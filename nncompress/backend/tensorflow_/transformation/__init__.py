from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json

from tensorflow import keras

from nncompress.backend.tensorflow_.transformation.pruning_parser import StopGradientLayer

def parse(model, parser_class, name=None, **kwargs):
    parsers = {}
    if name is not None:
        parser = parser_class(model, basestr=name+"_", **kwargs)
    else:
        parser = parser_class(model, **kwargs)
    parser.parse()
    if name is None:
        parsers["root"] = parser
    else:
        parsers[name] = parser

    for layer in model.layers:
        if layer.__class__.__name__ == "Functional":
            parsers_ = parse(layer, parser_class, name=layer.name, **kwargs)
            for name_, parser_ in parsers_.items():
                parsers[name_] = parser_

    return parsers

def inject(parsers, name=None, avoid=None, with_splits=False):

    if name is None:
        parser = parsers["root"]
    else:
        parser = parsers[name]

    model = parser._model
    imodel, igate_mapping = parser.inject(avoid=avoid, with_mapping=True, with_splits=with_splits)
    imodel_dict = json.loads(imodel.to_json())
    weights = {layer.name:layer.get_weights() for layer in imodel.layers}

    for idx, layer in enumerate(model.layers):
        if layer.name in parsers:
            isub_model, isub_gate_mapping = inject(parsers, layer.name, avoid=avoid, with_splits=with_splits)
            isub_model_dict = json.loads(isub_model.to_json())

            imodel_dict["config"]["layers"][idx]["config"]["layers"] = isub_model_dict["config"]["layers"]
            weights[layer.name] = isub_model.get_weights()

            igate_mapping.update(isub_gate_mapping)

    model_json = json.dumps(imodel_dict)
    custom_objects = {parser._gate_class.__name__:parser._gate_class, "StopGradientLayer":StopGradientLayer}
    custom_objects.update(parser._custom_objects)
    ret = keras.models.model_from_json(model_json, custom_objects=custom_objects)
    for layer in model.layers:
        ret.get_layer(layer.name).set_weights(weights[layer.name])
    return ret, igate_mapping

def cut(parsers, gmodel, name=None):

    if name is None:
        parser = parsers["root"]
    else:
        parser = parsers[name]

    icmodel = parser.cut(gmodel)
    icmodel_dict = json.loads(icmodel.to_json()) 
    weights = {layer.name:layer.get_weights() for layer in icmodel.layers}
    for idx, layer in enumerate(icmodel.layers):
        if layer.name in parsers:
            gmodel_layer = gmodel.get_layer(layer.name)
            cmodel = cut(parsers, gmodel_layer, name=layer.name)
            
            cmodel_dict = json.loads(cmodel.to_json())
            icmodel_dict["config"]["layers"][idx]["config"]["layers"] = cmodel_dict["config"]["layers"]
            weights[layer.name] = cmodel.get_weights()
            
    model_json = json.dumps(icmodel_dict)
    ret = keras.models.model_from_json(model_json, custom_objects=parser._custom_objects)
    for layer in icmodel.layers:
        ret.get_layer(layer.name).set_weights(weights[layer.name])
    return ret


def unfold(model, custom_objects=None):

    if type(model) == keras.Sequential:
        input_layer = keras.layers.Input(batch_shape=model.layers[0].input_shape, name="seq_input")
        prev_layer = input_layer
        for layer in model.layers:
            layer._inbound_nodes = []
            prev_layer = layer(prev_layer)
        model = keras.models.Model([input_layer], [prev_layer])

    model_dict = json.loads(model.to_json())
    weights = {}
    for layer in model.layers:
        if layer.__class__.__name__ == "Functional":
            for sub_layer in layer.layers:
                weights[sub_layer.name] = sub_layer.get_weights()
        else:
            weights[layer.name] = layer.get_weights()

    layers_ = []
    output_mapping = {}
    new_layers = []
    input_mapping = {}
    layers_dict = {}
    for idx, layer in enumerate(model_dict["config"]["layers"]):
        layers_dict[layer["name"]] = layer
        if layer["class_name"] == "Functional": 
            for sub_layer in layer["config"]["layers"]:
                layers_.append(sub_layer)
            output_mapping[layer["name"]] = layer["config"]["output_layers"]
            layer_name = layer["name"]
            
            for old, new in zip(layer["config"]["input_layers"], layer["inbound_nodes"][0]):
                input_mapping[old[0]] = new[0]
        else:
            new_layers.append(layer)
    model_dict["config"]["layers"] = new_layers

    new_outputs = []
    for output in model_dict["config"]["output_layers"]:
        if layers_dict[output[0]]["class_name"] == "Functional":
            new_outputs.append(output_mapping[output[0]][0])
        else:
            new_outputs.append(output)
    model_dict["config"]["output_layers"] = new_outputs

    for layer in model_dict["config"]["layers"]:
        if layer["class_name"] != "Functional": 
            for inbound in layer["inbound_nodes"]:
                for in_ in inbound:
                    if in_[0] in output_mapping:
                        layer_name = in_[0]
                        in_[0] = output_mapping[layer_name][0][0] # layer_name
                        in_[1] = 0 # flow
                        in_[2] = output_mapping[layer_name][0][2] # tensor_idx

    for layer in layers_:
        if layer["name"] in input_mapping:
            continue
        for inbound in layer["inbound_nodes"]:
            for in_ in inbound:
                if in_[0] in input_mapping:
                    in_[0] = input_mapping[in_[0]]
        model_dict["config"]["layers"].append(layer)

    # unfold activation
    outb = {}
    layers = {}
    for layer in model_dict["config"]["layers"]:
        layers[layer["config"]["name"]] = layer
        for inbound in layer["inbound_nodes"]:
            for i in inbound:
                if i[0] not in outb:
                    outb[i[0]] = []
                outb[i[0]].append(layer["config"]["name"])

    output_layers = set()
    for l in model_dict["config"]["output_layers"]:
        output_layers.add(l[0])

    new_acts = []
    for layer in model_dict["config"]["layers"]:
        if layer["config"]["name"] in output_layers:
            continue

        if "Conv2D" in layer["class_name"] or "Dense" in layer["class_name"]:
            if layer["config"]["activation"] is not None and layer["config"]["activation"] != "linear":
                activation = layer["config"]["activation"]
                layer["config"]["activation"] = None

                dict_ = {
                    'class_name': 'Activation',
                    'name':layer["name"]+"_nnc_act",
                    'config': {
                        'name': layer["name"]+"_nnc_act",
                        'trainable': True,
                        'dtype': 'float32',
                        'activation': activation
                    }
                }

                # old's inbound to new
                for out_layer in outb[layer["config"]["name"]]:
                    for inbound in layers[out_layer]["inbound_nodes"]:
                        for in_ in inbound:
                            if in_[0] == layer["config"]["name"]:
                                in_[0] = dict_["config"]["name"]
               
                # inbound of olds's to new
                dict_["inbound_nodes"] = []
                for inbound in layer["inbound_nodes"]:
                    dict_["inbound_nodes"].append(
                        [[layer["config"]["name"], 0, 0]]
                    )
                new_acts.append(dict_)
    for d in new_acts:
        model_dict["config"]["layers"].append(d)

    model_json = json.dumps(model_dict)
    ret = keras.models.model_from_json(model_json, custom_objects=custom_objects)

    for layer in ret.layers:
        if layer.name not in weights:
            continue
        layer.set_weights(weights[layer.name])

    return ret    
